"""D0.7: every integrity / leakage / provenance check proven by a case that trips it (the test-coverage review,
2026-11-02: one test per surviving mutant). Tamperings keep every hash consistent, so only the LOGICAL check can find
them.
"""
import copy
import json
import os

import numpy as np
import pytest

from _d06 import make_session
from _d07 import CLASSES, make_synthetic_session
from _helpers import FRAME, rgb, stamp, sync
from trainit_data._canonical import canonical_json, pretty_json, sha256_bytes
from trainit_data.capture import Capture
from trainit_data.dataset import create_dataset
from trainit_data.dataset_integrity import verify_dataset
from trainit_data.errors import DataError
from trainit_data.ground_truth import create_synthetic_label_set
from trainit_data.integrity import verify_session
from trainit_data.labels import LabelSet, content_id, create_label_set, read_entries
from trainit_data.observation import InstanceImageBlock
from trainit_data.splits import SplitAssignment, _split_id, build_split, check_split_rules
from trainit_data.synthetic import frame_of, generation_block
from trainit_data.versions import DatasetVersion, content_hash_of, create_version
from trainit_data.writer import ImageSample


# ---- helpers: hash-consistent edits -----------------------------------------------------------------------------

def _writable(path):
    os.chmod(path, 0o644)
    return path


def rewrite_index(store, session_id, mutate):
    """Edit a sealed session's capture lines and re-seal the header's index hash."""
    index = _writable(store.path(f'sessions/{session_id}/captures.jsonl'))
    lines = [json.loads(x) for x in index.read_text().splitlines()]
    mutate(lines)
    data = b''.join(canonical_json(x) + b'\n' for x in lines)
    index.write_bytes(data)
    edit_header(store, session_id, lambda d: d['index'].update(sha256=sha256_bytes(data)))


def edit_header(store, session_id, mutate):
    path = _writable(store.path(f'sessions/{session_id}/session.json'))
    doc = json.loads(path.read_text())
    mutate(doc)
    path.write_text(json.dumps(doc))


def rewrite_version(store, dataset, number, *, selection=None, split=None, manifest=None):
    """Edit a version's selection / split / manifest, keeping the pins and the content hash consistent."""
    base = f'datasets/{dataset}/versions/{number}'
    man_path = _writable(store.path(f'{base}/manifest.json'))
    doc = json.loads(man_path.read_text())
    if selection is not None:
        path = _writable(store.path(f'{base}/selection.jsonl'))
        lines = [json.loads(x) for x in path.read_text().splitlines()]
        selection(lines)
        data = b''.join(canonical_json(x) + b'\n' for x in lines)
        path.write_bytes(data)
        doc['content']['selection']['sha256'] = sha256_bytes(data)
    if split is not None:
        path = _writable(store.path(f'{base}/splits.json'))
        record = json.loads(path.read_text())
        split(record)
        record['split_id'] = _split_id(record)
        data = pretty_json(record)
        path.write_bytes(data)
        doc['content']['split'].update(sha256=sha256_bytes(data), split_id=record['split_id'])
    if manifest is not None:
        manifest(doc)
    doc['content_hash'] = content_hash_of(doc)
    man_path.write_text(json.dumps(doc))


def details(report):
    return ' | '.join(f.detail for f in report.findings)


@pytest.fixture
def syn(store):
    header = make_synthetic_session(store, runs=4, views=2)
    create_dataset(store, 'parts', name='Parts')
    return store, header


# ---- split / manifest / selection cross-checks (leakage) -------------------------------------------------------

def test_a_split_whose_grouping_is_not_the_manifests_is_found(store, vocab):
    real = make_session(store, vocab, simulated=False)
    create_dataset(store, 'parts', name='Parts')
    create_version(store, 'parts', sessions=[real.session_id], group_by='randomization_run')   # groups = {session}

    def as_session_split(record):
        record.update(schema='trainit-split/1.0', group_by='session')
    rewrite_version(store, 'parts', 1, split=as_session_split)
    report = verify_dataset(store, 'parts')
    assert not report.ok and 'groups by session, the manifest by randomization_run' in details(report)


def test_a_split_that_forgets_a_run_is_found(syn):
    store, header = syn
    create_version(store, 'parts', sessions=[header.session_id], group_by='randomization_run')

    def drop_one(record):
        record['assignment'].pop(sorted(record['assignment'])[0])
    rewrite_version(store, 'parts', 1, split=drop_one)
    report = verify_dataset(store, 'parts')
    assert not report.ok and 'does not assign exactly the version\'s groups' in details(report)


@pytest.mark.parametrize('grouping, mutate', [
    ('randomization_run', lambda lines: lines[0].pop('group')),
    ('session', lambda lines: lines[0].update(group='rr-' + '0' * 16)),
])
def test_a_selection_line_records_its_group_exactly_when_the_grouping_is_not_session(store, vocab, grouping, mutate):
    real = make_session(store, vocab, simulated=False)
    create_dataset(store, 'parts', name='Parts')
    create_version(store, 'parts', sessions=[real.session_id], group_by=grouping)
    rewrite_version(store, 'parts', 1, selection=mutate)
    report = verify_dataset(store, 'parts')
    assert not report.ok and 'records its group exactly when' in details(report)


def test_an_inherited_split_never_changes_its_grouping():
    parent = build_split({'ses-20261102T000000Z-00000000': 2}, seed=1)
    with pytest.raises(DataError) as e:
        build_split({'ses-20261102T000000Z-00000000': 2}, seed=1, parent=parent, parent_version=1,
                    group_by='randomization_run')
    assert e.value.code == 'DERIVATION_INVALID'
    child = build_split({'rr-' + '1' * 16: 2}, seed=1, group_by='randomization_run')
    assert any('changes the group_by' in p for p in check_split_rules(child, parent))


def test_a_grouped_split_or_manifest_needs_the_1_1_schema(syn):
    store, header = syn
    version = create_version(store, 'parts', sessions=[header.session_id], group_by='randomization_run')
    split = json.loads(store.path('datasets/parts/versions/1/splits.json').read_text())
    split['schema'] = 'trainit-split/1.0'
    split['split_id'] = _split_id(split)
    with pytest.raises(DataError, match='needs trainit-split/1.1') as e:
        SplitAssignment.from_dict(split)
    assert e.value.code == 'INVALID_VALUE'
    doc = copy.deepcopy(version.to_dict())
    doc['schema'] = 'trainit-dataset/1.0'
    doc['content_hash'] = content_hash_of(doc)
    with pytest.raises(DataError, match='needs trainit-dataset/1.1'):
        DatasetVersion.from_dict(doc)


# ---- ground truth derivation refusals -----------------------------------------------------------------------------

def _manual_mask_session(store, vocab, *, image_sensor='cam', mask_sensor='cam', mask_size=None, provenance=None):
    """A non-generated session with a color image and an instance mask (built by hand, MANUAL trigger)."""
    sensors = {}
    sensors.setdefault(image_sensor, []).append({'stream': 'color', 'modality': 'rgb'})
    sensors.setdefault(mask_sensor, []).append({'stream': 'instances', 'modality': 'instance_segmentation'})
    writer = store.create_session(source={'transport': 'INGEST', 'backend': 'mock'},
                                  clock={'session_domain': 'ROS_TIME'}, provenance=provenance,
                                  sensors=[{'sensor_id': k, 'streams': v} for k, v in sensors.items()])
    at = stamp(100)
    h, w = mask_size or (6, 8)
    mask = np.zeros((h, w), dtype=np.uint16)
    mask[1:3, 1:4] = 1
    block = {'annotator': 'example_ids', 'box_annotator': 'example_boxes',
             'instances': [{'id': 1, 'object': 'o00', 'class': 'cube', 'pixels': 6, 'box_tight_xywh_px': [1, 1, 3, 2]}]}
    samples = [ImageSample(image_sensor, 'color', at, rgb(1), 'rgb8', FRAME),
               ImageSample(mask_sensor, 'instances', at, mask, '16UC1', FRAME, block_fields=block)]
    writer.append_capture(trigger={'trigger_id': 'manual', 'kind': 'MANUAL', 'firing_index': 0, 'origin': 'OPERATOR',
                                   'issued': {'stamp': at.to_dict()}}, outcome='COMPLETE',
                          sync=sync(reference=f'{image_sensor}/color'), samples=samples)
    return writer.close()


def test_a_session_without_a_generation_block_holds_no_synthetic_ground_truth(store, vocab):
    plain = _manual_mask_session(store, vocab)
    create_dataset(store, 'parts', name='Parts')
    with pytest.raises(DataError, match='no generation block') as e:
        create_synthetic_label_set(store, 'parts', [plain.session_id], 'BBOX_2D')
    assert e.value.code == 'LABEL_INVALID'


def test_a_mask_of_another_sensor_in_the_capture_is_refused(store, vocab):
    other = _manual_mask_session(store, vocab, image_sensor='cam', mask_sensor='other_cam')
    create_dataset(store, 'parts', name='Parts')
    with pytest.raises(DataError, match='not an instance mask of the camera') as e:
        create_synthetic_label_set(store, 'parts', [other.session_id], 'BBOX_2D')
    assert e.value.code == 'LABEL_INVALID'


def test_one_label_set_has_one_generator_and_one_annotator(syn):
    store, a = syn
    b = make_synthetic_session(store, runs=4, views=2)
    edit_header(store, b.session_id, lambda d: d['provenance']['randomization'].update(generator='other_generator'))
    with pytest.raises(DataError, match='different generators'):
        create_synthetic_label_set(store, 'parts', [a.session_id, b.session_id], 'BBOX_2D')
    c = make_synthetic_session(store, runs=4, views=2)

    def other_annotator(lines):
        for line in lines:
            line['observations'][2]['data']['box_annotator'] = 'other_boxes'
    rewrite_index(store, c.session_id, other_annotator)
    with pytest.raises(DataError, match='one annotator per label set'):
        create_synthetic_label_set(store, 'parts', [a.session_id, c.session_id], 'BBOX_2D')


def test_annotator_boxes_are_required_for_an_annotator_label_set(syn):
    store, header = syn

    def no_box(lines):
        lines[0]['observations'][2]['data']['instances'][0].pop('box_tight_xywh_px')
    rewrite_index(store, header.session_id, no_box)
    with pytest.raises(DataError, match='has no annotator box') as e:
        create_synthetic_label_set(store, 'parts', [header.session_id], 'BBOX_2D')
    assert e.value.code == 'LABEL_INVALID'
    derived = create_synthetic_label_set(store, 'parts', [header.session_id], 'BBOX_2D', box_source='MASK_EXTENT')
    assert derived.entries['count'] == 8


def test_a_synthetic_label_set_that_leaves_an_image_out_is_found(syn):
    store, header = syn
    full = create_synthetic_label_set(store, 'parts', [header.session_id], 'BBOX_2D')
    create_label_set(store, 'parts', kind='BBOX_2D', classes=full.classes, provenance=dict(full.provenance),
                     entries=read_entries(store, full)[1:])
    report = verify_dataset(store, 'parts')
    assert not report.ok and 'are not labelled' in details(report)


# ---- the INSTANCE_MASK reader --------------------------------------------------------------------------------------

def _mask_entry(header_or_obs):
    color, inst = header_or_obs
    return {'observation_id': color.observation_id, 'mask_observation_id': inst.observation_id,
            'instances': [{'id': i['id'], 'class': i['class']} for i in inst.data.instances]}


def _create_mask(store, entry, classes=CLASSES):
    return create_label_set(store, 'parts', kind='INSTANCE_MASK', classes=classes, entries=[entry],
                            provenance={'source': 'HUMAN', 'tool': 'example'})


def test_a_mask_of_another_size_than_its_image_is_refused(store, vocab):
    session = _manual_mask_session(store, vocab, mask_size=(6, 10))
    create_dataset(store, 'parts', name='Parts')
    capture = next(store.open_session(session.session_id).captures())
    with pytest.raises(DataError, match='does not match the 8x6 image') as e:
        _create_mask(store, _mask_entry(capture.observations), classes=['cube'])
    assert e.value.code == 'LABEL_INVALID'


def test_a_mask_whose_pixels_contradict_its_block_is_refused_by_the_label_path_and_found_by_verify(syn):
    store, header = syn

    def more_pixels(lines):
        lines[0]['observations'][2]['data']['instances'][0]['pixels'] -= 1
    rewrite_index(store, header.session_id, more_pixels)
    report = verify_session(store, header.session_id)
    assert any(f.code == 'PAYLOAD_INVALID' and 'pixels' in f.detail for f in report.findings)
    capture = next(store.open_session(header.session_id).captures())
    with pytest.raises(DataError, match='pixels') as e:
        _create_mask(store, _mask_entry((capture.observations[0], capture.observations[2])))
    assert e.value.code == 'LABEL_INVALID'


def test_a_mask_entry_never_labels_the_mask_itself(syn):
    store, header = syn
    inst = next(store.open_session(header.session_id).captures()).observations[2]
    entry = _mask_entry((inst, inst))
    with pytest.raises(DataError, match='another observation than the labelled image'):
        _create_mask(store, entry)


def test_provenance_and_schema_minor_rules_of_label_sets(syn):
    store, header = syn
    with pytest.raises(DataError, match='derivation names the annotator'):
        create_label_set(store, 'parts', kind='BBOX_2D', classes=CLASSES, entries=[],
                         provenance={'source': 'IMPORTED', 'derivation': {'rule': 'mask_extent/1'}})
    for kind in ('INSTANCE_MASK', 'BBOX_2D'):
        ls = create_synthetic_label_set(store, 'parts', [header.session_id], kind)
        record = ls.to_dict()
        record['schema'] = 'trainit-labels/1.0'
        content = {k: v for k, v in record.items() if k not in ('label_set_id', 'created')}
        record['label_set_id'] = content_id(content)
        with pytest.raises(DataError, match='need trainit-labels/1.1'):
            LabelSet.from_dict(record)


# ---- synthetic session rules (verify) --------------------------------------------------------------------------------

def test_a_capture_no_generator_fired_inside_a_generation_is_found(syn):
    store, header = syn

    def manual(lines):
        trigger = lines[1]['trigger']
        trigger.update(kind='MANUAL', origin='OPERATOR')
        trigger.pop('context'), trigger.pop('reason')
    rewrite_index(store, header.session_id, manual)
    assert 'is not a generated frame' in details(verify_session(store, header.session_id))


def test_generated_frames_in_a_session_without_a_generation_block_are_found(syn):
    store, header = syn
    edit_header(store, header.session_id, lambda d: d['provenance'].pop('randomization'))
    assert 'session with no generation block' in details(verify_session(store, header.session_id))


@pytest.mark.parametrize('mutate, needle', [
    (lambda lines: lines[1]['trigger']['context']['trainit.randomization'].update(run_index=1), 'two run'),
    (lambda lines: lines[1]['trigger']['context']['trainit.randomization'].update(view=0), 'appears twice'),
])
def test_one_run_one_index_and_each_view_once(syn, mutate, needle):
    store, header = syn
    rewrite_index(store, header.session_id, mutate)
    report = verify_session(store, header.session_id)
    assert not report.ok and needle in details(report)


def test_more_views_than_declared_are_found(syn):
    store, header = syn
    edit_header(store, header.session_id, lambda d: d['provenance']['randomization'].update(views_per_run=1))
    assert 'outside the declared' in details(verify_session(store, header.session_id))


def test_the_randomization_context_belongs_to_generated_triggers_only(syn):
    store, header = syn
    reader = store.open_session(header.session_id)
    doc = next(reader.captures()).to_dict()
    manual = copy.deepcopy(doc)
    manual['trigger'].update(kind='MANUAL', origin='OPERATOR')
    with pytest.raises(DataError, match='only a SYNTHETIC_FRAME capture'):
        frame_of(Capture.from_dict(manual, reader.vocab))
    bare = copy.deepcopy(doc)
    bare['trigger'].pop('context')
    with pytest.raises(DataError, match='carries its trainit.randomization context'):
        frame_of(Capture.from_dict(bare, reader.vocab))


def test_rendered_pixels_are_never_claimed_bit_identical(syn):
    store, header = syn
    block = copy.deepcopy(store.open_session(header.session_id).header.provenance['randomization'])
    block['reproducibility']['render_bit_identical'] = True
    with pytest.raises(DataError, match='never promised bit-identical'):
        generation_block(block)


# ---- the instance_image block's own rules ------------------------------------------------------------------------

def _block(**over):
    doc = {'block': 'instance_image', 'version': '1.0', 'width': 8, 'height': 6, 'encoding': '16UC1',
           'frame_id': FRAME, 'annotator': 'example_ids', 'box_annotator': 'example_boxes',
           'instances': [{'id': 1, 'object': 'a', 'class': 'cube', 'pixels': 4, 'box_tight_xywh_px': [0, 0, 2, 2]},
                         {'id': 2, 'object': 'b', 'class': 'cube', 'pixels': 1}]}
    doc.update(over)
    return doc


@pytest.mark.parametrize('change, needle', [
    (lambda d: d['instances'][0].update(id=70000), 'at most 65535'),
    (lambda d: d['instances'][0].update(box_tight_xywh_px=[7, 0, 2, 2]), 'inside the 8x6 image'),
    (lambda d: d['instances'][0].update(pixels=5), 'do not fit'),
    (lambda d: d['instances'][0].update(occlusion=1.5), 'within'),
    (lambda d: d.pop('box_annotator'), 'name their box_annotator'),
    (lambda d: d.update(encoding='32FC1'), 'not allowed'),
    (lambda d: d['instances'][1].update(id=3), 'compact'),
    (lambda d: d['instances'][1].update(object='a'), 'unique object keys'),
])
def test_the_instance_block_refuses_what_it_cannot_mean(vocab, change, needle):
    doc = _block()
    change(doc)
    with pytest.raises(DataError, match=needle) as e:
        InstanceImageBlock.from_dict(doc, vocab.modalities['instance_segmentation'],
                                     vocab.data_blocks['instance_image'])
    assert e.value.code == 'INVALID_VALUE'


@pytest.mark.parametrize('extra', [{'unit_scale_m': 1.0}, {'transforms': ({'op': 'x'},)}])
def test_image_only_fields_on_another_block_are_refused_before_any_byte(store, vocab, extra):
    writer = store.create_session(source={'transport': 'INGEST', 'backend': 'mock'},
                                  clock={'session_domain': 'ROS_TIME'},
                                  sensors=[{'sensor_id': 'cam', 'streams': [{'stream': 'instances',
                                                                             'modality': 'instance_segmentation'}]}])
    at = stamp(100)
    mask = np.zeros((6, 8), np.uint16)
    sample = ImageSample('cam', 'instances', at, mask, '16UC1', FRAME, block_fields={'annotator': 'x', 'instances': []},
                         **extra)
    with pytest.raises(DataError, match='belong to the image block'):
        writer.append_capture(trigger={'trigger_id': 'manual', 'kind': 'MANUAL', 'firing_index': 0,
                                       'origin': 'OPERATOR', 'issued': {'stamp': at.to_dict()}},
                              outcome='COMPLETE', sync=sync('SINGLE_STREAM', None, reference='cam/instances'),
                              samples=[sample])
    writer.close()


# ---- second mutation pass (each check alone) ------------------------------------------------------------------------

def _rewrite_run(store, header, run_index, *, draws, run):
    def mutate(lines):
        for line in lines:
            context = line['trigger']['context']['trainit.randomization']
            if context['run_index'] == run_index:
                line['trigger']['reason']['draws'] = draws
                context['run'] = run
    rewrite_index(store, header.session_id, mutate)


def test_two_run_indices_that_drew_the_same_scene_are_found(syn):
    store, header = syn
    first = next(store.open_session(header.session_id).captures())
    context, reason = frame_of(first)
    _rewrite_run(store, header, 1, draws=reason['draws'], run=context['run'])      # run 1 now holds run 0's scene
    report = verify_session(store, header.session_id)
    assert not report.ok and 'has two run indices' in details(report) and 'holds two runs' not in details(report)


def test_one_run_index_that_holds_two_scenes_is_found(syn):
    from trainit_data.synthetic import run_id
    store, header = syn

    def second_view_other_scene(lines):
        def at(x):
            context = x['trigger']['context']['trainit.randomization']
            return (context['run_index'], context['view'])
        line = next(x for x in lines if at(x) == (1, 1))
        reason = line['trigger']['reason']
        reason['draws'] = dict(reason['draws'], light_intensity=reason['draws']['light_intensity'] + 1)
        line['trigger']['context']['trainit.randomization']['run'] = run_id(reason['recipe'], reason['draws'])
    rewrite_index(store, header.session_id, second_view_other_scene)
    report = verify_session(store, header.session_id)
    assert not report.ok and 'holds two runs' in details(report) and 'has two run indices' not in details(report)


def test_verify_finds_generated_captures_in_a_version_grouped_by_session(syn):
    store, header = syn
    create_version(store, 'parts', sessions=[header.session_id], group_by='randomization_run')

    def to_session_split(record):
        record.update(schema='trainit-split/1.0', group_by='session', assignment={header.session_id: 'train'},
                      overrides={}, counts={'train': {'groups': 1, 'captures': 8},
                                            'val': {'groups': 0, 'captures': 0},
                                            'test': {'groups': 0, 'captures': 0}})

    def to_session_manifest(doc):
        doc['schema'] = 'trainit-dataset/1.0'
        doc['content']['split'].pop('group_by')
        doc['content']['split']['counts'] = {'train': {'groups': 1, 'captures': 8},
                                             'val': {'groups': 0, 'captures': 0}, 'test': {'groups': 0, 'captures': 0}}
    rewrite_version(store, 'parts', 1, selection=lambda lines: [x.pop('group') for x in lines],
                    split=to_session_split, manifest=to_session_manifest)
    report = verify_dataset(store, 'parts')
    assert not report.ok and 'holds generated captures in a version grouped by session' in details(report)


def test_every_camera_rotation_gets_one_canonical_quaternion():
    from trainit_data.adapters.isaac_replicator import optical_pose_from_camera_params
    from _d07 import camera_params
    rng = np.random.default_rng(0)
    for _ in range(200):
        q, _ = np.linalg.qr(rng.normal(size=(3, 3)))
        q *= np.sign(np.linalg.det(q))                              # a rotation
        view = np.eye(4)
        view[:3, :3] = q
        _, rotation = optical_pose_from_camera_params(camera_params(cameraViewTransform=view.T.reshape(-1).tolist()))
        lead = rotation[3] if rotation[3] != 0.0 else next(v for v in rotation[:3] if v != 0.0)
        assert lead > 0.0, rotation
