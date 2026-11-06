"""D0.7: synthetic ground truth -> canonical label sets; randomization-run split groups; a mixed real + synthetic
version through the SAME exporter; the D0-G comparison; the CLI.
"""
import json
import os
import re
from pathlib import Path

import pytest

from _d06 import CLASSES as REAL_CLASSES, color_obs, make_session
from _d07 import CLASSES, make_synthetic_session
from trainit_data._canonical import canonical_json, sha256_bytes
from trainit_data.cli import main
from trainit_data.dataset import create_dataset
from trainit_data.dataset_integrity import verify_dataset
from trainit_data.errors import DataError
from trainit_data.exports import read_sidecar, verify_export
from trainit_data.formats import get_exporter
from trainit_data.ground_truth import create_synthetic_label_set
from trainit_data.labels import create_label_set, read_entries
from trainit_data.splits import hashed_split
from trainit_data.synthetic import compare_generations, frame_of
from trainit_data.versions import capture_splits, create_version, load_split, read_selection

YOLO = get_exporter('yolo')
MIXED = CLASSES + REAL_CLASSES                                   # one ontology for a real + synthetic version


@pytest.fixture
def syn(store):
    header = make_synthetic_session(store, runs=6, views=2)
    create_dataset(store, 'parts', name='Parts')
    return store, header


def _captures(store, header):
    return list(store.open_session(header.session_id).captures())


# ---- synthetic label sets -----------------------------------------------------------------------------------------

def test_the_three_synthetic_label_sets_carry_their_provenance_and_never_look_human(syn):
    store, header = syn
    mask = create_synthetic_label_set(store, 'parts', [header.session_id], 'INSTANCE_MASK')
    direct = create_synthetic_label_set(store, 'parts', [header.session_id], 'BBOX_2D')
    derived = create_synthetic_label_set(store, 'parts', [header.session_id], 'BBOX_2D', box_source='MASK_EXTENT')
    for ls in (mask, direct, derived):
        assert ls.schema == 'trainit-labels/1.1' and ls.provenance['source'] == 'SYNTHETIC_GROUND_TRUTH'
        assert ls.provenance['generator'] == 'isaac_replicator'
        assert ls.provenance['source_ref'] == f'sessions {header.session_id}'
        assert ls.entries['count'] == 12 and ls.classes == tuple(sorted(ls.classes))
    assert mask.provenance['annotator'] == 'instance_id_segmentation' and 'derivation' not in mask.provenance
    assert direct.provenance['annotator'] == 'bounding_box_2d_tight' and 'derivation' not in direct.provenance
    assert derived.provenance == {**derived.provenance, 'annotator': 'instance_id_segmentation',
                                  'derivation': {'rule': 'mask_extent/1'}}
    assert len({mask.label_set_id, direct.label_set_id, derived.label_set_id}) == 3
    # the annotator's boxes and the masks' extents are the same geometry here (within the 1 px rule: equal)
    assert read_entries(store, direct) == read_entries(store, derived)
    # the same derivation is the same (content-addressed) label set
    again = create_synthetic_label_set(store, 'parts', [header.session_id], 'INSTANCE_MASK')
    assert again.label_set_id == mask.label_set_id
    assert verify_dataset(store, 'parts').ok


def test_mask_entries_name_the_mask_of_the_same_capture_and_classify_every_id(syn):
    store, header = syn
    mask = create_synthetic_label_set(store, 'parts', [header.session_id], 'INSTANCE_MASK')
    for entry, capture in zip(read_entries(store, mask), _captures(store, header)):
        color, _, inst = capture.observations
        assert entry['observation_id'] == color.observation_id
        assert entry['mask_observation_id'] == inst.observation_id
        assert entry['instances'] == [{'id': i['id'], 'class': i['class']} for i in inst.data.instances]


@pytest.mark.parametrize('change, needle', [
    (lambda e: e.update(mask_observation_id=e['observation_id'].rsplit('/', 1)[0] + '/depth'), 'instance mask'),
    (lambda e: e['instances'].pop(), 'classifies ids'),
    (lambda e: e.update(mask_observation_id=re.sub(r'/0000(\d\d)/', '/000099/', e['mask_observation_id'])),
     'capture of the labelled image'),
    (lambda e: e['instances'].reverse(), 'sorted'),
])
def test_the_instance_mask_reader_refuses_a_wrong_mask_entry(syn, change, needle):
    store, header = syn
    capture = _captures(store, header)[0]
    color, _, inst = capture.observations
    entry = {'observation_id': color.observation_id, 'mask_observation_id': inst.observation_id,
             'instances': [{'id': i['id'], 'class': i['class']} for i in inst.data.instances]}
    change(entry)
    with pytest.raises(DataError, match=needle):
        create_label_set(store, 'parts', kind='INSTANCE_MASK', classes=CLASSES,
                         provenance={'source': 'SYNTHETIC_GROUND_TRUTH', 'generator': 'x', 'annotator': 'y'},
                         entries=[entry])


def test_a_mask_whose_payload_changed_is_refused(syn):
    store, header = syn
    inst = _captures(store, header)[0].observations[2]
    path = store.path(inst.payload.path)
    os.chmod(path, 0o644)
    path.write_bytes(path.read_bytes() + b'\0')
    with pytest.raises(DataError) as e:
        create_synthetic_label_set(store, 'parts', [header.session_id], 'INSTANCE_MASK')
    assert e.value.code == 'HASH_MISMATCH'


def test_the_ontology_may_be_an_ordered_superset_and_never_a_subset(syn):
    store, header = syn
    ls = create_synthetic_label_set(store, 'parts', [header.session_id], 'BBOX_2D', classes=MIXED)
    assert ls.classes == tuple(MIXED)
    with pytest.raises(DataError, match='not in the given ontology'):
        create_synthetic_label_set(store, 'parts', [header.session_id], 'BBOX_2D', classes=['cube'])


def test_a_non_synthetic_session_has_no_ground_truth_to_derive(store, vocab):
    real = make_session(store, vocab, simulated=False)
    create_dataset(store, 'parts', name='Parts')
    with pytest.raises(DataError, match='holds both'):
        create_synthetic_label_set(store, 'parts', [real.session_id], 'BBOX_2D')


def test_verify_rederives_a_synthetic_label_set_and_finds_an_edited_one(syn):
    """A label set CLAIMING synthetic ground truth from an annotator, but with other boxes, is created (every box is
    a valid box) and then found by verify, which re-derives it from the recorded annotator output."""
    store, header = syn
    direct = create_synthetic_label_set(store, 'parts', [header.session_id], 'BBOX_2D')
    entries = read_entries(store, direct)
    entries[0] = dict(entries[0], boxes=[dict(entries[0]['boxes'][0], bbox_xywh_px=[0.0, 0.0, 2.0, 2.0])]
                      + entries[0]['boxes'][1:])
    forged = create_label_set(store, 'parts', kind='BBOX_2D', classes=direct.classes, provenance=dict(
        direct.provenance), entries=entries)
    report = verify_dataset(store, 'parts')
    bad = [f for f in report.findings if f.code == 'LABEL_INVALID']
    assert not report.ok and len(bad) == 1 and forged.label_set_id in bad[0].artifact
    assert 'not what the recorded ground truth gives' in bad[0].detail
    # a derived set that claims the annotator (or the reverse) is found too
    derived = create_synthetic_label_set(store, 'parts', [header.session_id], 'BBOX_2D', box_source='MASK_EXTENT')
    fake = dict(derived.provenance, annotator='bounding_box_2d_tight')
    create_label_set(store, 'parts', kind='BBOX_2D', classes=derived.classes, provenance=fake,
                     entries=read_entries(store, derived))
    assert any('provenance.annotator' in f.detail for f in verify_dataset(store, 'parts').findings)


# ---- randomization runs as split groups -----------------------------------------------------------------------------

def test_a_version_grouped_by_randomization_run_never_splits_a_run(syn):
    store, header = syn
    version = create_version(store, 'parts', sessions=[header.session_id], group_by='randomization_run', seed=3,
                             ratios={'train': 0.5, 'val': 0.5, 'test': 0.0})
    assert version.schema == 'trainit-dataset/1.1' and version.group_by == 'randomization_run'
    split = load_split(store, version)
    assert split.record['schema'] == 'trainit-split/1.1' and split.group_by == 'randomization_run'
    runs = {}
    for capture in _captures(store, header):
        runs.setdefault(frame_of(capture)[0]['run'], []).append(capture.capture_id)
    assert sorted(split.assignment) == sorted(runs) and len(runs) == 6
    by_capture = capture_splits(store, version)
    for run, captures in runs.items():
        assert {by_capture[c] for c in captures} == {split.split_of(run)}           # no leakage across views
        assert split.split_of(run) == hashed_split(3, split.record['ratios'], run)
    assert {x['group'] for x in read_selection(store, version)} == set(runs)
    assert sum(n['groups'] for n in version.content['split']['counts'].values()) == 6
    assert len(set(split.assignment.values())) == 2                                  # both splits used
    assert verify_dataset(store, 'parts').ok


def test_generated_captures_are_never_grouped_by_session(syn, vocab):
    """A session group would let a regenerated scene (the same run in another session) sit in two splits."""
    store, header = syn
    duplicate = make_synthetic_session(store, runs=6, views=2)                   # the same runs, regenerated
    with pytest.raises(DataError, match='randomization_run') as e:
        create_version(store, 'parts', sessions=[header.session_id, duplicate.session_id])
    assert e.value.code == 'INVALID_VALUE'
    v = create_version(store, 'parts', sessions=[header.session_id, duplicate.session_id],
                       group_by='randomization_run')
    split, by_capture = load_split(store, v), capture_splits(store, v)
    assert len(split.assignment) == 6                                             # 6 scenes, not 2 sessions
    for capture in _captures(store, header):
        twin = capture.capture_id.replace(header.session_id, duplicate.session_id)
        assert by_capture[capture.capture_id] == by_capture[twin]                  # a scene and its regeneration
    assert verify_dataset(store, 'parts').ok


def test_a_version_of_live_sessions_grouped_by_session_is_the_d0_6_record(store, vocab):
    real = make_session(store, vocab, simulated=False)
    create_dataset(store, 'parts', name='Parts')
    version = create_version(store, 'parts', sessions=[real.session_id])
    assert version.schema == 'trainit-dataset/1.0' and 'group_by' not in version.content['split']
    assert load_split(store, version).record['schema'] == 'trainit-split/1.0'
    assert all('group' not in x for x in read_selection(store, version))


def test_the_grouping_is_inherited_and_changing_it_is_a_resplit(syn, vocab):
    store, header = syn
    real = make_session(store, vocab, simulated=False)
    create_version(store, 'parts', sessions=[header.session_id], group_by='randomization_run')
    v2 = create_version(store, 'parts', op='ADD_SESSIONS', parent=1, sessions=[real.session_id])
    assert v2.group_by == 'randomization_run' and real.session_id in load_split(store, v2).assignment
    with pytest.raises(DataError) as e:
        create_version(store, 'parts', op='ADD_SESSIONS', parent=1, sessions=[real.session_id], group_by='session')
    assert e.value.code == 'DERIVATION_INVALID'
    with pytest.raises(DataError, match='randomization_run'):                    # never by session: synthetic
        create_version(store, 'parts', op='RESPLIT', parent=2, group_by='session')
    v3 = create_version(store, 'parts', op='RESPLIT', parent=2, seed=99)
    assert v3.group_by == 'randomization_run' and v3.content['split']['split_id'] != v2.content['split']['split_id']
    v4 = create_version(store, 'parts', op='FILTER', parent=3, remove_sessions=[header.session_id])
    v5 = create_version(store, 'parts', op='RESPLIT', parent=4, group_by='session')    # live only: allowed
    assert v5.group_by == 'session' and list(load_split(store, v5).assignment) == [real.session_id]
    assert v4.group_by == 'randomization_run' and verify_dataset(store, 'parts').ok


def test_real_sessions_keep_their_session_group_and_the_same_bucket_in_a_run_grouped_version(syn, vocab):
    store, header = syn
    real = make_session(store, vocab, simulated=False)
    v = create_version(store, 'parts', sessions=[header.session_id, real.session_id], group_by='randomization_run',
                       seed=11)
    split = load_split(store, v)
    groups = {x['group'] for x in read_selection(store, v) if x['capture_id'].startswith(real.session_id)}
    assert groups == {real.session_id}
    d06 = create_version(store, 'parts', sessions=[real.session_id], seed=11)
    assert split.split_of(real.session_id) == load_split(store, d06).split_of(real.session_id)


def _rewrite_selection(store, number, mutate):
    rel = f'datasets/parts/versions/{number}'
    sel = store.path(f'{rel}/selection.jsonl')
    lines = [json.loads(x) for x in sel.read_text().splitlines()]
    mutate(lines)
    data = b''.join(canonical_json(x) + b'\n' for x in lines)
    os.chmod(sel, 0o644)
    sel.write_bytes(data)
    man = store.path(f'{rel}/manifest.json')
    doc = json.loads(man.read_text())
    doc['content']['selection']['sha256'] = sha256_bytes(data)
    from trainit_data.versions import content_hash_of
    doc['content_hash'] = content_hash_of(doc)
    os.chmod(man, 0o644)
    man.write_text(json.dumps(doc))


def test_verify_finds_a_capture_moved_to_another_group_even_with_consistent_hashes(syn):
    """Moving one view of a run into another run's group would leak the scene across splits."""
    store, header = syn
    create_version(store, 'parts', sessions=[header.session_id], group_by='randomization_run')
    _rewrite_selection(store, 1, lambda lines: lines[1].update(group=lines[2]['group']))
    report = verify_dataset(store, 'parts')
    assert not report.ok and any('with their groups' in f.detail for f in report.findings)


def test_a_generated_capture_with_a_forged_run_id_cannot_be_grouped(syn):
    store, header = syn
    capture = _captures(store, header)[0]
    forged = capture.to_dict()
    forged['trigger']['context']['trainit.randomization']['run'] = 'rr-' + '0' * 16
    from trainit_data.capture import Capture
    with pytest.raises(DataError, match='not the id of the recorded draws'):
        frame_of(Capture.from_dict(forged, store.open_session(header.session_id).vocab))


# ---- one mixed real + synthetic version, the same exporter ----------------------------------------------------------

@pytest.fixture
def mixed(syn, vocab, tmp_path):
    store, header = syn
    real = make_session(store, vocab, simulated=False)
    synthetic = create_synthetic_label_set(store, 'parts', [header.session_id], 'BBOX_2D', classes=MIXED)
    imported = create_label_set(store, 'parts', kind='BBOX_2D', classes=MIXED, provenance={'source': 'IMPORTED'},
                                entries=[{'observation_id': color_obs(real, i), 'boxes': [
                                    {'class': 'part', 'bbox_xywh_px': [1.0, 1.0, 4.0, 3.0]}]} for i in (1, 2, 3)])
    version = create_version(store, 'parts', sessions=[header.session_id, real.session_id],
                             label_sets=[synthetic.label_set_id, imported.label_set_id], group_by='randomization_run',
                             seed=5, ratios={'train': 0.5, 'val': 0.25, 'test': 0.25})
    return store, header, real, synthetic, imported, version, tmp_path


def test_a_mixed_version_exports_through_the_same_exporter_with_every_image_in_its_group_split(mixed):
    store, header, real, synthetic, imported, version, tmp = mixed
    with pytest.raises(DataError, match='mixed provenance'):
        YOLO.export(store, 'parts', 1, tmp / 'refused')
    out = tmp / 'out'
    sidecar = YOLO.export(store, 'parts', 1, out, {'allow_mixed_provenance': True})
    assert sidecar['summary']['classes'] == MIXED
    assert {x['source'] for x in sidecar['source']['label_sets']} == {'SYNTHETIC_GROUND_TRUTH', 'IMPORTED'}
    assert sidecar['source']['content_hash'] == version.content_hash
    by_capture = capture_splits(store, version)
    images = [f for f in sidecar['files'] if f['path'].startswith('images/')]
    assert len(images) == 12 + 3
    for f in images:
        split = f['path'].split('/')[1]
        assert by_capture[f['observation_id'].rsplit('/', 2)[0]] == split
    assert verify_export(out, store).ok
    data_yaml = (out / 'data.yaml').read_text()
    assert "  0: 'cube'" in data_yaml and "  3: 'part'" in data_yaml and 'path:' not in data_yaml
    again = YOLO.export(store, 'parts', 1, tmp / 'again', {'allow_mixed_provenance': True})
    assert again == sidecar
    for f in sidecar['files']:
        assert (out / f['path']).read_bytes() == (tmp / 'again' / f['path']).read_bytes()
    assert verify_dataset(store, 'parts').ok


def test_provenance_stays_recoverable_from_an_export(mixed):
    store, header, real, synthetic, imported, version, tmp = mixed
    out = tmp / 'out'
    YOLO.export(store, 'parts', 1, out, {'allow_mixed_provenance': True})
    sidecar = read_sidecar(out)
    sources = {}
    for f in sidecar['files']:
        if 'observation_id' in f:
            session = store.open_session(f['observation_id'].split('/')[0]).header
            sources[(session.source.transport, session.source.backend)] = sources.get(
                (session.source.transport, session.source.backend), 0) + 1
    assert sources == {('INGEST', 'isaac'): 12, ('LIVE_ROS', 'real'): 3}


def test_neither_the_exporter_nor_the_dataset_code_branches_on_a_source():
    root = Path(__file__).resolve().parents[1] / 'trainit_data'
    for rel in ('formats/yolo.py', 'versions.py', 'splits.py', 'exports.py', 'labels.py', 'ground_truth.py'):
        code = '\n'.join(line for line in (root / rel).read_text().splitlines()
                         if not line.lstrip().startswith('#'))
        body = re.sub(r'"""(?:.|\n)*?"""', '', code)
        body = re.sub(r'#.*', '', body).lower()
        for token in ("'isaac'", "'real'", "'ingest'", "'live_ros'", 'replicator', 'd455', 'realsense'):
            assert token not in body, (rel, token)


# ---- D0-G ---------------------------------------------------------------------------------------------------------

def test_two_generations_with_the_same_config_and_seed_record_the_same_generation(store):
    a = make_synthetic_session(store, runs=3, views=2)
    b = make_synthetic_session(store, runs=3, views=2, raw_offset=300)          # Replicator renumbered
    report = compare_generations(store, a.session_id, b.session_id)
    assert report['reproducible'] and all(report['checks'].values())
    assert report['frames'] == {'a': 6, 'b': 6} and report['runs'] == 3
    assert report['render']['promised'] is False and report['environment']['same']
    c = make_synthetic_session(store, runs=3, views=2, master_seed=8)
    other = compare_generations(store, a.session_id, c.session_id)
    assert not other['reproducible'] and not other['checks']['draws'] and not other['checks']['config_sha256']


# ---- CLI ------------------------------------------------------------------------------------------------------------

def _cli(store, capsys, *argv):
    code = main(['--root', str(store.root), '--json', *argv])
    return code, json.loads(capsys.readouterr().out)


def test_the_cli_runs_the_synthetic_path(store, capsys, tmp_path):
    a = make_synthetic_session(store, runs=4, views=2)
    b = make_synthetic_session(store, runs=4, views=2)
    code, out = _cli(store, capsys, 'compare-generation', a.session_id, b.session_id)
    assert code == 0 and out['command'] == 'compare-generation' and out['result']['reproducible']
    assert _cli(store, capsys, 'dataset', 'create', 'syn', '--name', 'Syn')[0] == 0
    code, out = _cli(store, capsys, 'labels', 'from-synthetic', 'syn', '--session', a.session_id, '--kind', 'BBOX_2D')
    assert code == 0 and out['command'] == 'labels from-synthetic'
    ls = out['result']['label_set']
    assert ls['provenance']['source'] == 'SYNTHETIC_GROUND_TRUTH' and ls['schema'] == 'trainit-labels/1.1'
    code, out = _cli(store, capsys, 'labels', 'from-synthetic', 'syn', '--session', a.session_id, '--kind',
                     'BBOX_2D', '--box-source', 'MASK_EXTENT')
    assert code == 0 and out['result']['label_set']['provenance']['derivation'] == {'rule': 'mask_extent/1'}
    code, out = _cli(store, capsys, 'version', 'create', 'syn', '--session', a.session_id, '--label-set',
                     ls['label_set_id'], '--group-by', 'randomization_run')
    assert code == 0 and out['result']['version']['content']['split']['group_by'] == 'randomization_run'
    code, out = _cli(store, capsys, 'export', 'yolo', 'syn', '1', str(tmp_path / 'out'))
    assert code == 0 and out['result']['export']['summary']['splits']['train']['images'] >= 0
    code, out = _cli(store, capsys, 'verify', '--dataset', 'syn')
    assert code == 0 and out['ok']
    code, out = _cli(store, capsys, 'labels', 'from-synthetic', 'syn', '--session', a.session_id, '--kind',
                     'BBOX_2D', '--class', 'cube')
    assert code == 1 and out['error']['code'] == 'LABEL_INVALID'
    assert main(['--root', str(store.root), 'compare-generation', a.session_id, b.session_id]) == 0
    assert 'REPRODUCIBLE' in capsys.readouterr().out


# ---- the vocabulary grew additively: a session pinned to the D0.6 vocabulary still reads, verifies, exports ---------

D06_VOCAB = 'sha256:44bd4c404e8c511946a84c75b0d5c0ec94c07b861c83b65b9fe2d49829046955'


def _d06_vocabulary(vocab):
    """The shipped vocabulary minus the rows D0.7 and D0.8 added: exactly the D0.1-D0.6 vocabulary (its pinned
    hash)."""
    from trainit_data.vocab import Vocabulary
    doc = vocab.to_dict()
    del doc['modalities']['instance_segmentation'], doc['data_blocks']['instance_image']
    del doc['sync_policies']['SAME_RENDER_TICK']
    del doc['sync_policies']['FIRST_ALIGNED_AT_OR_AFTER_BOUNDARY']            # D0.8, owner decision 2026-11-05
    for category, row in (('trigger_kinds', 'SYNTHETIC_FRAME'), ('trigger_origins', 'GENERATOR'),
                          ('context_namespaces', 'trainit.randomization')):
        doc[category].remove(row)
    return Vocabulary.from_dict(doc)


def test_a_session_pinned_to_the_d0_6_vocabulary_still_reads_verifies_and_mixes_with_synthetic_data(syn, vocab,
                                                                                                    tmp_path):
    from _helpers import FRAME as REAL_FRAME, calibration, rgb, stamp, sync
    from trainit_data.integrity import verify_session, verify_store
    from trainit_data.writer import ImageSample
    store, header = syn
    old = _d06_vocabulary(vocab)
    assert old.sha256 == D06_VOCAB and vocab.sha256 != D06_VOCAB
    cal = store.put_calibration(calibration(old, 'color', sensor_id='bench_camera'))
    writer = store.create_session(source={'transport': 'LIVE_ROS', 'backend': 'real'},
                                  clock={'session_domain': 'ROS_TIME'},
                                  sensors=[{'sensor_id': 'bench_camera', 'streams': [{'stream': 'color',
                                                                                      'modality': 'rgb'}]}],
                                  vocabulary=old)
    at = stamp(100)
    writer.append_capture(trigger={'trigger_id': 'manual', 'kind': 'MANUAL', 'firing_index': 0, 'origin': 'OPERATOR',
                                   'issued': {'stamp': at.to_dict()}}, outcome='COMPLETE',
                          sync=sync('SINGLE_STREAM', None, reference='bench_camera/color'),
                          samples=[ImageSample('bench_camera', 'color', at, rgb(1), 'rgb8', REAL_FRAME,
                                               calibration_id=cal)])
    real = writer.close()
    assert real.vocab['sha256'] == D06_VOCAB and store.open_session(real.session_id).vocab.sha256 == D06_VOCAB
    assert verify_session(store, real.session_id).ok
    synthetic = create_synthetic_label_set(store, 'parts', [header.session_id], 'BBOX_2D', classes=MIXED)
    imported = create_label_set(store, 'parts', kind='BBOX_2D', classes=MIXED, provenance={'source': 'IMPORTED'},
                                entries=[{'observation_id': f'{real.session_id}/000001/bench_camera/color',
                                          'boxes': [{'class': 'part', 'bbox_xywh_px': [1.0, 1.0, 4.0, 3.0]}]}])
    create_version(store, 'parts', sessions=[header.session_id, real.session_id], group_by='randomization_run',
                   label_sets=[synthetic.label_set_id, imported.label_set_id])
    sidecar = YOLO.export(store, 'parts', 1, tmp_path / 'out', {'allow_mixed_provenance': True})
    assert sum(n['images'] for n in sidecar['summary']['splits'].values()) == 13
    assert verify_dataset(store, 'parts').ok and verify_store(store).ok


# ---- review findings (2026-11-02) ----------------------------------------------------------------------------------

def test_a_synthetic_mask_label_set_with_another_class_than_the_ground_truth_is_found(syn):
    store, header = syn
    mask = create_synthetic_label_set(store, 'parts', [header.session_id], 'INSTANCE_MASK')
    entries = read_entries(store, mask)
    first = entries[0]['instances'][0]
    first['class'] = next(c for c in CLASSES if c != first['class'])
    create_label_set(store, 'parts', kind='INSTANCE_MASK', classes=mask.classes, provenance=dict(mask.provenance),
                     entries=entries)
    report = verify_dataset(store, 'parts')
    assert not report.ok and any('not what the recorded ground truth gives' in f.detail for f in report.findings)


def test_a_mask_of_another_camera_is_refused(syn):
    store, header = syn
    color, _, inst = _captures(store, header)[0].observations
    entry = {'observation_id': color.observation_id,
             'mask_observation_id': inst.observation_id.replace('/bin_camera/', '/side_camera/'),
             'instances': [{'id': i['id'], 'class': i['class']} for i in inst.data.instances]}
    with pytest.raises(DataError, match='camera of the labelled image'):
        create_label_set(store, 'parts', kind='INSTANCE_MASK', classes=CLASSES,
                         provenance={'source': 'SYNTHETIC_GROUND_TRUTH', 'generator': 'x', 'annotator': 'y'},
                         entries=[entry])


def test_d0_g_compares_two_different_generations_with_frames(store, vocab):
    a = make_synthetic_session(store, runs=2, views=1)
    real = make_session(store, vocab, simulated=False)
    with pytest.raises(DataError, match='not a session with itself'):
        compare_generations(store, a.session_id, a.session_id)
    with pytest.raises(DataError, match='not a synthetic generation'):
        compare_generations(store, a.session_id, real.session_id)
    empty = make_synthetic_session(store, runs=2, views=1, record_runs=0)
    with pytest.raises(DataError, match='no frame'):
        compare_generations(store, a.session_id, empty.session_id)
    fewer = make_synthetic_session(store, runs=2, views=1, record_runs=1)
    report = compare_generations(store, a.session_id, fewer.session_id)
    assert not report['reproducible'] and not report['checks']['run_structure']
    code = main(['--root', str(store.root), '--json', 'compare-generation', a.session_id, real.session_id])
    assert code == 1
