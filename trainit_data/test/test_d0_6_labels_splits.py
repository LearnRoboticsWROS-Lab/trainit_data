"""D0.6: label sets (a layer above acquisition, immutable, content-addressed) and group-aware splits."""
import json
import os

import pytest

from _d06 import CLASSES, HEIGHT, WIDTH, box, color_obs, make_session
from _helpers import open_session, snapshot
from trainit_data.dataset import create_dataset
from trainit_data.dataset_integrity import verify_dataset
from trainit_data.errors import DataError
from trainit_data.labels import (EDGE_TOLERANCE, LabelSet, create_label_set, list_label_sets, load_label_set,
                                 read_entries)
from trainit_data.splits import SplitAssignment, bucket_of, build_split, check_split_rules, hashed_split


@pytest.fixture
def cell(store, vocab):
    sim = make_session(store, vocab, simulated=True)
    real = make_session(store, vocab, simulated=False, seed=10)
    create_dataset(store, 'parts', name='Parts')
    return store, sim, real


def bbox(store, entries, kind='BBOX_2D', source='IMPORTED', **kw):
    provenance = {'source': source, **kw.pop('provenance', {})}
    return create_label_set(store, 'parts', kind=kind, classes=CLASSES, provenance=provenance, entries=entries, **kw)


# ---- label sets -------------------------------------------------------------------------------------------

def test_labels_never_touch_the_sessions_and_are_content_addressed(cell):
    store, sim, real = cell
    sessions_before = snapshot(store.path('sessions'))
    entries = [{'observation_id': color_obs(real, 2), 'boxes': [box('tray', 0, 0, WIDTH, HEIGHT)]},
               {'observation_id': color_obs(sim, 1), 'boxes': [box(), box('tray', 5, 2, 2, 2)]},
               {'observation_id': color_obs(sim, 3), 'boxes': []}]                      # an explicit negative
    first = bbox(store, entries)
    again = bbox(store, list(reversed(entries)))                                         # same content, other order
    assert again.label_set_id == first.label_set_id and list_label_sets(store, 'parts') == [first.label_set_id]
    assert snapshot(store.path('sessions')) == sessions_before
    stored = read_entries(store, first)
    assert [e['observation_id'] for e in stored] == sorted(e['observation_id'] for e in entries)
    assert first.entries['count'] == 3 and first.kind == 'BBOX_2D' and first.classes == tuple(CLASSES)
    for name in ('labelset.json', 'entries.jsonl'):
        path = store.path(f'datasets/parts/labels/{first.label_set_id}/{name}')
        assert not os.access(path, os.W_OK) or os.geteuid() == 0


def test_a_class_label_set(cell):
    store, sim, _ = cell
    ls = create_label_set(store, 'parts', kind='CLASS', classes=CLASSES, provenance={'source': 'HUMAN', 'author': 'qa'},
                          entries=[{'observation_id': color_obs(sim, 2), 'class': 'tray'}])
    assert read_entries(store, ls) == [{'observation_id': color_obs(sim, 2), 'class': 'tray'}]


@pytest.mark.parametrize('bad_box', [
    box(x=-0.5),                                   # left of the image
    box(x=5.0, w=4.0),                             # 5 + 4 > 8
    box(y=4.0, h=3.0),                             # 4 + 3 > 6
    box(w=0.0),
    box(h=-1.0),
])
def test_a_box_outside_its_image_is_refused_never_clipped(cell, bad_box):
    store, sim, _ = cell
    with pytest.raises(DataError) as e:
        bbox(store, [{'observation_id': color_obs(sim, 1), 'boxes': [bad_box]}])
    assert e.value.code == 'LABEL_INVALID'
    assert list_label_sets(store, 'parts') == []


def test_the_edge_tolerance_only_absorbs_rounding_and_the_value_is_stored_as_given(cell):
    store, sim, _ = cell
    tiny = EDGE_TOLERANCE * WIDTH / 2
    ls = bbox(store, [{'observation_id': color_obs(sim, 1), 'boxes': [box(x=-tiny, w=WIDTH)]}])
    assert read_entries(store, ls)[0]['boxes'][0]['bbox_xywh_px'][0] == -tiny


@pytest.mark.parametrize('entry, code', [
    ({'observation_id': 'ses-20000101T000000Z-00000000/000001/overhead_camera/color', 'boxes': []},
     'SESSION_NOT_FOUND'),
    ({'observation_id': 'SEQ/1', 'boxes': []}, 'INVALID_VALUE'),
    ({'boxes': []}, 'MISSING_FIELD'),
    ({'observation_id': None, 'boxes': [], 'extra': 1}, 'INVALID_VALUE'),
])
def test_an_entry_must_name_an_observation(cell, entry, code):
    store, _, _ = cell
    with pytest.raises(DataError) as e:
        bbox(store, [entry])
    assert e.value.code == code


def test_an_unknown_observation_class_or_a_double_label_is_refused(cell):
    store, sim, _ = cell
    missing = color_obs(sim, 1).replace('/000001/', '/000099/')
    for entries, code in (([{'observation_id': missing, 'boxes': []}], 'OBSERVATION_NOT_FOUND'),
                          ([{'observation_id': color_obs(sim, 1), 'boxes': [box('screw')]}], 'LABEL_INVALID'),
                          ([{'observation_id': color_obs(sim, 1), 'boxes': []}] * 2, 'LABEL_INVALID')):
        with pytest.raises(DataError) as e:
            bbox(store, entries)
        assert e.value.code == code


def test_labels_on_an_active_session_are_refused(cell):
    store, _, _ = cell
    writer = open_session(store)
    with pytest.raises(DataError) as e:
        bbox(store, [{'observation_id': f'{writer.session_id}/000001/overhead_camera/color', 'boxes': []}])
    assert e.value.code == 'SESSION_NOT_SEALED'
    writer.close()


def test_provenance_rules(cell):
    store, sim, _ = cell
    entry = [{'observation_id': color_obs(sim, 1), 'boxes': [box()]}]
    with pytest.raises(DataError, match='reserved') as e:
        bbox(store, entry, source='MODEL_PREDICTION')
    assert e.value.code == 'INVALID_VALUE'
    with pytest.raises(DataError, match='generator'):
        bbox(store, entry, source='SYNTHETIC_GROUND_TRUTH')
    with pytest.raises(DataError, match='annotator'):                      # D0.7: synthetic labels name their annotator
        bbox(store, entry, source='SYNTHETIC_GROUND_TRUTH', provenance={'generator': 'example_generator'})
    ls = bbox(store, entry, source='SYNTHETIC_GROUND_TRUTH', provenance={'generator': 'example_generator',
                                                                         'annotator': 'example_boxes'})
    assert ls.provenance == {'source': 'SYNTHETIC_GROUND_TRUTH', 'generator': 'example_generator',
                             'annotator': 'example_boxes'}
    with pytest.raises(DataError) as e:
        bbox(store, entry, kind='INSTANCE_MASK')
    assert e.value.code == 'MISSING_FIELD'                        # D0.7: read; an entry names its mask


def test_supersedes_names_an_existing_label_set_of_the_same_kind(cell):
    store, sim, _ = cell
    first = bbox(store, [{'observation_id': color_obs(sim, 1), 'boxes': [box()]}])
    with pytest.raises(DataError) as e:
        bbox(store, [{'observation_id': color_obs(sim, 1), 'boxes': []}], supersedes='ls-0000000000000000')
    assert e.value.code == 'LABEL_SET_NOT_FOUND'
    with pytest.raises(DataError):
        create_label_set(store, 'parts', kind='CLASS', classes=CLASSES, provenance={'source': 'HUMAN'},
                         entries=[{'observation_id': color_obs(sim, 1), 'class': 'part'}],
                         supersedes=first.label_set_id)


def test_a_label_set_record_is_strict_and_its_id_is_its_content(cell):
    store, sim, _ = cell
    ls = bbox(store, [{'observation_id': color_obs(sim, 1), 'boxes': [box()]}])
    doc = ls.to_dict()
    for mutate, code in ((lambda d: d.update(unknown=1), 'UNKNOWN_FIELD'),
                         (lambda d: d['provenance'].update(source='HUMAN'), 'HASH_MISMATCH'),
                         (lambda d: d['ontology'].update(classes=['part']), 'INVALID_VALUE')):
        bad = json.loads(json.dumps(doc))
        mutate(bad)
        with pytest.raises(DataError) as e:
            LabelSet.from_dict(bad)
        assert e.value.code == code


def test_tampered_entries_are_detected(cell):
    store, sim, _ = cell
    ls = bbox(store, [{'observation_id': color_obs(sim, 1), 'boxes': [box()]}])
    path = store.path(f'datasets/parts/labels/{ls.label_set_id}/entries.jsonl')
    os.chmod(path, 0o644)
    path.write_bytes(path.read_bytes().replace(b'4.0', b'7.5'))      # a box moved outside the image
    with pytest.raises(DataError) as e:
        read_entries(store, load_label_set(store, 'parts', ls.label_set_id))
    assert e.value.code == 'HASH_MISMATCH'
    assert not verify_dataset(store, 'parts').ok


# ---- splits -----------------------------------------------------------------------------------------------

def test_the_bucket_function_is_pinned():
    """A literal pin (computed independently with hashlib when written): the split of a group must not change
    across Python versions or releases."""
    assert bucket_of(0, 'g') == 2764
    assert bucket_of(20261102, 'ses-20261029T091917Z-fd274eee') == 2769
    assert [bucket_of(0, f'g{i}') for i in range(6)] == [8614, 6263, 419, 1584, 681, 1075]
    assert [hashed_split(0, {'train': 0.8, 'val': 0.1, 'test': 0.1}, f'g{i}') for i in range(6)] == \
        ['val', 'train', 'train', 'train', 'train', 'train']


def test_every_group_is_in_exactly_one_split_and_adding_groups_moves_none():
    groups = {f'ses-20260101T0000{i:02d}Z-0000000{i % 10}': 3 for i in range(40)}
    first = build_split(dict(list(groups.items())[:20]), seed=5)
    more = build_split(groups, seed=5)
    assert all(more.split_of(g) == first.split_of(g) for g in first.assignment)
    counts = more.record['counts']
    assert sum(c['groups'] for c in counts.values()) == 40 and sum(c['captures'] for c in counts.values()) == 120
    assert set(more.assignment.values()) <= {'train', 'val', 'test'}


def test_overrides_are_recorded_and_inheritance_keeps_the_parents_split():
    parent = build_split({'a': 1, 'b': 1}, seed=1, overrides={'a': 'test'})
    child = build_split({'a': 1, 'b': 1, 'c': 4}, seed=1, overrides={'c': 'val'}, parent=parent, parent_version=1)
    assert child.record['overrides'] == {'a': 'test', 'c': 'val'} and child.split_of('a') == 'test'
    assert child.record['inherited'] == {'version': 1, 'split_id': parent.split_id}
    assert check_split_rules(child, parent) == []
    with pytest.raises(DataError) as e:
        build_split({'a': 1}, seed=2, parent=parent, parent_version=1)
    assert e.value.code == 'DERIVATION_INVALID'
    with pytest.raises(DataError) as e:
        build_split({'a': 1, 'b': 1}, seed=1, overrides={'a': 'train'}, parent=parent, parent_version=1)
    assert e.value.code == 'DERIVATION_INVALID'


@pytest.mark.parametrize('ratios', [{'train': 0.8, 'val': 0.1, 'test': 0.2}, {'train': 1.2, 'val': -0.2, 'test': 0.0},
                                    {'train': 1.0, 'val': 0.0}])
def test_ratios_are_validated(ratios):
    with pytest.raises(DataError):
        build_split({'a': 1}, seed=0, ratios=ratios)


def test_a_hand_moved_group_is_a_leak_verify_reports():
    split = build_split({'a': 1, 'b': 1, 'c': 1}, seed=3)
    record = split.to_dict()
    moved = next(g for g, v in record['assignment'].items() if v != 'test')
    record['assignment'] = dict(record['assignment'], **{moved: 'test'})
    from trainit_data.splits import _split_id
    record['split_id'] = _split_id(record)
    record['counts'] = {k: {'groups': sum(v == k for v in record['assignment'].values()), 'captures':
                            sum(v == k for v in record['assignment'].values())} for k in ('train', 'val', 'test')}
    tampered = SplitAssignment.from_dict(record)
    assert check_split_rules(tampered) == [f'group {moved} is in test, the rule gives {split.split_of(moved)}']
