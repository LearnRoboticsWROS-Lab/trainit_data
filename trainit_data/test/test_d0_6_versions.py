"""D0.6: datasets, immutable versions, derivations, rights, content-kind neutrality (D0-E, D0-H in miniature).

Every test builds on two sealed sessions of DIFFERENT sources (simulated and physical, ``_d06.py``).
"""
import json
import os
import shutil
from pathlib import Path

import pytest

from _d06 import CLASSES, box, color_obs, make_session
from _helpers import snapshot
from trainit_data import _canonical, versions as V
from trainit_data.dataset import combine_rights, create_dataset, load_dataset
from trainit_data.dataset_integrity import verify_dataset
from trainit_data.errors import DataError
from trainit_data.integrity import verify_store
from trainit_data.labels import create_label_set
from trainit_data.store import Store
from trainit_data.versions import create_version, list_versions, load_split, load_version, read_selection

PKG = Path(__file__).resolve().parents[1] / 'trainit_data'


@pytest.fixture
def cell(store, vocab):
    sim = make_session(store, vocab, simulated=True)
    real = make_session(store, vocab, simulated=False, seed=10)
    create_dataset(store, 'parts', name='Parts')
    return store, sim, real


def labels(store, entries, source='IMPORTED', **kw):
    return create_label_set(store, 'parts', kind='BBOX_2D', classes=CLASSES, provenance={'source': source},
                            entries=entries, **kw)


def code_of(excinfo):
    return excinfo.value.code


# ---- dataset identity -------------------------------------------------------------------------------------

def test_a_dataset_identity_is_immutable_and_defaults_to_internal(cell):
    store, _, _ = cell
    identity = load_dataset(store, 'parts')
    assert identity.rights == {'classification': 'INTERNAL'} and identity.name == 'Parts'
    with pytest.raises(DataError) as e:
        create_dataset(store, 'parts', name='Other')
    assert code_of(e) == 'DATASET_EXISTS'
    path = store.path('datasets/parts/dataset.json')
    assert not os.access(path, os.W_OK) or os.geteuid() == 0     # read-only mode: an extra guard


def test_the_identity_is_content_kind_neutral_and_strict(cell):
    store, _, _ = cell
    record = json.loads(store.path('datasets/parts/dataset.json').read_text())
    assert set(record) == {'schema', 'dataset_id', 'name', 'rights', 'created'}   # nothing image- or YOLO-specific
    with pytest.raises(DataError) as e:
        create_dataset(store, 'Bad Id', name='x')
    assert code_of(e) == 'INVALID_VALUE'


# ---- versions: selection, sealed sessions, determinism ----------------------------------------------------

def test_a_version_selects_complete_captures_of_sealed_sessions_by_default(cell):
    store, sim, real = cell
    v = create_version(store, 'parts', sessions=[real.session_id, sim.session_id])
    sel = v.content['selection']
    assert sel['filter'] == {'include_outcomes': ['COMPLETE']}
    assert sel['captures'] == 6 and sel['excluded'] == {'COMPLETE': 0, 'FAILED': 0, 'PARTIAL': 2}
    assert [x['session_id'] for x in v.content['sessions']] == sorted([sim.session_id, real.session_id])
    lines = read_selection(store, v)
    assert {x['outcome'] for x in lines} == {'COMPLETE'} and len(lines) == 6
    assert v.derivation == {'op': 'CREATE'} and list_versions(store, 'parts') == [1]


def test_the_outcome_filter_is_explicit(cell):
    store, sim, real = cell
    v = create_version(store, 'parts', sessions=[sim.session_id], include_outcomes=['PARTIAL', 'COMPLETE'])
    assert v.content['selection']['captures'] == 4
    assert v.content['selection']['filter']['include_outcomes'] == ['COMPLETE', 'PARTIAL']


def test_an_active_session_is_refused(cell, vocab):
    store, sim, _ = cell
    from _helpers import open_session
    writer = open_session(store)
    with pytest.raises(DataError) as e:
        create_version(store, 'parts', sessions=[sim.session_id, writer.session_id])
    assert code_of(e) == 'SESSION_NOT_SEALED'
    assert list_versions(store, 'parts') == []
    writer.close()


def test_a_session_whose_sealed_index_changed_is_refused(cell):
    store, sim, _ = cell
    index = store.path(f'sessions/{sim.session_id}/captures.jsonl')
    index.write_bytes(index.read_bytes() + b'\n')
    with pytest.raises(DataError) as e:
        create_version(store, 'parts', sessions=[sim.session_id])
    assert code_of(e) == 'HASH_MISMATCH'


def test_the_content_hash_is_a_deterministic_function_of_the_content(cell, tmp_path, monkeypatch):
    store, sim, real = cell
    twin_root = tmp_path / 'twin'
    shutil.copytree(store.root, twin_root)
    v1 = create_version(store, 'parts', sessions=[sim.session_id, real.session_id], seed=7)
    monkeypatch.setattr(V, 'created_now', lambda: {'utc': '2030-01-01T00:00:00Z', 'by': 'someone else'})
    twin = Store.open(twin_root)
    v1b = create_version(twin, 'parts', sessions=[real.session_id, sim.session_id], seed=7)   # other order, time
    assert v1.content_hash == v1b.content_hash
    for name in ('selection.jsonl', 'splits.json'):
        assert store.path(f'datasets/parts/versions/1/{name}').read_bytes() == \
            twin.path(f'datasets/parts/versions/1/{name}').read_bytes()
    raw = json.loads(store.path('datasets/parts/versions/1/manifest.json').read_text())
    assert raw['content_hash'] == _canonical.sha256_json(
        {k: v for k, v in raw.items() if k not in ('content_hash', 'created', 'software')})


def test_a_version_is_never_overwritten(cell):
    store, sim, _ = cell
    create_version(store, 'parts', sessions=[sim.session_id])
    before = snapshot(store.root)
    with pytest.raises(DataError) as e:
        create_version(store, 'parts', sessions=[sim.session_id], version=1)
    assert code_of(e) == 'VERSION_EXISTS'
    assert snapshot(store.root) == before
    store.path('datasets/parts/versions/2').mkdir()                 # an interrupted creation holds number 2
    assert create_version(store, 'parts', sessions=[sim.session_id]).version == 3
    report = verify_dataset(store, 'parts')
    assert report.ok and any(f.severity == 'WARNING' and f.artifact.endswith('/versions/2') for f in report.findings)


def test_version_files_are_read_only_and_tampering_is_detected(cell):
    store, sim, real = cell
    create_version(store, 'parts', sessions=[sim.session_id, real.session_id])
    base = store.path('datasets/parts/versions/1')
    for name in ('manifest.json', 'selection.jsonl', 'splits.json'):
        assert not os.access(base / name, os.W_OK) or os.geteuid() == 0
    sel = base / 'selection.jsonl'
    os.chmod(sel, 0o644)
    sel.write_bytes(b''.join(sel.read_bytes().splitlines(keepends=True)[:-1]))      # drop a capture
    report = verify_dataset(store, 'parts')
    assert not report.ok and {f.code for f in report.findings} == {'HASH_MISMATCH'}
    manifest = base / 'manifest.json'
    os.chmod(manifest, 0o644)
    doc = json.loads(manifest.read_text())
    doc['content']['selection']['captures'] -= 1
    doc['content']['sessions'][0]['captures_selected'] -= 1
    manifest.write_text(json.dumps(doc))
    with pytest.raises(DataError) as e:
        load_version(store, 'parts', 1)
    assert code_of(e) == 'HASH_MISMATCH'
    assert not verify_store(store).ok


def test_verify_re_checks_what_a_version_pins_against_the_sessions(cell):
    store, sim, real = cell
    create_version(store, 'parts', sessions=[sim.session_id, real.session_id])
    assert verify_dataset(store, 'parts').ok and verify_store(store).ok
    payload = next(store.path(f'sessions/{real.session_id}/payload').rglob('*.png'))
    payload.write_bytes(payload.read_bytes() + b'x')                  # a session payload corrupted later
    report = verify_dataset(store, 'parts')
    assert not report.ok and any(real.session_id in f.detail for f in report.findings)


# ---- derivations ------------------------------------------------------------------------------------------

def test_relabel_is_a_new_version_and_the_old_one_is_unchanged(cell):
    """D0-E: a correction is a new label set + a new version; the parent's bytes and hash never change."""
    store, sim, real = cell
    first = labels(store, [{'observation_id': color_obs(sim, 1), 'boxes': [box()]}])
    v1 = create_version(store, 'parts', sessions=[sim.session_id, real.session_id], label_sets=[first.label_set_id])
    v1_bytes = snapshot(store.path('datasets/parts/versions/1'))
    fixed = labels(store, [{'observation_id': color_obs(sim, 1), 'boxes': [box(w=5.0)]}],
                   supersedes=first.label_set_id)
    assert fixed.label_set_id != first.label_set_id and fixed.supersedes == first.label_set_id
    v2 = create_version(store, 'parts', op='RELABEL', parent=1, label_sets=[fixed.label_set_id],
                        drop_label_sets=[first.label_set_id])
    assert v2.derivation == {'op': 'RELABEL', 'parent': {'version': 1, 'content_hash': v1.content_hash}}
    assert v2.label_set_ids == [fixed.label_set_id]
    assert v2.content['split']['split_id'] != '' and load_split(store, v2).record['inherited']['version'] == 1
    assert snapshot(store.path('datasets/parts/versions/1')) == v1_bytes
    assert load_version(store, 'parts', 1).content_hash == v1.content_hash
    assert verify_dataset(store, 'parts').ok


def test_add_sessions_inherits_the_split_and_never_moves_a_group(cell, vocab):
    store, sim, real = cell
    create_version(store, 'parts', sessions=[sim.session_id], overrides={sim.session_id: 'val'})
    v2 = create_version(store, 'parts', op='ADD_SESSIONS', parent=1, sessions=[real.session_id])
    split = load_split(store, v2)
    assert split.split_of(sim.session_id) == 'val' and split.record['overrides'] == {sim.session_id: 'val'}
    parent_split = load_split(store, load_version(store, 'parts', 1))
    assert split.record['inherited'] == {'version': 1, 'split_id': parent_split.split_id}
    third = make_session(store, vocab, simulated=True, seed=30)
    with pytest.raises(DataError) as e:
        create_version(store, 'parts', op='ADD_SESSIONS', parent=2, sessions=[third.session_id],
                       overrides={sim.session_id: 'train'})
    assert code_of(e) == 'DERIVATION_INVALID'
    with pytest.raises(DataError) as e:
        create_version(store, 'parts', op='ADD_SESSIONS', parent=2, sessions=[sim.session_id])
    assert code_of(e) == 'DERIVATION_INVALID'
    assert verify_dataset(store, 'parts').ok


def test_filter_only_narrows(cell):
    store, sim, real = cell
    create_version(store, 'parts', sessions=[sim.session_id, real.session_id], include_outcomes=['COMPLETE', 'PARTIAL'])
    v2 = create_version(store, 'parts', op='FILTER', parent=1, include_outcomes=['COMPLETE'])
    assert v2.content['selection']['captures'] == 6
    v3 = create_version(store, 'parts', op='FILTER', parent=2, remove_sessions=[real.session_id])
    assert v3.session_ids == [sim.session_id] and list(load_split(store, v3).assignment) == [sim.session_id]
    for kwargs in ({'include_outcomes': ['COMPLETE', 'FAILED']}, {}, {'remove_sessions': [sim.session_id]}):
        with pytest.raises(DataError) as e:
            create_version(store, 'parts', op='FILTER', parent=3, **kwargs)
        assert code_of(e) == 'DERIVATION_INVALID'
    assert verify_dataset(store, 'parts').ok


def test_resplit_changes_only_the_split(cell):
    store, sim, real = cell
    v1 = create_version(store, 'parts', sessions=[sim.session_id, real.session_id],
                        overrides={sim.session_id: 'train', real.session_id: 'train'})
    v2 = create_version(store, 'parts', op='RESPLIT', parent=1, overrides={real.session_id: 'test'})
    assert load_split(store, v2).split_of(real.session_id) == 'test' and 'inherited' not in load_split(store, v2).record
    assert v2.content['selection'] == v1.content['selection']
    with pytest.raises(DataError) as e:
        create_version(store, 'parts', op='RESPLIT', parent=2, overrides={real.session_id: 'test'})
    assert code_of(e) == 'DERIVATION_INVALID'                       # the same split again: nothing to derive


@pytest.mark.parametrize('op, kwargs', [
    ('CREATE', {'parent': 1}),
    ('ADD_SESSIONS', {}),
    ('RELABEL', {'parent': 1, 'sessions': ['x']}),
    ('FILTER', {'parent': 1, 'seed': 3}),
    ('RESPLIT', {'parent': 1, 'label_sets': ['ls-0000000000000000']}),
])
def test_an_op_refuses_arguments_it_does_not_take(cell, op, kwargs):
    store, sim, _ = cell
    create_version(store, 'parts', sessions=[sim.session_id])
    with pytest.raises(DataError) as e:
        create_version(store, 'parts', op=op, **kwargs)
    assert code_of(e) == 'DERIVATION_INVALID'


def test_two_label_sets_of_one_kind_never_label_one_observation_in_a_version(cell):
    store, sim, _ = cell
    a = labels(store, [{'observation_id': color_obs(sim, 1), 'boxes': [box()]}])
    b = labels(store, [{'observation_id': color_obs(sim, 1), 'boxes': [box(x=2.0)]}], source='HUMAN')
    with pytest.raises(DataError) as e:
        create_version(store, 'parts', sessions=[sim.session_id], label_sets=[a.label_set_id, b.label_set_id])
    assert code_of(e) == 'LABEL_CONFLICT'


# ---- rights -----------------------------------------------------------------------------------------------

def _b(source, classification, **extra):
    return {'source': source, 'rights': {'classification': classification, **extra}}


def test_rights_take_the_most_restrictive_and_are_never_promoted():
    assert combine_rights([_b('d', 'PUBLIC_APPROVED'), _b('s', 'INTERNAL')]) == {'classification': 'INTERNAL'}
    assert combine_rights([_b('d', 'INTERNAL'), _b('s', 'CUSTOMER_RESTRICTED', owner_org='acme')]) == \
        {'classification': 'CUSTOMER_RESTRICTED', 'owner_org': 'acme'}
    assert combine_rights([_b('d', 'PUBLIC_APPROVED', license='CC-BY-4.0'),
                           _b('s', 'PUBLIC_APPROVED', license='CC-BY-4.0')]) == \
        {'classification': 'PUBLIC_APPROVED', 'license': 'CC-BY-4.0'}


@pytest.mark.parametrize('basis', [
    [_b('a', 'CUSTOMER_RESTRICTED', owner_org='acme'), _b('b', 'CUSTOMER_RESTRICTED', owner_org='globex')],
    [_b('a', 'CUSTOMER_RESTRICTED', owner_org='acme'), _b('b', 'CUSTOMER_RESTRICTED')],
    [_b('a', 'PUBLIC_APPROVED', license='CC-BY-4.0'), _b('b', 'PUBLIC_APPROVED', license='CC0-1.0')],
    [_b('a', 'PUBLIC_APPROVED', license='CC-BY-4.0'), _b('b', 'PUBLIC_APPROVED')],
])
def test_ambiguous_rights_are_refused(basis):
    with pytest.raises(DataError) as e:
        combine_rights(basis)
    assert code_of(e) == 'RIGHTS_CONFLICT'


def test_a_version_carries_the_combined_rights_and_their_basis(store, vocab):
    a = make_session(store, vocab, simulated=True, rights={'classification': 'CUSTOMER_RESTRICTED',
                                                           'owner_org': 'example_customer'})
    b = make_session(store, vocab, simulated=False, rights={'classification': 'CUSTOMER_RESTRICTED',
                                                            'owner_org': 'other_customer'})
    create_dataset(store, 'parts', name='Parts', rights={'classification': 'PUBLIC_APPROVED', 'license': 'CC0-1.0'})
    v = create_version(store, 'parts', sessions=[a.session_id])
    assert v.rights == {'classification': 'CUSTOMER_RESTRICTED', 'owner_org': 'example_customer'}
    assert [x['source'] for x in v.rights_basis] == ['dataset:parts', a.session_id]
    with pytest.raises(DataError) as e:
        create_version(store, 'parts', sessions=[a.session_id, b.session_id])
    assert code_of(e) == 'RIGHTS_CONFLICT'


# ---- content-kind neutrality and source independence -------------------------------------------------------

def test_the_manifest_header_is_content_kind_neutral(cell):
    store, sim, _ = cell
    create_version(store, 'parts', sessions=[sim.session_id])
    raw = json.loads(store.path('datasets/parts/versions/1/manifest.json').read_text())
    assert set(raw) == {'schema', 'dataset_id', 'version', 'content_kind', 'derivation', 'rights', 'rights_basis',
                        'content', 'software', 'created', 'content_hash'}
    assert raw['content_kind'] == 'OBSERVATIONS' and set(raw['content']) == \
        {'sessions', 'calibrations', 'selection', 'label_sets', 'split'}


@pytest.mark.parametrize('kind, code', [('EPISODES', 'CONTENT_KIND_UNSUPPORTED'), ('IMAGES', 'INVALID_VALUE')])
def test_a_content_kind_without_a_reader_is_refused(cell, kind, code):
    store, sim, _ = cell
    create_version(store, 'parts', sessions=[sim.session_id])
    raw = json.loads(store.path('datasets/parts/versions/1/manifest.json').read_text())
    raw['content_kind'] = kind
    with pytest.raises(DataError) as e:
        V.DatasetVersion.from_dict(raw)
    assert code_of(e) == code


def test_one_version_mixes_simulated_and_physical_sources(cell):
    """D0-H in miniature: the two sources differ only in what the sessions themselves record."""
    store, sim, real = cell
    v = create_version(store, 'parts', sessions=[sim.session_id, real.session_id])
    refs = {x['session_id']: x for x in v.content['sessions']}
    assert refs[sim.session_id]['source']['backend'] == 'isaac' and refs[sim.session_id]['clock'] == \
        {'session_domain': 'SIM_TIME', 'sim_source': 'isaac'}
    assert refs[real.session_id]['source']['backend'] == 'real' and refs[real.session_id]['clock'] == \
        {'session_domain': 'ROS_TIME'}
    assert len(v.content['calibrations']) == 4


def test_no_dataset_label_split_or_export_code_branches_on_a_backend():
    names = ('dataset.py', 'versions.py', 'labels.py', 'splits.py', 'exports.py', 'dataset_integrity.py',
             '_sealed.py', 'formats/__init__.py', 'formats/yolo.py')
    for name in names:
        text = (PKG / name).read_text()
        for token in ("'isaac'", "'real'", "'mock'", "'replay'", "'gazebo'", 'realsense', 'd455', 'D455'):
            assert token not in text, f'{name} names the source {token}'
        uses = [line for line in text.splitlines() if '.backend' in line or '.transport' in line]
        # the one provenance copy into the manifest, and verify's comparison of it with the session
        expected = 1 if name in ('versions.py', 'dataset_integrity.py') else 0
        assert len(uses) == expected, f'{name}: {uses}'
