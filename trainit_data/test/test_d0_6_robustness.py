"""D0.6 robustness: interrupted creations, split lineage, tampering that keeps hashes consistent, export atomicity.

The cases the independent review of D0.6 found (2026-11-02), each pinned by a test.
"""
import dataclasses
import json
import os

import pytest

from _d06 import CLASSES, box, color_obs, make_session
from trainit_data import labels as L
from trainit_data.dataset import create_dataset, load_dataset
from trainit_data.dataset_integrity import verify_dataset
from trainit_data.errors import DataError
from trainit_data.exports import SIDECAR, verify_export
from trainit_data.formats import get_exporter
from trainit_data.formats.yolo import import_yolo_labels, parse_label_file, stem_of
from trainit_data.labels import create_label_set, list_label_sets
from trainit_data.versions import content_hash_of, create_version, derivation_problems, list_versions, load_split, \
    load_version

YOLO = get_exporter('yolo')


@pytest.fixture
def cell(store, vocab):
    sim = make_session(store, vocab, simulated=True)
    real = make_session(store, vocab, simulated=False, seed=10)
    create_dataset(store, 'parts', name='Parts')
    return store, sim, real


def bbox(store, entries, **kw):
    return create_label_set(store, 'parts', kind='BBOX_2D', classes=CLASSES, provenance={'source': 'IMPORTED'},
                            entries=entries, **kw)


# ---- interrupted creations -------------------------------------------------------------------------------------

def test_a_label_set_interrupted_after_its_entries_is_completed_by_the_same_import(cell, monkeypatch):
    store, sim, _ = cell
    entries = [{'observation_id': color_obs(sim, 1), 'boxes': [box()]}]
    real_write = L.write_new_file

    def crash_on_header(path, data):
        if str(path).endswith('labelset.json'):
            raise OSError(28, 'No space left on device')
        return real_write(path, data)
    monkeypatch.setattr(L, 'write_new_file', crash_on_header)
    with pytest.raises(DataError) as e:
        bbox(store, entries)
    assert e.value.code == 'WRITE_FAILED' and list_label_sets(store, 'parts') == []
    monkeypatch.setattr(L, 'write_new_file', real_write)
    assert any(f.severity == 'WARNING' for f in verify_dataset(store, 'parts').findings)
    ls = bbox(store, entries)                                         # the same import completes it
    assert list_label_sets(store, 'parts') == [ls.label_set_id] and verify_dataset(store, 'parts').ok


def test_an_interrupted_label_set_that_does_not_match_is_refused_not_guessed(cell):
    store, sim, _ = cell
    entries = [{'observation_id': color_obs(sim, 1), 'boxes': [box()]}]
    ls = bbox(store, entries)
    directory = store.path(f'datasets/parts/labels/{ls.label_set_id}')
    header = directory / 'labelset.json'
    os.chmod(directory / 'entries.jsonl', 0o644)
    header.unlink()                                                   # as if the header had never been written
    (directory / 'entries.jsonl').write_bytes(b'{}\n')                # ... and the entries were torn
    with pytest.raises(DataError) as e:
        bbox(store, entries)
    assert e.value.code == 'DATASET_INCONSISTENT' and 'move that directory' in e.value.message


def test_an_empty_dataset_directory_is_an_interrupted_creation_and_is_completed(store):
    store.path('datasets/parts').mkdir(parents=True)
    create_dataset(store, 'parts', name='Parts')
    assert load_dataset(store, 'parts').name == 'Parts'


@pytest.mark.parametrize('name', ['²', '01', 'draft'])
def test_a_stray_version_directory_never_crashes_listing_or_verify(cell, name):
    store, sim, _ = cell
    create_version(store, 'parts', sessions=[sim.session_id])
    store.path(f'datasets/parts/versions/{name}').mkdir()
    assert list_versions(store, 'parts') == [1]
    report = verify_dataset(store, 'parts')
    assert report.ok and any(f.severity == 'WARNING' and f.artifact.endswith(f'/{name}') for f in report.findings)
    assert create_version(store, 'parts', sessions=[sim.session_id]).version == 2


# ---- split lineage ----------------------------------------------------------------------------------------------

def test_a_session_filtered_out_and_added_back_returns_to_its_split(cell):
    """No leakage across versions: a session evaluated in test never comes back as train."""
    store, sim, real = cell
    create_version(store, 'parts', sessions=[sim.session_id, real.session_id],
                   overrides={sim.session_id: 'train', real.session_id: 'test'})
    v2 = create_version(store, 'parts', op='FILTER', parent=1, remove_sessions=[real.session_id])
    assert load_split(store, v2).record['retired'] == {real.session_id: 'test'}
    v3 = create_version(store, 'parts', op='ADD_SESSIONS', parent=2, sessions=[real.session_id])
    assert load_split(store, v3).split_of(real.session_id) == 'test'
    with pytest.raises(DataError) as e:
        create_version(store, 'parts', op='ADD_SESSIONS', parent=2, sessions=[real.session_id],
                       overrides={real.session_id: 'train'})
    assert e.value.code == 'DERIVATION_INVALID'
    assert verify_dataset(store, 'parts').ok


# ---- verify sees what the hashes alone cannot ---------------------------------------------------------------------

def _rewrite_manifest(store, number, mutate):
    path = store.path(f'datasets/parts/versions/{number}/manifest.json')
    os.chmod(path, 0o644)
    doc = json.loads(path.read_text())
    mutate(doc)
    doc['content_hash'] = content_hash_of(doc)                       # a hash-consistent edit
    path.write_text(json.dumps(doc))


def test_verify_compares_the_recorded_source_and_counts_with_the_sessions(cell):
    store, sim, real = cell
    create_version(store, 'parts', sessions=[sim.session_id, real.session_id])

    def mislabel(doc):
        ref = next(x for x in doc['content']['sessions'] if x['session_id'] == sim.session_id)
        ref['source']['backend'] = 'real'
        doc['content']['selection']['excluded']['PARTIAL'] = 0
    _rewrite_manifest(store, 1, mislabel)
    details = [f.detail for f in verify_dataset(store, 'parts').findings]
    assert any('source / clock' in d for d in details) and any('selection.excluded' in d for d in details)


def test_a_version_that_breaks_its_ops_rule_is_reported(cell):
    store, sim, real = cell
    create_version(store, 'parts', sessions=[sim.session_id, real.session_id], include_outcomes=['COMPLETE', 'PARTIAL'])
    v2 = create_version(store, 'parts', op='FILTER', parent=1, include_outcomes=['COMPLETE'])
    v1 = load_version(store, 'parts', 1)
    assert derivation_problems(v2, v1) == []
    as_relabel = dataclasses.replace(v2, derivation={'op': 'RELABEL', 'parent': v2.derivation['parent']})
    assert len(derivation_problems(as_relabel, v1)) == 2


# ---- exports ------------------------------------------------------------------------------------------------------

@pytest.fixture
def exported(cell, tmp_path):
    store, sim, real = cell
    ls = bbox(store, [{'observation_id': color_obs(sim, 1), 'boxes': [box()]},
                      {'observation_id': color_obs(real, 2), 'boxes': [box('tray')]}])
    create_version(store, 'parts', sessions=[sim.session_id, real.session_id], label_sets=[ls.label_set_id],
                   overrides={sim.session_id: 'train', real.session_id: 'val'})
    out = tmp_path / 'out'
    YOLO.export(store, 'parts', 1, out)
    return store, sim, real, out


def test_an_edited_label_with_a_matching_sidecar_is_caught_against_the_store(exported):
    store, sim, real, out = exported
    rel = f'labels/val/{stem_of(color_obs(real, 2))}.txt'
    (out / rel).write_text('0 0.500000 0.500000 0.250000 0.250000\n')         # another class, another box
    sidecar = json.loads((out / SIDECAR).read_text())
    from trainit_data._canonical import sha256_bytes
    entry = next(e for e in sidecar['files'] if e['path'] == rel)
    entry['sha256'], entry['bytes'] = sha256_bytes((out / rel).read_bytes()), len((out / rel).read_bytes())
    sidecar['summary']['splits']['val']['boxes'] = 1
    (out / SIDECAR).write_text(json.dumps(sidecar))
    assert verify_export(out).ok                                     # self-consistent on its own...
    report = verify_export(out, store)                               # ...but not the export the store gives
    assert not report.ok and rel in {f.artifact for f in report.findings}


def test_a_malformed_sidecar_is_a_finding_never_a_crash(exported):
    store, _, _, out = exported
    sidecar = json.loads((out / SIDECAR).read_text())
    next(e for e in sidecar['files'] if 'observation_id' in e)['observation_id'] = 'bogus'
    (out / SIDECAR).write_text(json.dumps(sidecar))
    report = verify_export(out, store)
    assert not report.ok and report.findings[0].artifact == SIDECAR


def test_a_failed_export_leaves_nothing_behind(cell, tmp_path):
    store, sim, real = cell
    ls = bbox(store, [{'observation_id': color_obs(sim, 1), 'boxes': [box()]}])
    create_version(store, 'parts', sessions=[sim.session_id], label_sets=[ls.label_set_id])
    payload = next(store.path(f'sessions/{sim.session_id}/payload/000001').glob('*color*'))
    payload.write_bytes(payload.read_bytes() + b'!')
    exports = tmp_path / 'exports'
    (exports / 'out').mkdir(parents=True)                              # an empty destination is allowed
    with pytest.raises(DataError) as e:
        YOLO.export(store, 'parts', 1, exports / 'out')
    assert e.value.code == 'HASH_MISMATCH'
    assert sorted(p.name for p in exports.iterdir()) == ['out'] and not any((exports / 'out').iterdir())


def test_an_export_into_the_store_root_is_refused(cell, tmp_path):
    store, _, _ = cell
    with pytest.raises(DataError) as e:
        YOLO.export(store, 'parts', 1, store.root / 'exports' / 'x')
    assert e.value.code == 'EXPORT_DESTINATION_INVALID'


def test_an_export_with_no_labelled_observation_in_the_selection_is_refused(cell, tmp_path):
    store, sim, _ = cell
    ls = bbox(store, [{'observation_id': color_obs(sim, 4), 'boxes': [box()]}])        # a PARTIAL capture
    create_version(store, 'parts', sessions=[sim.session_id], label_sets=[ls.label_set_id])
    with pytest.raises(DataError, match='nothing to export') as e:
        YOLO.export(store, 'parts', 1, tmp_path / 'out')
    assert e.value.code == 'EXPORT_REFUSED' and not (tmp_path / 'out').exists()


# ---- import strictness ---------------------------------------------------------------------------------------------

@pytest.mark.parametrize('data, needle', [
    (b'0\x1f0.5\x1f0.5 0.1 0.1\n', 'class'),           # an ASCII control character is no separator
    (b'\n', 'empty line'),                             # a newline is not an empty file
    (b'00 0.5 0.5 0.1 0.1\n', 'class'),
    (b'0 0.5 0.5 0.1 0.1\r', 'got'),                   # a lone carriage return
])
def test_label_lines_are_parsed_strictly(data, needle):
    with pytest.raises(DataError, match=needle) as e:
        parse_label_file(data, 2, 'x.txt')
    assert e.value.code == 'LABEL_INVALID'


def test_tabs_separate_fields_and_a_box_too_small_to_export_is_refused(cell, tmp_path):
    store, sim, _ = cell
    assert parse_label_file(b'1\t0.5 0.5\t0.1 0.1\n', 2, 'x') == [(1, 0.5, 0.5, 0.1, 0.1)]
    with pytest.raises(DataError, match='smaller than') as e:
        bbox(store, [{'observation_id': color_obs(sim, 1), 'boxes': [box(w=1e-5)]}])
    assert e.value.code == 'LABEL_INVALID'
    directory = tmp_path / 'labels'
    directory.mkdir()
    (directory / f'{stem_of(color_obs(sim, 1))}.txt').write_text('0 0.5 0.5 0.0000001 0.5\n')
    with pytest.raises(DataError, match='smaller than'):
        import_yolo_labels(store, 'parts', directory, CLASSES)
