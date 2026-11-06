"""D0.6 verify: every LOGICAL check, each proven by a HASH-CONSISTENT tampering (the hashes alone see nothing).

A byte edit of a pinned file is caught by its sha256 before any semantic check runs, so it proves only the hash.
Here each record is rewritten with its hashes (content hash, split id, file pins) recomputed, and verify must still
find the inconsistency through the check written for it.
"""
import json
import os
import shutil

import pytest

from _d06 import CLASSES, box, color_obs, make_session
from trainit_data._canonical import canonical_json, pretty_json, sha256_bytes, sha256_json
from trainit_data.dataset import create_dataset
from trainit_data.dataset_integrity import verify_dataset
from trainit_data.errors import DataError
from trainit_data.exports import SIDECAR, verify_export
from trainit_data.formats import get_exporter
from trainit_data.formats.yolo import import_yolo_labels, stem_of
from trainit_data.labels import content_id, create_label_set
from trainit_data.splits import _split_id
from trainit_data.versions import content_hash_of, create_version

YOLO = get_exporter('yolo')


@pytest.fixture
def cell(store, vocab):
    sim = make_session(store, vocab, simulated=True)
    real = make_session(store, vocab, simulated=False, seed=10)
    create_dataset(store, 'parts', name='Parts')
    return store, sim, real


def bbox(store, entries, source='IMPORTED', **kw):
    return create_label_set(store, 'parts', kind='BBOX_2D', classes=CLASSES, provenance={'source': source},
                            entries=entries, **kw)


# ---- hash-consistent rewriting ----------------------------------------------------------------------------------

def _writable(path):
    os.chmod(path, 0o644)
    return path


def rewrite_manifest(store, number, mutate):
    path = _writable(store.path(f'datasets/parts/versions/{number}/manifest.json'))
    doc = json.loads(path.read_text())
    mutate(doc)
    doc['content_hash'] = content_hash_of(doc)
    path.write_bytes(pretty_json(doc))


def rewrite_split(store, number, mutate, *, keep_counts_pin=False):
    """Edit splits.json, recompute its split id, re-pin it in the manifest (and its counts, unless told not to)."""
    path = _writable(store.path(f'datasets/parts/versions/{number}/splits.json'))
    record = json.loads(path.read_text())
    mutate(record)
    record['split_id'] = _split_id(record)
    data = pretty_json(record)
    path.write_bytes(data)

    def repin(doc):
        doc['content']['split'].update(split_id=record['split_id'], sha256=sha256_bytes(data))
        if not keep_counts_pin:
            doc['content']['split']['counts'] = record['counts']
    rewrite_manifest(store, number, repin)


def recount(record):
    record['counts'] = {k: {'groups': 0, 'captures': 0} for k in ('train', 'val', 'test')}
    for group, split in record['assignment'].items():
        record['counts'][split]['groups'] += 1
        record['counts'][split]['captures'] += 3            # every fixture session selects 3 COMPLETE captures


def findings(store):
    report = verify_dataset(store, 'parts')
    return report, [(f.code, f.severity, f.artifact, f.detail) for f in report.findings]


def assert_found(store, code, needle, artifact_end=None):
    report, found = findings(store)
    assert not report.ok, found
    hits = [f for f in found
            if f[0] == code and needle in f[3] and (artifact_end is None or f[2].endswith(artifact_end))]
    assert hits, found


# ---- the manifest's pins ----------------------------------------------------------------------------------------

def _first_session(doc):
    return doc['content']['sessions'][0]


@pytest.mark.parametrize('case', ['calibration_pin', 'index_pin', 'calibration_ids', 'captures_selected',
                                  'rights_basis'])
def test_a_hash_consistent_manifest_edit_is_found(cell, case):
    store, sim, real = cell
    create_version(store, 'parts', sessions=[sim.session_id, real.session_id])
    if case == 'calibration_pin':
        cal = sorted(json.loads(store.path('datasets/parts/versions/1/manifest.json').read_text())
                     ['content']['calibrations'])[0]
        rewrite_manifest(store, 1, lambda d: d['content']['calibrations'][cal].update(sha256='sha256:' + '0' * 64))
        assert_found(store, 'HASH_MISMATCH', 'not the calibration the version pins', f'calibrations/{cal}.json')
    elif case == 'index_pin':
        rewrite_manifest(store, 1, lambda d: _first_session(d).update(index_sha256='sha256:' + '1' * 64))
        assert_found(store, 'DATASET_INCONSISTENT', 'is not the session the version pins')
    elif case == 'calibration_ids':
        def drop(d):
            gone = _first_session(d)['calibration_ids'].pop()
            del d['content']['calibrations'][gone]
        rewrite_manifest(store, 1, drop)
        assert_found(store, 'DATASET_INCONSISTENT', 'calibration ids differ')
    elif case == 'captures_selected':
        def shift(d):
            d['content']['sessions'][0]['captures_selected'] -= 1
            d['content']['sessions'][1]['captures_selected'] += 1   # the total still adds up
        rewrite_manifest(store, 1, shift)
        assert_found(store, 'DATASET_INCONSISTENT', 'captures pass the filter')
    else:
        def promote(d):
            for b in d['rights_basis']:
                b['rights'] = {'classification': 'PUBLIC_APPROVED', 'license': 'CC0-1.0'}
            d['rights'] = {'classification': 'PUBLIC_APPROVED', 'license': 'CC0-1.0'}
        rewrite_manifest(store, 1, promote)                      # a PROMOTION, made self-consistent
        assert_found(store, 'DATASET_INCONSISTENT', 'rights basis')


def test_a_rights_promotion_without_its_basis_never_even_loads(cell):
    store, sim, _ = cell
    create_version(store, 'parts', sessions=[sim.session_id])
    rewrite_manifest(store, 1, lambda d: d.update(rights={'classification': 'PUBLIC_APPROVED'}))
    assert_found(store, 'INVALID_VALUE', 'combination of their basis')


def test_a_wrong_parent_pin_is_found(cell):
    store, sim, real = cell
    create_version(store, 'parts', sessions=[sim.session_id])
    create_version(store, 'parts', op='ADD_SESSIONS', parent=1, sessions=[real.session_id])
    rewrite_manifest(store, 2, lambda d: d['derivation']['parent'].update(content_hash='sha256:' + '2' * 64))
    assert_found(store, 'DATASET_INCONSISTENT', 'parent version 1 has content hash')


def test_a_selection_that_is_not_the_filtered_sessions_is_found(cell):
    store, sim, real = cell
    create_version(store, 'parts', sessions=[sim.session_id, real.session_id])
    path = _writable(store.path('datasets/parts/versions/1/selection.jsonl'))
    lines = path.read_bytes().splitlines(keepends=True)
    partial = {'capture_id': f'{sim.session_id}/000004', 'outcome': 'PARTIAL'}
    data = b''.join(lines[:-1]) + canonical_json(partial) + b'\n'     # a COMPLETE capture swapped for a PARTIAL one
    path.write_bytes(data)
    rewrite_manifest(store, 1, lambda d: d['content']['selection'].update(sha256=sha256_bytes(data)))
    assert_found(store, 'DATASET_INCONSISTENT', 'not exactly the pinned sessions')


# ---- label pins and label meaning --------------------------------------------------------------------------------

def test_label_set_pins_and_counts_are_checked(cell):
    store, sim, _ = cell
    ls = bbox(store, [{'observation_id': color_obs(sim, 1), 'boxes': [box()]}])
    create_version(store, 'parts', sessions=[sim.session_id], label_sets=[ls.label_set_id])
    rewrite_manifest(store, 1, lambda d: d['content']['label_sets'][0].update(entries_selected=2))
    assert_found(store, 'DATASET_INCONSISTENT', 'entries in the selection')
    rewrite_manifest(store, 1, lambda d: d['content']['label_sets'][0].update(
        entries_selected=1, sha256='sha256:' + '3' * 64))
    assert_found(store, 'HASH_MISMATCH', 'not the label set version 1 pins')


def test_two_label_sets_labelling_one_observation_are_a_conflict_verify_reports(cell, tmp_path):
    store, sim, _ = cell
    a = bbox(store, [{'observation_id': color_obs(sim, 1), 'boxes': [box()]}])
    b = bbox(store, [{'observation_id': color_obs(sim, 1), 'boxes': [box(x=2.0)]}])
    create_version(store, 'parts', sessions=[sim.session_id], label_sets=[a.label_set_id])

    def add_b(d):
        d['content']['label_sets'].append({'label_set_id': b.label_set_id, 'sha256': b.sha256, 'kind': 'BBOX_2D',
                                           'source': 'IMPORTED', 'entries_selected': 1})
        d['content']['label_sets'].sort(key=lambda x: x['label_set_id'])
    rewrite_manifest(store, 1, add_b)
    assert_found(store, 'LABEL_CONFLICT', 'is labelled by')
    with pytest.raises(DataError) as e:                            # the exporter refuses it too
        YOLO.export(store, 'parts', 1, tmp_path / 'out')
    assert e.value.code == 'DATASET_INCONSISTENT'


def test_a_dangling_supersedes_is_found(cell, tmp_path):
    store, sim, _ = cell
    first = bbox(store, [{'observation_id': color_obs(sim, 1), 'boxes': [box()]}])
    second = bbox(store, [{'observation_id': color_obs(sim, 1), 'boxes': [box(w=5.0)]}], supersedes=first.label_set_id)
    shutil.move(str(store.path(f'datasets/parts/labels/{first.label_set_id}')), str(tmp_path / 'moved'))
    assert_found(store, 'LABEL_SET_NOT_FOUND', first.label_set_id,
                 f'labels/{second.label_set_id}/labelset.json')


def test_a_hand_written_label_set_with_a_box_outside_its_image_is_found(cell):
    """A label set not written by create_label_set (hash-consistent, id = content) still meets the box rule."""
    store, sim, _ = cell
    data = canonical_json({'observation_id': color_obs(sim, 1), 'boxes': [box(x=6.0, w=4.0)]}) + b'\n'
    record = {'schema': 'trainit-labels/1.0', 'dataset_id': 'parts', 'kind': 'BBOX_2D',
              'ontology': {'classes': CLASSES, 'sha256': sha256_json(CLASSES)}, 'provenance': {'source': 'IMPORTED'},
              'entries': {'path': 'entries.jsonl', 'sha256': sha256_bytes(data), 'count': 1}}
    ls_id = content_id(record)
    record.update(label_set_id=ls_id, created={'utc': '2026-11-02T00:00:00Z', 'by': 'hand'})
    directory = store.path(f'datasets/parts/labels/{ls_id}')
    directory.mkdir(parents=True)
    (directory / 'entries.jsonl').write_bytes(data)
    (directory / 'labelset.json').write_bytes(pretty_json(record))
    assert_found(store, 'LABEL_INVALID', 'outside', f'labels/{ls_id}/entries.jsonl')


def test_a_label_on_an_observation_without_an_image_block_is_refused(cell):
    store, sim, _ = cell
    from trainit_data import labels as L
    obs = type('Obs', (), {'observation_id': color_obs(sim, 1), 'data': type('D', (), {'block': 'vector'})()})()
    with pytest.raises(DataError, match='no image data block') as e:
        L.check_entry_against({'observation_id': obs.observation_id, 'boxes': []}, obs, 'x')
    assert e.value.code == 'LABEL_INVALID'


# ---- the split ---------------------------------------------------------------------------------------------------

def test_a_group_moved_between_splits_is_a_leak_verify_reports(cell):
    store, sim, real = cell
    create_version(store, 'parts', sessions=[sim.session_id, real.session_id], seed=4)

    def move(record):
        group = sorted(record['assignment'])[0]
        record['assignment'][group] = 'test' if record['assignment'][group] != 'test' else 'train'
        recount(record)
    rewrite_split(store, 1, move)
    assert_found(store, 'DATASET_INCONSISTENT', 'the rule gives', 'splits.json')


def test_split_counts_and_coverage_are_checked(cell):
    store, sim, real = cell
    create_version(store, 'parts', sessions=[sim.session_id, real.session_id])
    rewrite_split(store, 1, lambda r: r['counts']['train'].update(captures=r['counts']['train']['captures'] + 1))
    assert_found(store, 'DATASET_INCONSISTENT', 'split counts disagree')
    rewrite_split(store, 1, lambda r: (r['assignment'].update({'ses-20000101T000000Z-00000000': 'val'}), recount(r)))
    assert_found(store, 'DATASET_INCONSISTENT', 'exactly the version')


def test_inheritance_is_checked_against_the_op_and_the_parent(cell):
    store, sim, real = cell
    create_version(store, 'parts', sessions=[sim.session_id, real.session_id])
    create_version(store, 'parts', op='FILTER', parent=1, remove_sessions=[real.session_id])
    rewrite_split(store, 2, lambda r: (r.pop('inherited'), r.pop('retired', None)))
    assert_found(store, 'DATASET_INCONSISTENT', 'must inherit')


def test_an_inherited_split_id_must_be_the_parents(cell):
    store, sim, real = cell
    create_version(store, 'parts', sessions=[sim.session_id, real.session_id])
    create_version(store, 'parts', op='FILTER', parent=1, remove_sessions=[real.session_id])
    rewrite_split(store, 2, lambda r: r['inherited'].update(split_id='sp-' + '0' * 16))
    assert_found(store, 'DATASET_INCONSISTENT', 'not its parent')


def test_a_create_version_must_not_inherit(cell):
    store, sim, _ = cell
    create_version(store, 'parts', sessions=[sim.session_id])
    rewrite_split(store, 1, lambda r: r.update(inherited={'version': 1, 'split_id': r['split_id']}))
    assert_found(store, 'DATASET_INCONSISTENT', 'must not')


# ---- stray files are warnings ------------------------------------------------------------------------------------

def test_stray_files_are_warnings_and_the_dataset_stays_ok(cell):
    store, sim, _ = cell
    ls = bbox(store, [{'observation_id': color_obs(sim, 1), 'boxes': [box()]}])
    create_version(store, 'parts', sessions=[sim.session_id], label_sets=[ls.label_set_id])
    for rel in ('datasets/parts/notes.txt', 'datasets/parts/versions/1/extra.txt',
                f'datasets/parts/labels/{ls.label_set_id}/extra.txt'):
        store.path(rel).write_text('x')
    report, found = findings(store)
    assert report.ok
    warned = {f[2] for f in found if f[1] == 'WARNING'}
    assert {'datasets/parts/notes.txt', 'datasets/parts/versions/1', f'datasets/parts/labels/{ls.label_set_id}'} \
        <= warned


# ---- verify-export's structure checks ------------------------------------------------------------------------------

@pytest.fixture
def exported(cell, tmp_path):
    store, sim, real = cell
    ls = bbox(store, [{'observation_id': color_obs(sim, 1), 'boxes': [box()]},
                      {'observation_id': color_obs(sim, 2), 'boxes': [box('tray')]},
                      {'observation_id': color_obs(real, 1), 'boxes': [box()]}])
    create_version(store, 'parts', sessions=[sim.session_id, real.session_id], label_sets=[ls.label_set_id],
                   overrides={sim.session_id: 'train', real.session_id: 'test'})
    out = tmp_path / 'exports' / 'out'
    sidecar = YOLO.export(store, 'parts', 1, out)
    return store, sim, real, out, sidecar


def _sidecar(out):
    return json.loads((out / SIDECAR).read_text())


def _write_sidecar(out, doc):
    (out / SIDECAR).write_bytes(pretty_json(doc))


def codes(report):
    return {(f.code, f.artifact) for f in report.findings}


def test_the_sidecar_links_every_file_and_the_test_split_is_exported(exported):
    store, sim, real, out, sidecar = exported
    for entry in sidecar['files']:
        on_disk = out.joinpath(*entry['path'].split('/'))
        assert entry['bytes'] == on_disk.stat().st_size
        if entry['path'].startswith('images/'):
            assert stem_of(entry['observation_id']) == on_disk.stem
        elif entry['path'].startswith('labels/'):
            assert stem_of(entry['label_of']) == on_disk.stem
    assert (out / 'images' / 'test' / f'{stem_of(color_obs(real, 1))}.png').is_file()
    assert 'test: images/test' in (out / 'data.yaml').read_text()
    assert sidecar['summary']['splits']['test'] == {'images': 1, 'boxes': 1}


def test_a_missing_or_symlinked_file_is_found(exported, tmp_path):
    store, sim, _, out, _ = exported
    label = out / 'labels' / 'train' / f'{stem_of(color_obs(sim, 1))}.txt'
    label.unlink()
    assert ('PAYLOAD_MISSING', label.relative_to(out).as_posix()) in codes(verify_export(out))
    (tmp_path / 'elsewhere.txt').write_text('0 0.5 0.5 0.1 0.1\n')
    label.symlink_to(tmp_path / 'elsewhere.txt')
    assert ('PAYLOAD_MISSING', label.relative_to(out).as_posix()) in codes(verify_export(out))


def test_an_image_without_its_label_is_found(exported):
    store, sim, _, out, _ = exported
    rel = f'labels/train/{stem_of(color_obs(sim, 1))}.txt'
    (out / rel).unlink()
    doc = _sidecar(out)
    doc['files'] = [f for f in doc['files'] if f['path'] != rel]
    _write_sidecar(out, doc)
    report = verify_export(out)
    assert any('without its label file' in f.detail for f in report.findings)


def test_swapped_label_links_and_a_lying_summary_are_found(exported):
    store, sim, _, out, _ = exported
    doc = _sidecar(out)
    a, b = (f for f in doc['files'] if f['path'].startswith('labels/train/'))
    a['label_of'], b['label_of'] = b['label_of'], a['label_of']
    _write_sidecar(out, doc)
    assert any('do not name the same observation' in f.detail for f in verify_export(out).findings)
    doc = _sidecar(out)
    a, b = (f for f in doc['files'] if f['path'].startswith('labels/train/'))
    a['label_of'], b['label_of'] = b['label_of'], a['label_of']
    doc['summary']['splits']['train']['boxes'] = 7
    _write_sidecar(out, doc)
    assert any('summary.splits.train' in f.detail for f in verify_export(out).findings)


def test_a_store_version_that_changed_since_the_export_is_found(exported):
    store, sim, real, out, _ = exported
    assert verify_export(out, store).ok
    rewrite_manifest(store, 1, lambda d: d['rights_basis'][0].update(rights={'classification': 'INTERNAL',
                                                                             'notes': 'edited later'}))
    report = verify_export(out, store)
    assert not report.ok and any(f.artifact == SIDECAR for f in report.findings)


def test_the_exporter_refuses_a_session_the_version_does_not_pin(exported, tmp_path):
    store, sim, real, _, _ = exported
    rewrite_manifest(store, 1, lambda d: next(x for x in d['content']['sessions']
                                              if x['session_id'] == sim.session_id).update(
        index_sha256='sha256:' + '4' * 64))
    with pytest.raises(DataError, match='not the one the version pins') as e:
        YOLO.export(store, 'parts', 1, tmp_path / 'exports' / 'again')
    assert e.value.code == 'DATASET_INCONSISTENT'


def test_an_export_from_a_rebuilt_store_is_byte_identical(cell, tmp_path):
    """D0-F across rebuilds: the same version in another store (another root, exported at another time) exports the
    same bytes.

    The version is built ONCE and the store copied. Building it twice is NOT "the same inputs": a version pins each
    label set by the sha256 of its whole header, ``created`` (UTC, whole seconds) included, so two creations either
    side of a second boundary are two different pins. That made this test fail about once in six runs (D0.9)."""
    store, sim, real = cell
    ls = bbox(store, [{'observation_id': color_obs(sim, 1), 'boxes': [box()]},
                      {'observation_id': color_obs(real, 2), 'boxes': [box('tray')]}])
    create_version(store, 'parts', sessions=[sim.session_id, real.session_id], label_sets=[ls.label_set_id])
    twin_root = tmp_path / 'twin'
    shutil.copytree(store.root, twin_root)
    from trainit_data.store import Store
    twin = Store.open(twin_root)
    YOLO.export(store, 'parts', 1, tmp_path / 'exports' / 'a')
    YOLO.export(twin, 'parts', 1, tmp_path / 'exports' / 'b')
    from _helpers import snapshot
    assert snapshot(tmp_path / 'exports' / 'a') == snapshot(tmp_path / 'exports' / 'b')


# ---- the importer names file:line ----------------------------------------------------------------------------------

def test_a_bad_line_is_named_by_its_file_and_line(cell, tmp_path):
    store, sim, _ = cell
    nested = tmp_path / 'labels' / 'sub' / 'dir'
    nested.mkdir(parents=True)
    (tmp_path / 'labels' / f'{stem_of(color_obs(sim, 1))}.txt').write_text('0 0.5 0.5 0.2 0.2\n')
    (nested / f'{stem_of(color_obs(sim, 2))}.txt').write_text('0 0.5 0.5 0.2 0.2\n1 0.5 0.5 0.2 0.2\n0 0.5 0.5 2 0.2\n')
    with pytest.raises(DataError) as e:
        import_yolo_labels(store, 'parts', tmp_path / 'labels', CLASSES)
    assert e.value.code == 'LABEL_INVALID'
    assert f'sub/dir/{stem_of(color_obs(sim, 2))}.txt:3' in e.value.message


@pytest.mark.parametrize('text, ok', [('0 5e-1 +0.5 1e-1 0.1\n', True), ('0 NaN 0.5 0.1 0.1\n', False),
                                      ('0 0.5 0.5 -0.0 0.1\n', False), ('﻿0 0.5 0.5 0.1 0.1\n', False)])
def test_number_forms(cell, tmp_path, text, ok):
    store, sim, _ = cell
    directory = tmp_path / 'labels'
    directory.mkdir()
    (directory / f'{stem_of(color_obs(sim, 1))}.txt').write_bytes(text.encode('utf-8'))
    (directory / 'notes.md').write_text('ignored: not a .txt file')
    if ok:
        import_yolo_labels(store, 'parts', directory, CLASSES)
    else:
        with pytest.raises(DataError) as e:
            import_yolo_labels(store, 'parts', directory, CLASSES)
        assert e.value.code == 'LABEL_INVALID'


def test_a_label_on_a_depth_stream_is_imported_but_never_exported(cell, tmp_path):
    store, sim, _ = cell
    directory = tmp_path / 'labels'
    directory.mkdir()
    depth = color_obs(sim, 1).replace('/color', '/depth')
    (directory / f'{stem_of(depth)}.txt').write_text('0 0.5 0.5 0.2 0.2\n')
    ls = import_yolo_labels(store, 'parts', directory, CLASSES)
    create_version(store, 'parts', sessions=[sim.session_id], label_sets=[ls.label_set_id])
    with pytest.raises(DataError, match='rgb8'):
        YOLO.export(store, 'parts', 1, tmp_path / 'exports' / 'out')
