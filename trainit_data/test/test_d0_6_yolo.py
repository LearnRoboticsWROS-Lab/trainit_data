"""D0.6: the YOLO detection exporter (D0-F: deterministic) and the strict YOLO label importer.

One export of one version mixing a simulated and a physical session (D0-H in miniature); no Ultralytics.
"""
import json
import subprocess

import pytest

from _d06 import CLASSES, box, color_obs, make_session, yolo_line
from _helpers import snapshot
from trainit_data.dataset import create_dataset
from trainit_data.errors import DataError
from trainit_data.exports import SIDECAR, read_sidecar, verify_export
from trainit_data.formats import get_exporter
from trainit_data.formats.yolo import import_yolo_labels, parse_label_file, stem_of
from trainit_data.labels import create_label_set, list_label_sets, read_entries
from trainit_data.versions import create_version

YOLO = get_exporter('yolo')


@pytest.fixture
def version(store, vocab):
    sim = make_session(store, vocab, simulated=True)
    real = make_session(store, vocab, simulated=False, seed=10)
    create_dataset(store, 'parts', name='Parts')
    ls = create_label_set(store, 'parts', kind='BBOX_2D', classes=CLASSES, provenance={'source': 'IMPORTED'},
                          entries=[{'observation_id': color_obs(sim, 1), 'boxes': [box(), box('tray', 0, 0, 8, 6)]},
                                   {'observation_id': color_obs(sim, 2), 'boxes': []},
                                   {'observation_id': color_obs(real, 3), 'boxes': [box('tray', 2.5, 1.5, 3, 2)]},
                                   {'observation_id': color_obs(real, 4), 'boxes': [box()]}])      # PARTIAL: excluded
    create_version(store, 'parts', sessions=[sim.session_id, real.session_id], label_sets=[ls.label_set_id],
                   overrides={sim.session_id: 'train', real.session_id: 'val'})
    return store, sim, real, ls


def tree(root):
    return snapshot(root)


# ---- export -----------------------------------------------------------------------------------------------

def test_the_export_layout_and_its_content(version, tmp_path):
    store, sim, real, ls = version
    sidecar = YOLO.export(store, 'parts', 1, tmp_path / 'out')
    files = tree(tmp_path / 'out')
    a, b, neg = (stem_of(color_obs(sim, 1)), stem_of(color_obs(real, 3)), stem_of(color_obs(sim, 2)))
    assert sorted(files) == sorted([f'images/train/{a}.png', f'labels/train/{a}.txt', f'images/train/{neg}.png',
                                    f'labels/train/{neg}.txt', f'images/val/{b}.png', f'labels/val/{b}.txt',
                                    'data.yaml', SIDECAR])
    assert files[f'labels/train/{a}.txt'] == (yolo_line(0, 1, 1, 4, 3) + yolo_line(1, 0, 0, 8, 6)).encode()
    assert files[f'labels/val/{b}.txt'] == yolo_line(1, 2.5, 1.5, 3, 2).encode()
    assert files[f'labels/train/{neg}.txt'] == b''                                  # an explicit negative
    payload = next(o for c in store.open_session(sim.session_id).captures() for o in c.observations
                   if o.observation_id == color_obs(sim, 1)).payload
    assert files[f'images/train/{a}.png'] == store.path(payload.path).read_bytes()   # the canonical bytes
    data_yaml = files['data.yaml'].decode()
    assert 'train: images/train\nval: images/val\n' in data_yaml and 'test:' not in data_yaml
    assert "names:\n  0: 'part'\n  1: 'tray'\n" in data_yaml and '\npath:' not in data_yaml and '/home' not in data_yaml
    assert sidecar['summary']['splits'] == {'train': {'images': 2, 'boxes': 2}, 'val': {'images': 1, 'boxes': 1},
                                            'test': {'images': 0, 'boxes': 0}}
    assert sidecar['source']['label_sets'] == [{'label_set_id': ls.label_set_id, 'sha256': ls.sha256,
                                                'source': 'IMPORTED'}]
    assert sidecar['rights'] == {'classification': 'INTERNAL'}
    assert sidecar['exporter']['config'] == {'task': 'detect', 'label_sets': [ls.label_set_id],
                                             'allow_mixed_provenance': False}
    assert verify_export(tmp_path / 'out', store).ok


def test_two_exports_of_one_version_are_byte_identical(version, tmp_path):
    """D0-F: deterministic; no timestamp, no machine path."""
    store = version[0]
    first = YOLO.export(store, 'parts', 1, tmp_path / 'a')
    second = YOLO.export(store, 'parts', 1, tmp_path / 'deeper' / 'b')
    assert tree(tmp_path / 'a') == tree(tmp_path / 'deeper' / 'b')
    assert first == second and first['export_id'].startswith('exp-')
    text = (tmp_path / 'a' / SIDECAR).read_text()
    assert str(tmp_path) not in text and 'utc' not in text


def test_verify_export_detects_tampering(version, tmp_path):
    store, sim, real, _ = version
    out = tmp_path / 'out'
    YOLO.export(store, 'parts', 1, out)
    label = out / 'labels' / 'val' / f'{stem_of(color_obs(real, 3))}.txt'
    label.write_text('1 0.5 0.5 0.1 0.1\n')
    (out / 'extra.txt').write_text('x')
    report = verify_export(out, store)
    assert not report.ok
    assert {(f.code, f.artifact) for f in report.findings} >= {('HASH_MISMATCH', label.relative_to(out).as_posix()),
                                                               ('DATASET_INCONSISTENT', 'extra.txt')}
    (out / 'extra.txt').unlink()
    (out / 'data.yaml').write_text((out / 'data.yaml').read_text().replace("'tray'", "'bin'"))
    findings = verify_export(out, store).findings
    assert any(f.artifact == 'data.yaml' and f.code == 'DATASET_INCONSISTENT' for f in findings)


def test_verify_export_against_the_store_catches_a_foreign_image(version, tmp_path):
    store, sim, real, _ = version
    out = tmp_path / 'out'
    YOLO.export(store, 'parts', 1, out)
    sidecar = json.loads((out / SIDECAR).read_text())
    a, b = (e for e in sidecar['files'] if 'observation_id' in e and e['path'].startswith('images/train'))
    a['observation_id'], b['observation_id'] = b['observation_id'], a['observation_id']   # images swapped in the record
    (out / SIDECAR).write_text(json.dumps(sidecar))
    assert not any(f.artifact.startswith('images/') for f in verify_export(out).findings)   # hashes alone agree
    report = verify_export(out, store)
    assert {f.artifact for f in report.findings if f.code == 'DATASET_INCONSISTENT'} >= {a['path'], b['path']}


@pytest.mark.parametrize('where', ['nonempty', 'repository', 'store'])
def test_an_export_never_overwrites_never_lands_in_git_or_in_the_store(version, tmp_path, where):
    store = version[0]
    if where == 'nonempty':
        dest = tmp_path / 'busy'
        dest.mkdir()
        (dest / 'keep.txt').write_text('mine')
    elif where == 'repository':
        repo = tmp_path / 'repo'
        repo.mkdir()
        subprocess.run(['git', 'init', '-q', str(repo)], check=True)
        dest = repo / 'export'
    else:
        dest = store.root / 'sessions' / 'export'
    with pytest.raises(DataError) as e:
        YOLO.export(store, 'parts', 1, dest)
    assert e.value.code == 'EXPORT_DESTINATION_INVALID'


def test_segmentation_without_an_instance_mask_label_set_is_refused(version, tmp_path):
    """D0.6 deferred task segment; M4.5 implements it over INSTANCE_MASK label sets only (test_m45_segment.py):
    a version holding BBOX_2D label sets alone has nothing to segment, and a box is never turned into a polygon."""
    store, _, _, ls = version
    with pytest.raises(DataError, match='no INSTANCE_MASK label set') as e:
        YOLO.export(store, 'parts', 1, tmp_path / 'out', {'task': 'segment'})
    assert e.value.code == 'EXPORT_REFUSED' and not (tmp_path / 'out').exists()
    with pytest.raises(DataError, match='not a INSTANCE_MASK label set') as e:
        YOLO.export(store, 'parts', 1, tmp_path / 'out', {'task': 'segment', 'label_sets': [ls.label_set_id]})
    with pytest.raises(DataError, match='options of task segment') as e:
        YOLO.export(store, 'parts', 1, tmp_path / 'out', {'epsilon_px': 2.0})
    assert e.value.code == 'EXPORT_REFUSED' and not (tmp_path / 'out').exists()


def test_mixed_provenance_is_refused_unless_allowed_and_then_recorded(version, tmp_path):
    store, sim, real, ls = version
    human = create_label_set(store, 'parts', kind='BBOX_2D', classes=CLASSES, provenance={'source': 'HUMAN'},
                             entries=[{'observation_id': color_obs(sim, 3), 'boxes': [box()]}])
    create_version(store, 'parts', op='RELABEL', parent=1, label_sets=[human.label_set_id])
    with pytest.raises(DataError) as e:
        YOLO.export(store, 'parts', 2, tmp_path / 'a')
    assert e.value.code == 'EXPORT_REFUSED'
    sidecar = YOLO.export(store, 'parts', 2, tmp_path / 'b', {'allow_mixed_provenance': True})
    assert sidecar['exporter']['config']['allow_mixed_provenance'] is True
    assert sorted(x['source'] for x in sidecar['source']['label_sets']) == ['HUMAN', 'IMPORTED']
    only = YOLO.export(store, 'parts', 2, tmp_path / 'c', {'label_sets': [human.label_set_id]})
    assert only['summary']['splits']['train'] == {'images': 1, 'boxes': 1}


def test_different_ontologies_and_non_rgb_labels_are_refused(version, tmp_path):
    store, sim, real, ls = version
    other = create_label_set(store, 'parts', kind='BBOX_2D', classes=['part'], provenance={'source': 'IMPORTED'},
                             entries=[{'observation_id': color_obs(sim, 3), 'boxes': [box()]}])
    create_version(store, 'parts', op='RELABEL', parent=1, label_sets=[other.label_set_id])
    with pytest.raises(DataError, match='ontolog') as e:
        YOLO.export(store, 'parts', 2, tmp_path / 'a')
    assert e.value.code == 'EXPORT_REFUSED'
    depth = color_obs(sim, 3).replace('/color', '/depth')
    on_depth = create_label_set(store, 'parts', kind='BBOX_2D', classes=CLASSES, provenance={'source': 'IMPORTED'},
                                entries=[{'observation_id': depth, 'boxes': [box()]}])
    create_version(store, 'parts', op='RELABEL', parent=1, label_sets=[on_depth.label_set_id],
                   drop_label_sets=[ls.label_set_id])
    with pytest.raises(DataError, match='rgb8') as e:
        YOLO.export(store, 'parts', 3, tmp_path / 'b')
    assert e.value.code == 'EXPORT_REFUSED'


def test_a_version_without_box_labels_has_nothing_to_export(store, vocab, tmp_path):
    sim = make_session(store, vocab, simulated=True)
    create_dataset(store, 'parts', name='Parts')
    create_version(store, 'parts', sessions=[sim.session_id])
    with pytest.raises(DataError) as e:
        YOLO.export(store, 'parts', 1, tmp_path / 'out')
    assert e.value.code == 'EXPORT_REFUSED'


def test_an_altered_payload_is_refused_at_export(version, tmp_path):
    store, sim, _, _ = version
    obs = next(o for c in store.open_session(sim.session_id).captures() for o in c.observations
               if o.observation_id == color_obs(sim, 1))
    path = store.path(obs.payload.path)
    path.write_bytes(path.read_bytes()[:-1] + b'\x00')
    with pytest.raises(DataError) as e:
        YOLO.export(store, 'parts', 1, tmp_path / 'out')
    assert e.value.code == 'HASH_MISMATCH'


# ---- import -----------------------------------------------------------------------------------------------

def test_import_round_trips_an_export(version, tmp_path):
    store, sim, real, ls = version
    out = tmp_path / 'out'
    YOLO.export(store, 'parts', 1, out)
    imported = import_yolo_labels(store, 'parts', out, CLASSES, source='HUMAN', tool='example_tool')
    assert imported.provenance == {'source': 'HUMAN', 'source_ref': 'yolo-detection-txt', 'tool': 'example_tool'}
    selected = [e for e in read_entries(store, ls) if e['observation_id'] != color_obs(real, 4)]
    got = read_entries(store, imported)
    assert [e['observation_id'] for e in got] == [e['observation_id'] for e in selected]
    for a, b in zip(got, selected):
        assert [x['class'] for x in a['boxes']] == [x['class'] for x in b['boxes']]
        for p, q in zip(a['boxes'], b['boxes']):
            assert p['bbox_xywh_px'] == pytest.approx(q['bbox_xywh_px'], abs=1e-5)
    create_version(store, 'parts', op='RELABEL', parent=1, label_sets=[imported.label_set_id],
                   drop_label_sets=[ls.label_set_id])
    YOLO.export(store, 'parts', 2, tmp_path / 'again')
    labels_of = lambda root: {k: v for k, v in tree(root).items() if k.startswith('labels/')}   # noqa: E731
    assert labels_of(tmp_path / 'again') == labels_of(out)              # the label text survives the round trip


def _write(directory, files):
    directory.mkdir(parents=True, exist_ok=True)
    for name, text in files.items():
        (directory / name).write_bytes(text if isinstance(text, bytes) else text.encode())
    return directory


@pytest.mark.parametrize('text, needle', [
    ('0 0.5 0.5 0.2\n', '4 fields'),
    ('0 0.5 0.5 0.2 0.2 0.3 0.3\n', 'segmentation'),
    ('2 0.5 0.5 0.2 0.2\n', 'class'),
    ('-1 0.5 0.5 0.2 0.2\n', 'class'),
    ('0.0 0.5 0.5 0.2 0.2\n', 'class'),
    ('0 nan 0.5 0.2 0.2\n', 'decimal'),
    ('0 inf 0.5 0.2 0.2\n', 'decimal'),
    ('0 1_0 0.5 0.2 0.2\n', 'decimal'),
    ('0 1.2 0.5 0.2 0.2\n', '[0, 1]'),
    ('0 0.5 -0.5 0.2 0.2\n', '[0, 1]'),
    ('0 0.5 0.5 0 0.2\n', 'positive'),
    ('0 0.05 0.5 0.2 0.2\n', 'outside'),
    ('0 0.5 0.5 0.2 0.2\n\n0 0.5 0.5 0.2 0.2\n', 'empty line'),
    ('0 0.5 0.5 0.2 0.2 é\n'.encode('utf-8'), 'ASCII'),
])
def test_a_bad_line_refuses_the_whole_import_naming_it(version, tmp_path, text, needle):
    store, sim, _, _ = version
    before = list_label_sets(store, 'parts')
    good = stem_of(color_obs(sim, 1))
    bad = stem_of(color_obs(sim, 2))
    directory = _write(tmp_path / 'labels', {f'{good}.txt': '0 0.5 0.5 0.2 0.2\n', f'{bad}.txt': text})
    with pytest.raises(DataError, match=needle) as e:
        import_yolo_labels(store, 'parts', directory, CLASSES)
    assert e.value.code == 'LABEL_INVALID' and f'{bad}.txt' in e.value.message
    assert list_label_sets(store, 'parts') == before                    # nothing written


@pytest.mark.parametrize('files, code', [
    ({'image_0001.txt': ''}, 'LABEL_INVALID'),
    ({'ses-20000101T000000Z-00000000__000001__overhead_camera__color.txt': ''}, 'SESSION_NOT_FOUND'),
])
def test_a_file_that_names_no_observation_is_refused(version, tmp_path, files, code):
    with pytest.raises(DataError) as e:
        import_yolo_labels(version[0], 'parts', _write(tmp_path / 'labels', files), CLASSES)
    assert e.value.code == code


def test_the_same_stem_twice_and_an_empty_directory_are_refused(version, tmp_path):
    store, sim, _, _ = version
    stem = stem_of(color_obs(sim, 1))
    _write(tmp_path / 'labels' / 'a', {f'{stem}.txt': ''})
    _write(tmp_path / 'labels' / 'b', {f'{stem}.txt': ''})
    with pytest.raises(DataError, match='labelled by both'):
        import_yolo_labels(store, 'parts', tmp_path / 'labels', CLASSES)
    (tmp_path / 'empty').mkdir()
    with pytest.raises(DataError, match='no .txt'):
        import_yolo_labels(store, 'parts', tmp_path / 'empty', CLASSES)


def test_crlf_and_an_empty_file_are_accepted_and_only_import_sources_are_allowed(version, tmp_path):
    store, sim, _, _ = version
    directory = _write(tmp_path / 'labels', {f'{stem_of(color_obs(sim, 1))}.txt': '0 0.5 0.5 0.25 0.5\r\n',
                                             f'{stem_of(color_obs(sim, 3))}.txt': ''})
    ls = import_yolo_labels(store, 'parts', directory, CLASSES)
    entries = read_entries(store, ls)
    assert entries[0]['boxes'] == [{'class': 'part', 'bbox_xywh_px': [3.0, 1.5, 2.0, 3.0]}]
    assert entries[1]['boxes'] == []
    with pytest.raises(DataError):
        import_yolo_labels(store, 'parts', directory, CLASSES, source='SYNTHETIC_GROUND_TRUTH')


def test_parse_label_file_tolerates_six_decimal_rounding_at_the_edges_only():
    assert parse_label_file(b'0 0.062500 0.500000 0.125001 1.000000\n', 1, 'x')    # rounding beyond the edge
    with pytest.raises(DataError):
        parse_label_file(b'0 0.062500 0.500000 0.126000 1.000000\n', 1, 'x')    # a real overflow


def test_read_sidecar_refuses_a_directory_that_is_no_export(tmp_path):
    with pytest.raises(DataError) as e:
        read_sidecar(tmp_path)
    assert e.value.code == 'PAYLOAD_MISSING'
