"""M4.5: the YOLO SEGMENTATION export (``yolo`` 1.1, task ``segment``) of INSTANCE_MASK label sets.

Synthetic, customer-neutral shapes drawn into a Replicator-SHAPED instance-id image (plain numpy, no Isaac): squares,
discs, an L, a ring, an instance in two pieces, touching instances, an instance on the image border, a tiny one. The
round trip rasterizes the exported polygons back the way Ultralytics does (de-normalize, truncate to int, fill the
polygon including its boundary) with Pillow's ``ImageDraw.polygon``. No cv2, no Ultralytics.
"""
import json

import numpy as np
import pytest
from PIL import Image, ImageDraw

from _d07 import camera_params, scene
from _helpers import snapshot
from trainit_data.adapters.isaac_replicator import ReplicatorIngest
from trainit_data.cli import main
from trainit_data.dataset import create_dataset
from trainit_data.errors import DataError
from trainit_data.exports import SIDECAR, verify_export
from trainit_data.formats import get_exporter
from trainit_data.formats._polygons import (POLICY, doubled_area, instance_polygon, largest_component,
                                            simplify_closed, trace_outer_boundary)
from trainit_data.formats.yolo import DEFAULT_EPSILON_PX, parse_segment_file, segment_label_lines, stem_of
from trainit_data.ground_truth import create_synthetic_label_set
from trainit_data.labels import read_entries
from trainit_data.versions import create_version

YOLO = get_exporter('yolo')
W, H = 96, 64
PINNED_05 = 'sha256:6a95336e0f23c5e82ec5ca0f20cc4b9a63c551df076678e308786015b193117e'     # epsilon_px 0.5
PINNED_15 = 'sha256:d62c81530a42cdf535c1af57a1b4b765a69bc66f0f5f02af3802faa8330e611f'     # epsilon_px 1.5
CLASSES = ['cube', 'cylinder', 'sphere']


# ---- synthetic shapes ---------------------------------------------------------------------------------------------

def square(x, y, side, w=W, h=H):
    m = np.zeros((h, w), dtype=bool)
    m[y:y + side, x:x + side] = True
    return m


def disc(cx, cy, r, w=W, h=H):
    yy, xx = np.mgrid[:h, :w]
    return (xx - cx) ** 2 + (yy - cy) ** 2 <= r * r


def ell(x, y, w=W, h=H):
    m = np.zeros((h, w), dtype=bool)
    m[y:y + 20, x:x + 6] = True
    m[y + 14:y + 20, x:x + 16] = True
    return m


def ring(cx, cy, outer, inner, w=W, h=H):
    return disc(cx, cy, outer, w, h) & ~disc(cx, cy, inner, w, h)


# Each frame: [(object class, bool mask)], painted in order (a later object hides an earlier one where they overlap).
FRAMES = [
    [('cube', square(4, 4, 12)), ('sphere', disc(40, 30, 12)), ('cylinder', ell(70, 10))],
    [('sphere', ring(30, 30, 16, 8)), ('cube', square(60, 5, 10) | square(80, 40, 4))],      # a hole; two pieces
    [('cube', square(10, 10, 10)), ('cylinder', square(20, 10, 10)),                          # touching
     ('sphere', square(0, 50, 14) | square(82, 0, 14))],                                     # on the border
    [('cube', square(40, 40, 3))],                                                           # tiny: dropped
]


def frame_arrays(objects):
    ids = np.full((H, W), 1, dtype=np.uint32)                       # the floor: an unlabelled prim
    id_to_prim, labelled = {1: '/World/floor'}, {}
    for k, (cls, mask) in enumerate(objects):
        raw = 10 + k
        ids[mask] = raw
        id_to_prim[raw] = f'/World/parts/o{k:02d}/mesh'
        labelled[f'/World/parts/o{k:02d}'] = {'object': f'o{k:02d}', 'class': cls}
    rgb = np.zeros((H, W, 3), dtype=np.uint8)
    rgb[..., 1] = (ids % 7 * 30).astype(np.uint8)
    return rgb, ids, id_to_prim, labelled


def make_shapes_session(store, frames=FRAMES):
    runs = len(frames)
    ingest = ReplicatorIngest(
        store, sensor_id='bin_camera', frame_id='bin_camera_optical_frame',
        config={'generator': 'example_shapes', 'resolution': [W, H], 'classes': CLASSES, 'runs': runs},
        master_seed=3, recipe='example_shapes/1', runs=runs, views_per_run=1, scene=scene(),
        camera_params=camera_params(W, H), camera_prim='/World/Camera',
        environment={'isaac_sim': 'example-5.1'}, software={'example_generator': '1.0'})
    for run_index, objects in enumerate(frames):
        rgb, ids, id_to_prim, labelled = frame_arrays(objects)
        ingest.add_frame(run_index=run_index, view=0, draws={'frame': run_index}, view_draws={},
                         sim_time_s=round((run_index + 1) / 60, 9), rgb=rgb, depth=None, instance_ids=ids,
                         id_to_prim=id_to_prim, objects=labelled)
    return ingest.close()


@pytest.fixture
def shapes(store):
    header = make_shapes_session(store)
    create_dataset(store, 'shapes', name='Shapes')
    ls = create_synthetic_label_set(store, 'shapes', [header.session_id], 'INSTANCE_MASK')
    create_version(store, 'shapes', sessions=[header.session_id], label_sets=[ls.label_set_id],
                   group_by='randomization_run', ratios={'train': 1.0, 'val': 0.0, 'test': 0.0}, seed=1)
    return store, header, ls


def rasterize(points, w, h):
    """Ultralytics' polygon2mask at mask ratio 1 (int truncation + an inclusive polygon fill), in Pillow."""
    image = Image.new('L', (w, h), 0)
    ImageDraw.Draw(image).polygon([(int(x * w), int(y * h)) for x, y in points], fill=1, outline=1)
    return np.asarray(image).astype(bool)


def iou(a, b):
    return np.count_nonzero(a & b) / np.count_nonzero(a | b)


def fill_holes(mask):
    """The mask with every background region not connected (4-neighbours) to the border filled -- reference only."""
    outside = np.zeros_like(mask)
    outside[0, :], outside[-1, :], outside[:, 0], outside[:, -1] = ~mask[0, :], ~mask[-1, :], ~mask[:, 0], ~mask[:, -1]
    while True:
        grown = outside.copy()
        grown[1:, :] |= outside[:-1, :]
        grown[:-1, :] |= outside[1:, :]
        grown[:, 1:] |= outside[:, :-1]
        grown[:, :-1] |= outside[:, 1:]
        grown &= ~mask
        if (grown == outside).all():
            return ~outside
        outside = grown


# ---- the polygon policy (pure) ------------------------------------------------------------------------------------

def test_a_square_is_its_four_corner_pixels_and_rasterizes_back_exactly():
    mask = square(5, 7, 10)
    vertices, info = instance_polygon(mask, 1.0)
    assert vertices == [(5, 7), (14, 7), (14, 16), (5, 16)]                  # clockwise from the top-left pixel
    assert info == {'components': 1, 'pixels': 100, 'dropped_pixels': 0}
    points = [((c + 0.5) / W, (r + 0.5) / H) for c, r in vertices]
    assert (rasterize(points, W, H) == mask).all()


def test_an_l_shape_keeps_its_corners():
    # the inner corner is a one-pixel diagonal step (15, 18) -> (16, 19): within 1 px of the chord, within 0.5 px not
    assert instance_polygon(ell(10, 5), 1.0)[0] == [(10, 5), (15, 5), (15, 18), (25, 19), (25, 24), (10, 24)]
    assert instance_polygon(ell(10, 5), 0.5)[0] == [(10, 5), (15, 5), (15, 18), (16, 19), (25, 19), (25, 24), (10, 24)]


@pytest.mark.parametrize('radius', [10, 11, 13, 20, 31])
def test_discs_round_trip_with_a_high_iou_at_the_default_epsilon(radius):
    """IoU >= 0.95 for every disc of radius >= 10 px, wherever its centre falls inside a pixel. (epsilon 1.0 would
    not hold it: its chords cut off up to ~8 % of a radius-10 disc -- hence the default 0.5.)"""
    rng = np.random.default_rng(radius)
    w = h = 2 * radius + 11
    for _ in range(12):
        mask = disc(w / 2 + rng.uniform(-1, 1), h / 2 + rng.uniform(-1, 1), radius, w, h)
        vertices, info = instance_polygon(mask, DEFAULT_EPSILON_PX)
        assert info['components'] == 1 and len(vertices) >= 8
        points = [(float(f'{(c + 0.5) / w:.6f}'), float(f'{(r + 0.5) / h:.6f}')) for c, r in vertices]
        assert iou(rasterize(points, w, h), mask) >= 0.95


def test_the_unsimplified_boundary_rasterizes_back_to_the_filled_component_exactly():
    """With epsilon 0 only collinear boundary pixels go: the round trip is exact, holes filled (random blobs)."""
    rng = np.random.default_rng(5)
    for _ in range(60):
        mask = rng.random((24, 30)) < 0.55
        component, _, _ = largest_component(mask)
        vertices, info = instance_polygon(mask, 0.0)
        if vertices is None:
            continue
        points = [(float(f'{(c + 0.5) / 30:.6f}'), float(f'{(r + 0.5) / 24:.6f}')) for c, r in vertices]
        assert (rasterize(points, 30, 24) == fill_holes(component)).all()


def test_a_ring_exports_its_outer_boundary_only():
    outer, _ = instance_polygon(disc(30, 30, 16), 1.0)
    vertices, info = instance_polygon(ring(30, 30, 16, 8), 1.0)
    assert vertices == outer and info['components'] == 1


def test_the_largest_of_two_components_is_kept_and_the_other_counted():
    vertices, info = instance_polygon(square(60, 5, 10) | square(80, 40, 4), 1.0)
    assert vertices == [(60, 5), (69, 5), (69, 14), (60, 14)]
    assert info == {'components': 2, 'pixels': 100, 'dropped_pixels': 16}
    # a tie goes to the component met first in raster order
    _, tie = instance_polygon(square(50, 30, 5) | square(10, 2, 5), 1.0)
    assert instance_polygon(square(50, 30, 5) | square(10, 2, 5), 1.0)[0][0] == (10, 2) and tie['dropped_pixels'] == 25


def test_diagonal_pixels_are_one_8_connected_component():
    mask = np.zeros((6, 6), dtype=bool)
    mask[1, 1] = mask[2, 2] = mask[3, 3] = True
    component, count, dropped = largest_component(mask)
    assert count == 1 and dropped == 0 and component.sum() == 3


def test_degenerate_instances_have_no_polygon():
    line = np.zeros((10, 30), dtype=bool)
    line[4, 2:25] = True
    assert instance_polygon(line, 1.0)[0] is None                            # a one-pixel line: zero area
    single = np.zeros((5, 5), dtype=bool)
    single[2, 2] = True
    assert trace_outer_boundary(single) == [(2, 2)] and instance_polygon(single, 1.0)[0] is None
    assert instance_polygon(np.zeros((5, 5), dtype=bool), 1.0) == (None, {'components': 0, 'pixels': 0,
                                                                          'dropped_pixels': 0})
    assert simplify_closed([(0, 0), (1, 1)], 1.0) == [(0, 0), (1, 1)] and doubled_area([(0, 0), (2, 0), (2, 2)]) == 4


def test_segment_label_lines_refuse_an_unclassified_mask_id():
    mask = np.zeros((H, W), dtype=np.uint16)
    mask[square(4, 4, 12)] = 1
    mask[disc(40, 30, 12)] = 2
    with pytest.raises(DataError, match=r'mask ids \[2\] have no class') as e:
        segment_label_lines(mask, [{'id': 1, 'class': 'cube'}], CLASSES, W, H, 1.0, 16, 'here')
    assert e.value.code == 'EXPORT_REFUSED'
    data, count, records = segment_label_lines(mask, [{'id': 1, 'class': 'cube'}, {'id': 2, 'class': 'sphere'}],
                                               CLASSES, W, H, 1.0, 16, 'here')
    assert count == 2 and data.decode().startswith('0 0.046875 0.070312 0.161458 0.070312 ')
    assert [r['vertices'] for r in records][0] == 4


# ---- the export -----------------------------------------------------------------------------------------------------

def _obs(header, frame):
    return f'{header.session_id}/{frame + 1:06d}/bin_camera/color'           # capture sequence numbers start at 1


def _labels(out, header, frame):
    obs = _obs(header, frame)
    return (out / 'labels' / 'train' / f'{stem_of(obs)}.txt').read_bytes(), obs


def test_the_segment_export_layout_content_and_sidecar(shapes, tmp_path):
    store, header, ls = shapes
    out = tmp_path / 'out'
    sidecar = YOLO.export(store, 'shapes', 1, out, {'task': 'segment'})
    files = snapshot(out)
    stems = [stem_of(_obs(header, frame)) for frame in range(4)]
    assert sorted(files) == sorted([f'images/train/{s}.png' for s in stems] + [f'labels/train/{s}.txt' for s in stems]
                                   + ['data.yaml', SIDECAR])
    assert sidecar['exporter']['name'] == 'yolo' and sidecar['exporter']['version'] == '1.1'
    assert sidecar['exporter']['config'] == {'task': 'segment', 'label_sets': [ls.label_set_id],
                                             'allow_mixed_provenance': False, 'epsilon_px': 0.5, 'min_area_px': 16}
    summary = sidecar['summary']
    assert summary['task'] == 'segment' and summary['classes'] == CLASSES
    assert summary['splits'] == {'train': {'images': 4, 'instances': 8}, 'val': {'images': 0, 'instances': 0},
                                 'test': {'images': 0, 'instances': 0}}
    polygons = summary['polygons']
    assert polygons['policy'] == dict(POLICY, epsilon_px=0.5, min_area_px=16)
    assert {k: v for k, v in polygons.items() if k not in ('policy', 'vertices')} == {
        'instances': 9, 'exported': 8, 'multi_component_instances': 2, 'dropped_components': 2,
        'dropped_component_pixels': 16 + 14 * 14, 'dropped_instances': 1,
        'dropped': [{'observation_id': _obs(header, 3), 'id': 1, 'class': 'cube',
                     'reason': 'BELOW_MIN_AREA', 'pixels': 9}]}
    data_yaml = files['data.yaml'].decode()
    assert data_yaml.startswith(f'# TrainIt YOLO segmentation export {sidecar["export_id"]}')
    assert "train: images/train\nnc: 3\nnames:\n  0: 'cube'\n  1: 'cylinder'\n  2: 'sphere'\n" in data_yaml

    # frame 0: a square, an L (cylinder), a disc -- in compact-id (= object key) order
    text, _ = _labels(out, header, 0)
    lines = parse_segment_file(text, 3, 'f0')
    assert [cls for cls, _ in lines] == [0, 2, 1]
    assert lines[0][1] == [(0.046875, 0.070312), (0.161458, 0.070312), (0.161458, 0.242188), (0.046875, 0.242188)]
    assert len(lines[2][1]) == 7                                            # the L at epsilon 0.5 (its inner step kept)
    assert verify_export(out, store).ok


def test_the_exported_polygons_rasterize_back_onto_the_masks(shapes, tmp_path):
    store, header, ls = shapes
    out = tmp_path / 'out'
    YOLO.export(store, 'shapes', 1, out, {'task': 'segment'})
    for seq, objects in enumerate(FRAMES):
        text, obs = _labels(out, header, seq)
        polygons = parse_segment_file(text, 3, obs)
        painted = np.zeros((H, W), dtype=np.int32)
        for k, (_, mask) in enumerate(objects):
            painted[mask] = k + 1                                          # what is visible of each object
        visible = [painted == k + 1 for k in range(len(objects))]
        kept = [m for m in visible if largest_component(m)[0].sum() >= 16]
        assert len(polygons) == len(kept)
        for (cls, points), mask in zip(polygons, kept):
            component = fill_holes(largest_component(mask)[0])
            assert iou(rasterize(points, W, H), component) >= 0.9, (seq, cls)


def test_touching_and_border_instances(shapes, tmp_path):
    store, header, _ = shapes
    out = tmp_path / 'out'
    YOLO.export(store, 'shapes', 1, out, {'task': 'segment'})
    text, _ = _labels(out, header, 2)
    lines = parse_segment_file(text, 3, 'f2')
    assert [cls for cls, _ in lines] == [0, 1, 2]
    left, right, border = (pts for _, pts in lines)
    assert max(x for x, _ in left) < min(x for x, _ in right)               # touching, not overlapping
    # the two 14 x 14 border squares tie: the one met first in raster order (top-right) is kept
    assert max(x for x, _ in border) == 0.994792 and min(y for _, y in border) == 0.007812   # column 95 / row 0
    assert max(y for _, y in border) == 0.210938


def test_the_tiny_instance_leaves_an_empty_label_file(shapes, tmp_path):
    store, header, _ = shapes
    out = tmp_path / 'out'
    YOLO.export(store, 'shapes', 1, out, {'task': 'segment'})
    assert _labels(out, header, 3)[0] == b''
    small = YOLO.export(store, 'shapes', 1, tmp_path / 'small', {'task': 'segment', 'min_area_px': 4})
    assert small['summary']['polygons']['dropped'] == [] and small['summary']['splits']['train']['instances'] == 9


def test_two_segment_exports_are_byte_identical_and_options_change_the_id(shapes, tmp_path):
    store = shapes[0]
    first = YOLO.export(store, 'shapes', 1, tmp_path / 'a', {'task': 'segment'})
    second = YOLO.export(store, 'shapes', 1, tmp_path / 'deeper' / 'b', {'task': 'segment', 'epsilon_px': 0.5})
    assert snapshot(tmp_path / 'a') == snapshot(tmp_path / 'deeper' / 'b') and first == second
    coarse = YOLO.export(store, 'shapes', 1, tmp_path / 'c', {'task': 'segment', 'epsilon_px': 3.0})
    assert coarse['export_id'] != first['export_id']
    assert coarse['summary']['polygons']['vertices'] < first['summary']['polygons']['vertices']
    assert verify_export(tmp_path / 'c', store).ok


def test_verify_export_detects_a_tampered_segment_label_file(shapes, tmp_path):
    store, header, _ = shapes
    out = tmp_path / 'out'
    YOLO.export(store, 'shapes', 1, out, {'task': 'segment'})
    label = out / 'labels' / 'train' / f'{stem_of(_obs(header, 0))}.txt'
    rel = label.relative_to(out).as_posix()
    original = label.read_bytes()
    label.write_bytes(original.replace(b'0.046875', b'0.050000', 1))            # a moved vertex
    findings = verify_export(out, store).findings
    assert {(f.code, f.artifact) for f in findings} >= {('HASH_MISMATCH', rel)}
    label.write_bytes(b'0 0.1 0.1 0.2 0.2\n')                                    # a box line in a segment export
    findings = verify_export(out).findings
    assert any(f.code == 'LABEL_INVALID' and f.artifact == rel for f in findings)
    label.write_bytes(b'0 0.1 0.1 0.2 0.2 0.3 0.3\n')                            # three collinear vertices
    assert any('zero area' in f.detail for f in verify_export(out).findings)
    label.write_bytes(original)
    assert verify_export(out, store).ok
    sidecar = json.loads((out / SIDECAR).read_text())
    sidecar['summary']['polygons']['exported'] = 7                               # a summary that lies
    (out / SIDECAR).write_text(json.dumps(sidecar))
    assert not verify_export(out).ok


def test_the_detect_config_is_unchanged_and_refuses_segment_options(shapes, tmp_path):
    store, header, ls = shapes
    boxes = create_synthetic_label_set(store, 'shapes', [header.session_id], 'BBOX_2D', box_source='MASK_EXTENT')
    create_version(store, 'shapes', op='RELABEL', parent=1, label_sets=[boxes.label_set_id])
    sidecar = YOLO.export(store, 'shapes', 2, tmp_path / 'det')
    assert sidecar['exporter']['config'] == {'task': 'detect', 'label_sets': [boxes.label_set_id],
                                             'allow_mixed_provenance': False}
    assert set(sidecar['summary']) == {'task', 'classes', 'splits'}
    seg = YOLO.export(store, 'shapes', 2, tmp_path / 'seg', {'task': 'segment'})
    assert seg['exporter']['config']['label_sets'] == [ls.label_set_id]           # the INSTANCE_MASK set only
    with pytest.raises(DataError, match='options of task segment'):
        YOLO.export(store, 'shapes', 2, tmp_path / 'x', {'min_area_px': 4})
    for bad in ({'epsilon_px': -1.0}, {'min_area_px': 0}, {'epsilon_px': 'one'}):
        with pytest.raises(DataError) as e:
            YOLO.export(store, 'shapes', 2, tmp_path / 'x', dict(bad, task='segment'))
        assert e.value.code == 'INVALID_VALUE'
    assert not (tmp_path / 'x').exists()


def test_the_cli_exports_and_verifies_a_segment_export(shapes, tmp_path, capsys):
    store, _, ls = shapes
    out = tmp_path / 'cli'
    code = main(['--root', str(store.root), '--json', 'export', 'yolo', 'shapes', '1', str(out), '--task', 'segment',
                 '--epsilon-px', '1.5', '--min-area-px', '8'])
    doc = json.loads(capsys.readouterr().out)
    assert code == 0 and doc['result']['export']['exporter']['config'] == {
        'task': 'segment', 'label_sets': [ls.label_set_id], 'allow_mixed_provenance': False, 'epsilon_px': 1.5,
        'min_area_px': 8}
    assert doc['result']['export']['summary']['polygons']['policy']['epsilon_px'] == 1.5
    code = main(['--root', str(store.root), 'verify-export', str(out)])
    text = capsys.readouterr().out
    assert code == 0 and 'verify-export: OK' in text
    code = main(['--root', str(store.root), 'export', 'yolo', 'shapes', '1', str(tmp_path / 'b'), '--task', 'segment'])
    text = capsys.readouterr().out
    assert code == 0 and 'polygon(s)' in text and '1 dropped' in text
    code = main(['--root', str(store.root), '--json', 'export', 'yolo', 'shapes', '1', str(tmp_path / 'c'),
                 '--epsilon-px', '2'])
    assert code == 1 and json.loads(capsys.readouterr().out)['error']['code'] == 'EXPORT_REFUSED'


def test_the_mask_entries_the_export_reads(shapes):
    """The fixture is what the tests assume: compact ids in object-key order, the tiny instance kept by the labels."""
    store, header, ls = shapes
    entries = read_entries(store, ls)
    assert [[i['class'] for i in e['instances']] for e in entries] == [['cube', 'sphere', 'cylinder'],
                                                                         ['sphere', 'cube'],
                                                                         ['cube', 'cylinder', 'sphere'], ['cube']]


def test_the_detection_output_is_the_1_0_output_and_a_1_0_export_still_verifies(shapes, tmp_path, monkeypatch):
    """Exporter 1.1 changes no detection byte but the exporter version (and so the export id in the sidecar and in
    data.yaml's comment line); an export made by 1.0 still verifies, with a WARNING that 1.1 cannot regenerate it."""
    from trainit_data.formats import yolo
    store, header, _ = shapes
    boxes = create_synthetic_label_set(store, 'shapes', [header.session_id], 'BBOX_2D', box_source='MASK_EXTENT')
    create_version(store, 'shapes', op='RELABEL', parent=1, label_sets=[boxes.label_set_id])
    new = YOLO.export(store, 'shapes', 2, tmp_path / 'new')
    with monkeypatch.context() as m:
        m.setattr(yolo, 'VERSION', '1.0')
        old = YOLO.export(store, 'shapes', 2, tmp_path / 'old')
    assert old['exporter']['version'] == '1.0' and old['export_id'] != new['export_id']
    a, b = snapshot(tmp_path / 'old'), snapshot(tmp_path / 'new')
    assert {k for k in a if a[k] != b[k]} == {'data.yaml', SIDECAR}
    assert a['data.yaml'].split(b'\n')[1:] == b['data.yaml'].split(b'\n')[1:]
    report = verify_export(tmp_path / 'old', store)
    assert report.ok and [f.severity for f in report.findings] == ['WARNING']


def _frame_label_bytes(epsilon):
    out = b''
    for objects in FRAMES:
        mask = np.zeros((H, W), dtype=np.uint16)
        for k, (_, m) in enumerate(objects):
            mask[m] = k + 1
        instances = [{'id': k + 1, 'class': cls} for k, (cls, _) in enumerate(objects)]
        out += segment_label_lines(mask, instances, CLASSES, W, H, epsilon, 16, 'pinned')[0] + b'--\n'
    return out


def test_the_label_bytes_are_pinned_across_interpreters():
    """A literal hash: run in the ROS Python 3.10 AND Isaac Lab's 3.11, it proves the polygons and their 6-decimal
    text are identical there (integer geometry, correctly rounded division and formatting). It changes only with an
    intended change of the polygon policy (a new policy version), shown to the owner."""
    from trainit_data._canonical import sha256_bytes
    assert sha256_bytes(_frame_label_bytes(0.5)) == PINNED_05
    assert sha256_bytes(_frame_label_bytes(1.5)) == PINNED_15
