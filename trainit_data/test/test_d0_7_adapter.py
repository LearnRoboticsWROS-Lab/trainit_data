"""D0.7: the pure Isaac Replicator ingest adapter -- annotator-shaped numpy -> one canonical synthetic session.

No Isaac is needed (or allowed) here: the fixtures (``_d07.py``) produce Replicator-SHAPED output in plain numpy.
"""
import json
import math
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from _d07 import (FRAME, RECIPE, SENSOR, H, W, camera_params, draws_for, make_synthetic_session, render,
                  scene)
from trainit_data.adapters.isaac_replicator import (ReplicatorIngest, compact_instances,
                                                    intrinsics_from_camera_params, optical_pose_from_camera_params)
from trainit_data.errors import DataError
from trainit_data.integrity import verify_session
from trainit_data.observation import mask_extent
from trainit_data.synthetic import frame_of, run_id, run_seed

PKG_ROOT = Path(__file__).resolve().parents[1]


# ---- the dependency boundary ------------------------------------------------------------------------------------

def test_the_adapter_imports_without_isaac_and_loads_no_simulator_module():
    code = ('import sys, trainit_data.adapters.isaac_replicator, trainit_data.ground_truth, trainit_data.synthetic\n'
            'bad = sorted(m for m in sys.modules if m.split(".")[0] in {"omni", "isaacsim", "carb", "pxr", "cv2", '
            '"torch", "ultralytics", "lerobot", "datasets", "rclpy"})\n'
            'assert not bad, bad\nprint("clean")')
    env = {k: v for k, v in os.environ.items() if k != 'PYTHONPATH'}
    env['PYTHONPATH'] = str(PKG_ROOT)
    out = subprocess.run([sys.executable, '-c', code], capture_output=True, text=True, env=env, cwd=PKG_ROOT)
    assert out.returncode == 0 and out.stdout.strip() == 'clean', out.stderr


# ---- camera_params -> calibration ---------------------------------------------------------------------------------

def test_intrinsics_come_from_focal_length_and_aperture_and_match_the_projection():
    intr = intrinsics_from_camera_params(camera_params())
    fx = 18.0 * W / 20.955
    assert intr['width'] == W and intr['height'] == H and intr['distortion_model'] == 'plumb_bob'
    assert math.isclose(intr['K'][0], fx) and math.isclose(intr['K'][4], fx)          # square pixels
    assert intr['K'][2] == W / 2 and intr['K'][5] == H / 2                              # pixel-edge centre
    assert intr['P'][:4] == [intr['K'][0], 0.0, W / 2, 0.0] and intr['D'] == [0.0] * 5


@pytest.mark.parametrize('override, needle', [
    ({'cameraModel': 'fisheyePolynomial'}, 'pinhole'),
    ({'cameraApertureOffset': [0.5, 0.0]}, 'aperture offset'),
    ({'cameraProjection': [3.0] + [0.0] * 15}, 'projection'),
    ({'cameraFocalLength': 0.0}, 'positive'),
    ({'renderProductResolution': [W]}, 'renderProductResolution'),
])
def test_camera_params_that_are_not_a_plain_pinhole_are_refused(override, needle):
    with pytest.raises(DataError, match=needle) as e:
        intrinsics_from_camera_params(camera_params(**override))
    assert e.value.code == 'INVALID_VALUE'


def test_the_extrinsic_is_the_optical_frame_pose_in_the_world():
    # A USD camera 1 m above the origin looking down (-z): the optical frame (+z forward, +y down) is the camera
    # turned 180 degrees about x.
    translation, rotation = optical_pose_from_camera_params(camera_params(z=1.0))
    assert translation == [0.0, 0.0, 1.0] and rotation == [1.0, 0.0, 0.0, 0.0]
    # A camera rotated 90 degrees about z (row-vector view transform of its inverse) keeps a unit quaternion
    c, s_ = 0.0, 1.0
    view = [c, -s_, 0, 0, s_, c, 0, 0, 0, 0, 1, 0, 0, 0, -2.0, 1]
    translation, rotation = optical_pose_from_camera_params(camera_params(cameraViewTransform=view))
    assert math.isclose(sum(v * v for v in rotation), 1.0) and rotation[3] >= 0.0
    assert translation == [0.0, 0.0, 2.0]
    with pytest.raises(DataError, match='rigid'):
        optical_pose_from_camera_params(camera_params(cameraViewTransform=[2, 0, 0, 0, 0, 2, 0, 0, 0, 0, 2, 0, 0, 0,
                                                                           0, 1]))


# ---- annotator output -> compact instances ------------------------------------------------------------------------

def test_raw_instance_ids_become_compact_ids_in_object_key_order_whatever_replicator_numbered():
    draws = draws_for(7, 0)
    masks = []
    for offset in (1, 500):                                       # Replicator renumbers between runs
        _, _, ids, id_to_prim, objects, boxes = render(draws, 0, offset)
        mask, instances = compact_instances(ids, id_to_prim, objects, boxes)
        masks.append((mask, instances))
    assert np.array_equal(masks[0][0], masks[1][0]) and masks[0][1] == masks[1][1]
    mask, instances = masks[0]
    assert [i['id'] for i in instances] == list(range(1, len(draws['objects']) + 1))
    assert [i['object'] for i in instances] == sorted(o['object'] for o in draws['objects'])
    assert set(np.unique(mask)) == {0} | {i['id'] for i in instances}              # the floor is 0
    for inst, obj in zip(instances, sorted(draws['objects'], key=lambda o: o['object'])):
        assert inst['class'] == obj['class'] and inst['box_tight_xywh_px'] == obj['rect_px']
        assert inst['pixels'] == obj['rect_px'][2] * obj['rect_px'][3]
        assert mask_extent(mask == inst['id']) == obj['rect_px']


def test_an_object_made_of_several_prims_is_one_instance_and_invisible_objects_are_left_out():
    ids = np.zeros((4, 6), dtype=np.uint32)
    ids[0:2, 0:2], ids[2:4, 0:2] = 5, 9                           # two meshes of one object
    id_to_prim = {5: '/W/a/top', 9: '/W/a/bottom', 3: '/W/hidden/mesh'}
    objects = {'/W/a': {'object': 'a', 'class': 'cube'}, '/W/hidden': {'object': 'b', 'class': 'cube'}}
    mask, instances = compact_instances(ids, id_to_prim, objects)
    assert instances == [{'id': 1, 'object': 'a', 'class': 'cube', 'pixels': 8}]
    assert int(mask.sum()) == 8


# ---- one canonical session ----------------------------------------------------------------------------------------

def test_a_generation_is_one_ordinary_canonical_session_that_verifies(store):
    header = make_synthetic_session(store, runs=3, views=2)
    assert verify_session(store, header.session_id).ok
    assert header.source.transport == 'INGEST' and header.source.backend == 'isaac'
    assert header.source.adapter == 'trainit_data.adapters.isaac_replicator'
    assert header.clock.session_domain == 'SIM_TIME' and header.clock.sim_source == 'isaac'
    assert [st.modality for st in header.sensors[0].streams] == ['rgb', 'depth', 'instance_segmentation']
    gen = header.provenance['randomization']
    assert gen['generator'] == 'isaac_replicator' and gen['recipe'] == RECIPE and gen['master_seed'] == 7
    assert gen['config_sha256'] == header.recording_config['resolved_sha256']
    assert gen['reproducibility'] == {'claim': 'SAME_DRAWS', 'render_bit_identical': False,
                                      'environment': {'isaac_sim': 'example-5.1', 'replicator': 'example-1.12',
                                                      'numpy': np.__version__}}
    assert header.provenance['scene']['base_stage']['completeness'] == 'RESOLVED_LAYERS'
    assert header.counters['captures'] == {'attempted': 6, 'complete': 6, 'partial': 0, 'failed': 0}

    reader = store.open_session(header.session_id)
    captures = list(reader.captures())
    for n, capture in enumerate(captures):
        context, reason = frame_of(capture)
        assert (context['run_index'], context['view']) == divmod(n, 2)
        assert context['run'] == run_id(RECIPE, reason['draws'])
        assert reason['run_seed'] == run_seed(7, context['run_index'])
        assert reason['draws'] == draws_for(7, context['run_index'])
        assert capture.trigger.kind == 'SYNTHETIC_FRAME' and capture.trigger.origin == 'GENERATOR'
        assert capture.sync.policy.name == 'SAME_RENDER_TICK' and capture.sync.status == 'SYNCED'
        color, depth, inst = capture.observations
        assert color.data.encoding == 'rgb8' and [t.op for t in color.data.transforms] == ['drop_alpha']
        assert depth.data.encoding == '32FC1' and depth.data.unit_scale_m == 1.0
        assert depth.quality.invalid_depth_fraction == round(1 / (W * H), 6)
        assert inst.data.block == 'instance_image' and inst.payload.media_type == 'image/png;bits=16'
        assert inst.data.annotator == 'instance_id_segmentation'
        assert inst.data.box_annotator == 'bounding_box_2d_tight'
        assert color.stamp.clock == ('SIM_TIME', 'isaac') and color.stamp.origin == 'RENDER_TICK'
        assert color.stamp.to_ns() == depth.stamp.to_ns() == inst.stamp.to_ns()

    cals = {cid: store.get_calibration(cid, reader.vocab) for cid in header.calibration_ids}
    assert sorted(c.stream for c in cals.values()) == ['color', 'depth', 'instances']
    for cal in cals.values():
        assert cal.intrinsics_provenance.kind == 'SYNTHETIC_RENDER'
        assert [e.provenance.kind for e in cal.extrinsics] == ['SYNTHETIC_SCENE']
        assert cal.extrinsics[0].provenance.ref['stage_sha256'] == header.provenance['scene']['base_stage']['sha256']
        assert cal.optical_frame == FRAME and cal.sensor_id == SENSOR
    depth_cal = next(c for c in cals.values() if c.stream == 'depth')
    assert depth_cal.depth.registered_to == 'color' and depth_cal.depth.invalid == 'nan_or_inf'


def test_one_run_shares_one_run_id_and_the_same_draws_are_the_same_run_anywhere(store):
    a = make_synthetic_session(store, runs=3, views=2)
    b = make_synthetic_session(store, runs=3, views=2, raw_offset=900)
    runs = []
    for header in (a, b):
        captures = list(store.open_session(header.session_id).captures())
        runs.append([frame_of(c)[0]['run'] for c in captures])
    assert runs[0] == runs[1] and len(set(runs[0])) == 3
    assert runs[0][0] == runs[0][1] != runs[0][2]


# ---- refusals: a contradiction is never written ---------------------------------------------------------------------

@pytest.fixture
def ingest(store):
    return make_synthetic_session(store, runs=3, views=2, close=False, record_runs=1)     # run 0 recorded


def _frame(run_index=1, view=0, **over):
    draws = draws_for(7, run_index)
    rgb, depth, ids, id_to_prim, objects, boxes = render(draws, view, 1)
    frame = dict(run_index=run_index, view=view, draws=draws, view_draws={}, sim_time_s=1.0, rgb=rgb, depth=depth,
                 instance_ids=ids, id_to_prim=id_to_prim, objects=objects, tight_boxes=boxes)
    frame.update(over)
    return frame


@pytest.mark.parametrize('change, needle', [
    (lambda f: f['tight_boxes'][0].update(x_max=f['tight_boxes'][0]['x_max'] + 3), 'differ by more than'),
    (lambda f: f['objects'][next(iter(f['objects']))].update({'class': 'pyramid'}), 'semantic classes'),
    (lambda f: f.update(view=5), 'outside the declared'),
    (lambda f: f.update(sim_time_s=-1.0), 'time'),
    (lambda f: f.update(rgb=np.zeros((H, W + 1, 4), np.uint8)), 'render product'),
    (lambda f: f.update(draws={'objects': []}), 'other draws'),
    (lambda f: f.update(camera_params=camera_params(z=2.0)), 'camera moved'),
])
def test_a_frame_that_contradicts_itself_or_the_session_is_refused_before_any_byte(store, ingest, change, needle):
    frame = _frame(run_index=0, view=1) if needle == 'other draws' else _frame()
    change(frame)
    before = sorted(p.relative_to(store.root) for p in store.root.rglob('*'))
    with pytest.raises(DataError, match=needle) as e:
        ingest.add_frame(**frame)
    assert e.value.code == 'INVALID_VALUE'
    assert sorted(p.relative_to(store.root) for p in store.root.rglob('*')) == before
    ingest.close()


def test_a_view_is_recorded_once(ingest):
    with pytest.raises(DataError, match='already recorded'):
        ingest.add_frame(**_frame(run_index=0, view=0))
    ingest.close()


def test_the_scene_manifest_never_names_an_absolute_machine_path(store):
    bad = scene(assets=[{'id': '/home/someone/assets/part.usd', 'sha256': 'sha256:' + '0' * 64}])
    with pytest.raises(DataError, match='absolute'):
        ReplicatorIngest(store, sensor_id=SENSOR, frame_id=FRAME, config={}, master_seed=1, recipe=RECIPE, runs=1,
                         views_per_run=1, scene=bad, camera_params=camera_params(), camera_prim='/World/Camera',
                         environment={})
    partial = scene(base_stage={'sha256': 'sha256:' + '1' * 64, 'completeness': 'RESOLVED_LAYERS', 'layers': [],
                                'unresolved': ['props/part.usd']})
    with pytest.raises(DataError, match='PARTIAL'):
        ReplicatorIngest(store, sensor_id=SENSOR, frame_id=FRAME, config={}, master_seed=1, recipe=RECIPE, runs=1,
                         views_per_run=1, scene=partial, camera_params=camera_params(), camera_prim='/World/Camera',
                         environment={})


# ---- verify sees what the hashes alone cannot ---------------------------------------------------------------------

def _edit_header(store, session_id, mutate):
    path = store.path(f'sessions/{session_id}/session.json')
    doc = json.loads(path.read_text())
    mutate(doc)
    os.chmod(path, 0o644)
    path.write_text(json.dumps(doc))


@pytest.mark.parametrize('mutate, needle', [
    (lambda d: d['provenance']['randomization'].update(config_sha256='sha256:' + '0' * 64), 'config_sha256'),
    (lambda d: d['provenance']['randomization'].update(runs=1), 'runs'),
    (lambda d: d['provenance']['randomization'].update(recipe='other/1'), 'recipe'),
    (lambda d: d['provenance']['scene'].update(kind='SOMETHING'), 'generation provenance'),
])
def test_verify_checks_the_generation_rules(store, mutate, needle):
    header = make_synthetic_session(store, runs=2, views=2)
    _edit_header(store, header.session_id, mutate)
    report = verify_session(store, header.session_id)
    assert not report.ok and any(needle in f.detail for f in report.findings)


def test_verify_decodes_every_instance_mask_against_its_block(store):
    header = make_synthetic_session(store, runs=1, views=1)
    captures = list(store.open_session(header.session_id).captures())
    inst = captures[0].observations[2]
    block = inst.data
    mask = np.zeros((H, W), dtype=np.uint16)
    assert any('mask ids' in p for p in block.mask_problems(mask))
    assert block.mask_problems(np.zeros((H + 1, W), np.uint16))[0].startswith('the mask is')
    from trainit_data.labels import load_mask
    real = load_mask(store, inst, 'x')
    assert block.mask_problems(real) == []
    shifted = real.copy()
    first = block.instances[0]
    x, y, w, h = first['box_tight_xywh_px']
    shifted[y, x + w: x + w + 3] = first['id']                    # 3 more columns: pixel count and extent change
    problems = block.mask_problems(shifted)
    assert any('pixels' in p for p in problems) and any('differ by more than' in p for p in problems)


# ---- review findings (2026-11-02) ----------------------------------------------------------------------------------

@pytest.mark.parametrize('noise', [1e-12, -1e-12, 0.0])
def test_the_calibration_identity_never_depends_on_floating_point_noise(noise):
    """A straight-down camera is 180 degrees about x (w = 0): the sign is chosen AFTER rounding."""
    view = [1, 0, 0, 0, 0, 1, noise, 0, 0, -noise, 1, 0, 0, 0, -0.6, 1]
    assert optical_pose_from_camera_params(camera_params(cameraViewTransform=view)) == ([0.0, 0.0, 0.6],
                                                                                        [1.0, 0.0, 0.0, 0.0])


def test_an_object_of_several_prims_gets_the_union_of_its_boxes():
    ids = np.zeros((6, 6), dtype=np.uint32)
    ids[0:2, 0:2], ids[3:6, 3:6] = 5, 9
    objects = {'/W/a': {'object': 'a', 'class': 'cube'}, '/W/a_lid': {'object': 'a', 'class': 'cube'}}
    boxes = [{'prim': '/W/a', 'x_min': 0, 'y_min': 0, 'x_max': 1, 'y_max': 1},
             {'prim': '/W/a_lid', 'x_min': 3, 'y_min': 3, 'x_max': 5, 'y_max': 5}]
    mask, instances = compact_instances(ids, {5: '/W/a', 9: '/W/a_lid'}, objects, boxes)
    assert instances == [{'id': 1, 'object': 'a', 'class': 'cube', 'pixels': 13, 'box_tight_xywh_px': [0, 0, 6, 6]}]
    assert mask_extent(mask == 1) == [0, 0, 6, 6]


def test_a_refused_first_frame_does_not_lock_its_run_and_two_runs_never_share_draws(ingest):
    bad = _frame(run_index=1)
    bad['tight_boxes'][0]['x_max'] += 3
    with pytest.raises(DataError):
        ingest.add_frame(**bad)
    other = _frame(run_index=1, draws=draws_for(7, 2))          # run 1 was never committed: other draws are fine
    ingest.add_frame(**other)
    with pytest.raises(DataError, match='same draws as run 1'):
        ingest.add_frame(**_frame(run_index=2))                  # run 2 with run 1's (= draws_for(7, 2)) draws
    ingest.close()


@pytest.mark.parametrize('mutate, needle', [
    (lambda reason, context: reason.update(run_seed=reason['run_seed'] + 1), 'run_seed'),
    (lambda reason, context: context.update(view=7), 'outside the declared'),
])
def test_verify_checks_each_frame_against_the_generation_block(store, mutate, needle):
    header = make_synthetic_session(store, runs=2, views=2)
    index = store.path(f'sessions/{header.session_id}/captures.jsonl')
    lines = [json.loads(x) for x in index.read_text().splitlines()]
    mutate(lines[1]['trigger']['reason'], lines[1]['trigger']['context']['trainit.randomization'])
    from trainit_data._canonical import canonical_json, sha256_bytes
    data = b''.join(canonical_json(x) + b'\n' for x in lines)
    os.chmod(index, 0o644)
    index.write_bytes(data)
    _edit_header(store, header.session_id, lambda d: d['index'].update(sha256=sha256_bytes(data)))
    report = verify_session(store, header.session_id)
    assert not report.ok and any(needle in f.detail for f in report.findings)
