"""D0.3: an Isaac-shaped live source through the SAME recorder (no Isaac-specific code path).

The source reproduces the reference cell's documented graph (one render product; rgb8 + 32FC1 depth in the
colour optical frame on the same tick; ONLY the colour CameraInfo; sim time on /clock). The recording
configuration is the cell's choice A: the depth stream's CameraInfo is the colour one (registered depth). The
live Isaac run itself is the owner-guided D0-B acceptance; these tests pin the behaviour without a GPU.
"""
import time

import numpy as np
import pytest
import yaml

rclpy = pytest.importorskip('rclpy', reason='the acquisition tests need a sourced ROS 2 Humble')

from ros_graph import Graph, SimCamera, static_tf, wait_for  # noqa: E402
from test_ros_acquisition import SERVICE, SLOT, cli_verify  # noqa: E402
from trainit_data.integrity import verify_session  # noqa: E402
from trainit_data.store import Store  # noqa: E402

FRAME = 'camera_color_optical_frame'
# Neutral, installed (apt) artifacts stand in for a cell's description and stage.
DESCRIPTION = {'package': 'realsense2_description', 'file': 'urdf/test_d455_camera.urdf.xacro',
               'args': {'use_nominal_extrinsics': 'true'}}
SCENE = {'package': 'realsense2_description', 'file': 'package.xml', 'prim': '/World/Camera'}


def isaac_recording(**overrides):
    doc = {'schema': 'trainit-recording/1.0', 'source': {'backend': 'isaac'},
           'sensors': [{'sensor_id': 'cell_camera', 'profile': 'realsense_d455',
                        'streams': {'color': {}, 'depth': {'info': '/camera/color/camera_info'}},   # choice A
                        'extrinsic_parent_frame': 'base_link'}],
           'triggers': [{'id': 'manual', 'kind': 'MANUAL'}],
           'sync': {'policy': 'FIRST_AT_OR_AFTER_BOUNDARY', 'policy_version': 1, 'timeout_ms': 1000},
           'provenance': {'description': DESCRIPTION, 'scene': SCENE}}
    doc.update(overrides)
    return doc


@pytest.fixture
def graph():
    g = Graph()
    yield g
    g.close()


@pytest.fixture
def store_root(tmp_path):
    root = tmp_path / 'TrainItData'
    Store.open(root, create=True)
    return root


def start(graph, tmp_path, store_root, doc, name='isaac'):
    from trainit_data_ros.recorder_node import Recorder
    path = tmp_path / f'{name}.recording.yaml'
    path.write_text(yaml.safe_dump(doc))
    return graph.add(Recorder(context=graph.ctx, namespace=SLOT, parameter_overrides=graph.params(
        {'config_file': str(path), 'store_root': str(store_root), 'use_sim_time': True})))


def flowing(rec):
    return wait_for(lambda: all(v > 3 for v in rec.coordinator.received.values())
                    and rec.get_clock().now().nanoseconds > 0, 10.0)


def session(graph, client, store_root, captures=3, gap=0.1):
    started = graph.call(client, f'{SERVICE}/start_session')
    assert started.success, started.message
    answers = []
    for _ in range(captures):
        answers.append(graph.call(client, f'{SERVICE}/capture'))
        time.sleep(gap)
    closed = graph.call(client, f'{SERVICE}/close_session')
    assert closed.success, closed.message
    return Store.open(store_root).open_session(started.message), answers


def test_d0b_shape_isaac_rgbd_is_a_canonical_sim_time_session(graph, tmp_path, store_root):
    camera = SimCamera(graph)
    static_tf(graph)
    rec = start(graph, tmp_path, store_root, isaac_recording())
    assert flowing(rec)
    reader, answers = session(graph, graph.helper('test_client'), store_root)
    assert all(a.success and a.message.split()[1] == 'COMPLETE' for a in answers), [a.message for a in answers]
    header = reader.header
    assert header.source.backend == 'isaac'
    assert header.source.descriptor == {'intrinsics_kind': 'SYNTHETIC_RENDER', 'stamp_origin': 'HEADER',
                                        'registration_basis': 'same_render_product', 'supplied_by':
                                        'trainit_hardware_layer profile realsense_d455 backends.isaac'}
    assert (header.clock.session_domain, header.clock.sim_source) == ('SIM_TIME', 'isaac')
    assert header.recording_config['resolved'] == rec.cfg
    assert rec.cfg['sensors'][0]['streams']['depth']['info'] == '/camera/color/camera_info'
    description = header.provenance['robot']['description']
    from trainit_data_ros.provenance import description_provenance
    assert description == description_provenance(DESCRIPTION)          # recomputed independently
    assert len(description['files']) >= 2                               # the include closure, not just the root
    assert header.provenance['scene']['sim_stage']['prim'] == '/World/Camera'
    store = Store.open(store_root)
    for capture in reader.captures():
        assert capture.outcome == 'COMPLETE' and capture.sync.status == 'SYNCED'       # one render tick
        colour, depth = capture.observations
        assert colour.stamp.to_ns() == depth.stamp.to_ns() >= capture.trigger.issued.stamp.to_ns()
        assert (colour.stamp.clock_domain, colour.stamp.sim_source) == ('SIM_TIME', 'isaac')
        assert colour.data.frame_id == depth.data.frame_id == FRAME
        assert (depth.data.encoding, depth.data.unit_scale_m) == ('32FC1', 1.0)
        array = reader.load_payload(depth)
        assert array.dtype == np.float32 and np.isnan(array[0, 0]) and array[0, 1] == np.inf and array[0, 2] == -np.inf
        n = int(reader.load_payload(colour)[0, 0, 0])
        expected = camera.depth_array(n)
        assert array.tobytes() == expected.tobytes()                                    # metres, bit-exact
        for obs in capture.observations:
            cal = store.get_calibration(obs.data.calibration_id, reader.vocab)
            assert cal.intrinsics_provenance.kind == 'SYNTHETIC_RENDER'
            assert cal.intrinsics_provenance.ref == {'source': 'sensor_msgs/CameraInfo'}   # no topic in the identity
            assert 'pixel-edge' in cal.intrinsics_provenance.notes                        # the profile's isaac note
            assert cal.optical_frame == FRAME
            i = cal.intrinsics
            assert (i.width, i.height, i.distortion_model) == (16, 12, 'plumb_bob') and i.D == (0.0,) * 5
            assert i.K == (8.0, 0.0, 8.0, 0.0, 8.0, 6.0, 0.0, 0.0, 1.0)
            assert i.P == (8.0, 0.0, 8.0, 0.0, 0.0, 8.0, 6.0, 0.0, 0.0, 0.0, 1.0, 0.0)
            assert cal.extrinsics[0].parent_frame == 'base_link'
            assert cal.extrinsics[0].provenance.kind == 'NOMINAL_DESCRIPTION'
            assert cal.extrinsics[0].provenance.ref == {'description_sha256': description['sha256']}
            assert cal.transform_tree['description_sha256'] == description['sha256']
        depth_cal = store.get_calibration(depth.data.calibration_id, reader.vocab)
        assert depth_cal.depth.registered_to == 'color'
        assert depth_cal.depth.registration_basis == 'same_render_product'
        assert depth_cal.depth.invalid == 'nan_or_inf'
    assert verify_session(store, reader.session_id).ok
    code, _ = cli_verify(store_root, reader.session_id)
    assert code == 0


def test_interval_on_sim_time_pauses_with_the_simulation(graph, tmp_path, store_root):
    camera = SimCamera(graph)
    static_tf(graph)
    rec = start(graph, tmp_path, store_root, isaac_recording(
        triggers=[{'id': 'every_200ms', 'kind': 'INTERVAL', 'period_ms': 200, 'max_captures': 8},
                  {'id': 'manual', 'kind': 'MANUAL'}]))
    assert flowing(rec)
    client = graph.helper('test_client')
    started = graph.call(client, f'{SERVICE}/start_session')
    assert wait_for(lambda: rec.firings['every_200ms'] >= 3, 10.0)
    camera.pause()                                         # no ticks: no /clock, no frames
    time.sleep(0.3)
    paused_at = rec.firings['every_200ms']
    time.sleep(1.0)                                        # a wall second of pause
    assert rec.firings['every_200ms'] == paused_at          # INTERVAL is on the session clock: it paused too
    camera.resume()
    assert wait_for(lambda: rec.firings['every_200ms'] == 8, 10.0)
    after_resume = graph.call(client, f'{SERVICE}/capture')      # MANUAL works again after the pause
    time.sleep(0.8)
    assert rec.firings['every_200ms'] == 8                      # max_captures stops the trigger
    assert graph.call(client, f'{SERVICE}/close_session').success
    assert after_resume.message.split()[1] == 'COMPLETE'
    reader = Store.open(store_root).open_session(started.message)
    interval = [c for c in reader.captures() if c.trigger.kind == 'INTERVAL']
    gaps = np.diff([c.trigger.issued.stamp.to_ns() for c in interval])
    assert len(interval) == 8 and all(gap == 200_000_000 for gap in gaps)   # the exact sim-time grid, pause included
    assert all(c.outcome == 'COMPLETE' for c in reader.captures())
    assert verify_session(Store.open(store_root), started.message).ok


def test_a_capture_while_paused_is_bounded_by_the_host_clock(graph, tmp_path, store_root):
    camera = SimCamera(graph)
    static_tf(graph)
    rec = start(graph, tmp_path, store_root, isaac_recording(
        sync={'policy': 'FIRST_AT_OR_AFTER_BOUNDARY', 'policy_version': 1, 'timeout_ms': 400}))
    assert flowing(rec)
    client = graph.helper('test_client')
    started = graph.call(client, f'{SERVICE}/start_session')
    camera.pause()
    time.sleep(0.2)
    t0 = time.monotonic()
    answer = graph.call(client, f'{SERVICE}/capture')       # the boundary is the frozen sim time
    elapsed = time.monotonic() - t0
    assert graph.call(client, f'{SERVICE}/close_session').success
    assert elapsed < 3.0                                      # never an unbounded wait
    [capture] = list(Store.open(store_root).open_session(started.message).captures())
    # The last frame of the paused scene carries exactly the frozen time: it is AT the boundary, a true capture of
    # the paused state. (Real Isaac may stamp its last frame BEFORE the last /clock: then TIMEOUT -- still bounded.)
    assert answer.success and capture.outcome == 'COMPLETE'
    assert all(o.stamp.to_ns() == capture.trigger.issued.stamp.to_ns() for o in capture.observations)


def test_a_simulation_reset_is_a_recorded_clock_jump_never_a_mixed_capture(graph, tmp_path, store_root):
    camera = SimCamera(graph, start_s=20.0)
    static_tf(graph)
    rec = start(graph, tmp_path, store_root, isaac_recording())
    assert flowing(rec)
    client = graph.helper('test_client')
    started = graph.call(client, f'{SERVICE}/start_session')
    assert graph.call(client, f'{SERVICE}/capture').message.split()[1] == 'COMPLETE'
    camera.reset(0.5)                                       # stop / play: sim time starts again near 0
    assert wait_for(lambda: rec.clock_jumps == 1, 5.0)
    time.sleep(0.3)
    after = graph.call(client, f'{SERVICE}/capture')
    closed = graph.call(client, f'{SERVICE}/close_session')
    assert closed.success
    reader = Store.open(store_root).open_session(started.message)
    first, second = list(reader.captures())
    assert first.trigger.issued.stamp.sec >= 20 and second.trigger.issued.stamp.sec < 5
    assert after.message.split()[1] == 'COMPLETE'
    for obs in second.observations:                          # only the NEW timeline
        assert obs.stamp.to_ns() >= second.trigger.issued.stamp.to_ns() and obs.stamp.sec < 5
    codes = [w['code'] for w in reader.header.warnings]
    assert 'CLOCK_JUMP_BACKWARDS' in codes
    assert verify_session(Store.open(store_root), started.message).ok


@pytest.mark.parametrize('camera_kw, outcome, reasons', [
    (dict(color_info=False), 'FAILED', {'color': 'NO_CAMERA_INFO', 'depth': 'NO_CAMERA_INFO'}),   # choice A
    (dict(depth_frame='camera_depth_optical_frame'), 'PARTIAL', {'depth': 'FRAME_MISMATCH'}),
    (dict(depth_encoding='8UC1'), 'PARTIAL', {'depth': 'ENCODING_REFUSED'}),
    (dict(depth=False), 'PARTIAL', {'depth': 'TIMEOUT'}),
], ids=['missing_camera_info', 'wrong_frame', 'wrong_depth_encoding', 'missing_depth'])
def test_isaac_shaped_failures_are_never_complete(graph, tmp_path, store_root, camera_kw, outcome, reasons):
    SimCamera(graph, **camera_kw)
    static_tf(graph)
    rec = start(graph, tmp_path, store_root, isaac_recording(
        sync={'policy': 'FIRST_AT_OR_AFTER_BOUNDARY', 'policy_version': 1, 'timeout_ms': 400}))
    assert wait_for(lambda: rec.coordinator.received['cell_camera/color'] > 3, 10.0)
    reader, answers = session(graph, graph.helper('test_client'), store_root, captures=1)
    [capture] = list(reader.captures())
    assert capture.outcome == outcome
    assert {m.stream.split('/')[1]: m.reason for m in capture.missing} == reasons


def test_the_isaac_session_replays_bit_exact_with_its_provenance(graph, tmp_path, store_root):
    from trainit_data_ros.replay_node import Replay
    SimCamera(graph)
    static_tf(graph)
    rec = start(graph, tmp_path, store_root, isaac_recording())
    assert flowing(rec)
    client = graph.helper('test_client')
    original, _ = session(graph, client, store_root, captures=3, gap=0.15)
    graph.reset()                                          # the simulation, its TF and the recorder are gone
    client = graph.helper('test_client')

    replay = graph.add(Replay(context=graph.ctx, parameter_overrides=graph.params(
        {'session_id': original.session_id, 'store_root': str(store_root), 'rate_hz': 4.0})))
    from trainit_data_ros.recorder_node import Recorder
    path = tmp_path / 'replay.recording.yaml'
    replay_doc = isaac_recording(source={'backend': 'replay', 'replay_of': original.session_id})
    replay_doc.pop('provenance')                             # a replay inherits the description provenance
    path.write_text(yaml.safe_dump(replay_doc))
    replay_rec = graph.add(Recorder(context=graph.ctx, namespace=SLOT, parameter_overrides=graph.params(
        {'config_file': str(path), 'store_root': str(store_root)})))
    assert wait_for(lambda: all(v > 3 for v in replay_rec.coordinator.received.values()), 10.0), \
        (replay.published, replay.errors, dict(replay_rec.coordinator.received))
    replayed, answers = session(graph, client, store_root, captures=3, gap=0.3)
    assert all(a.success for a in answers), [a.message for a in answers]
    assert all(c.outcome == 'COMPLETE' and c.sync.status == 'SYNCED' for c in replayed.captures())

    assert replayed.header.source.backend == 'replay' and replayed.header.source.replay_of == original.session_id
    assert replayed.header.source.descriptor['intrinsics_kind'] == 'SYNTHETIC_RENDER'
    assert replayed.header.provenance['robot'] == original.header.provenance['robot']
    assert list(replayed.header.calibration_ids) == list(original.header.calibration_ids)   # same content, same id
    originals = {}
    for capture in original.captures():
        colour, depth = capture.observations
        originals[original.load_payload(colour).tobytes()] = original.load_payload(depth).tobytes()
    for capture in replayed.captures():
        colour, depth = capture.observations
        key = replayed.load_payload(colour).tobytes()
        assert replayed.load_payload(depth).tobytes() == originals[key]                       # NaN / inf bit-exact
    assert verify_session(Store.open(store_root), replayed.session_id).ok


def test_a_reset_during_an_interval_trigger_re_anchors_on_the_new_timeline(graph, tmp_path, store_root):
    camera = SimCamera(graph, start_s=20.0)
    static_tf(graph)
    rec = start(graph, tmp_path, store_root, isaac_recording(
        triggers=[{'id': 'every_200ms', 'kind': 'INTERVAL', 'period_ms': 200}]))
    assert flowing(rec)
    client = graph.helper('test_client')
    started = graph.call(client, f'{SERVICE}/start_session')
    assert wait_for(lambda: rec.firings['every_200ms'] >= 3, 10.0)
    camera.reset(0.5)
    assert wait_for(lambda: rec.clock_jumps == 1, 5.0)
    fired = rec.firings['every_200ms']
    assert wait_for(lambda: rec.firings['every_200ms'] >= fired + 3, 10.0)
    time.sleep(0.3)
    assert graph.call(client, f'{SERVICE}/close_session').success
    reader = Store.open(store_root).open_session(started.message)
    for capture in reader.captures():
        boundary = capture.trigger.issued.stamp.to_ns()
        if capture.outcome == 'COMPLETE':
            # never a frame of the other timeline: every observation is at most one frame after its boundary
            assert all(0 <= o.stamp.to_ns() - boundary < 100_000_000 for o in capture.observations)
    new = [c for c in reader.captures() if c.trigger.issued.stamp.sec < 5]
    assert len(new) >= 3 and all(c.outcome == 'COMPLETE' for c in new)
    assert 'CLOCK_JUMP_BACKWARDS' in [w['code'] for w in reader.header.warnings]
