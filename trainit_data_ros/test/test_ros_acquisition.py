"""D0-A and the D0.2 acceptance on a real ROS graph (private domain 87, in-process nodes).

MOCK CAMERA (trainit_hardware_layer, the registered-depth contract of 0.2.2) -> the camera contract ->
the recorder in slot ``r1`` -> a canonical session -> ``trainit-data verify`` PASS; MANUAL and INTERVAL;
the replay of that session re-recorded; the failure suite. No file is edited by hand.
"""
import json
import os
import subprocess
import sys
import time

import numpy as np
import pytest
import yaml

rclpy = pytest.importorskip('rclpy', reason='the acquisition tests need a sourced ROS 2 Humble')

from ros_graph import Graph, fake_camera, moving_tf, static_tf, wait_for  # noqa: E402
from trainit_data._canonical import sha256_bytes  # noqa: E402
import trainit_data_ros  # noqa: E402
from trainit_data.integrity import verify_session  # noqa: E402
from trainit_data.store import Store  # noqa: E402

SLOT = 'r1'
SERVICE = f'/{SLOT}/trainit_data_recorder'
FRAME = 'camera_color_optical_frame'


def recording(**overrides):
    doc = {'schema': 'trainit-recording/1.0', 'source': {'backend': 'mock'},
           'sensors': [{'sensor_id': 'overhead_camera', 'profile': 'realsense_d455',
                        'streams': {'color': {}, 'depth': {}}, 'extrinsic_parent_frame': 'base_link'}],
           'triggers': [{'id': 'manual', 'kind': 'MANUAL'}],
           'sync': {'policy': 'FIRST_AT_OR_AFTER_BOUNDARY', 'policy_version': 1, 'timeout_ms': 1000}}
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


def start_recorder(graph, tmp_path, store_root, doc, *, use_sim_time=False, name='rec'):
    from trainit_data_ros.recorder_node import Recorder
    path = tmp_path / f'{name}.recording.yaml'
    path.write_text(yaml.safe_dump(doc))
    node = Recorder(context=graph.ctx, namespace=SLOT, parameter_overrides=graph.params(
        {'config_file': str(path), 'store_root': str(store_root), 'use_sim_time': use_sim_time}))
    return graph.add(node)


def mock_camera(graph, rate_hz=30.0):
    from trainit_hardware_layer.camera.mock_camera_node import MockCamera
    return graph.add(MockCamera(context=graph.ctx, parameter_overrides=graph.params({'rate_hz': rate_hz})))


def session_of(store_root, response):
    assert response.success, response.message
    return Store.open(store_root).open_session(response.message)


def cli_verify(store_root, session_id):
    env = dict(os.environ)
    out = subprocess.run([sys.executable, '-m', 'trainit_data', '--root', str(store_root), '--json', 'verify',
                          session_id], capture_output=True, text=True, env=env)
    return out.returncode, json.loads(out.stdout)


def streams_flowing(rec):
    return wait_for(lambda: all(v > 3 for v in rec.coordinator.received.values()), 10.0)


# ---- D0-A ---------------------------------------------------------------------------------------------

def test_d0a_manual_mock_session_verifies(graph, tmp_path, store_root):
    mock_camera(graph)
    static_tf(graph)
    rec = start_recorder(graph, tmp_path, store_root, recording())
    client = graph.helper('test_client')
    started = graph.call(client, f'{SERVICE}/start_session')
    assert started.success, started.message
    assert streams_flowing(rec)
    answers = [graph.call(client, f'{SERVICE}/capture') for _ in range(3)]
    assert all(a.success and a.message.endswith('COMPLETE') for a in answers), [a.message for a in answers]
    closed = graph.call(client, f'{SERVICE}/close_session')
    assert closed.success, closed.message

    reader = Store.open(store_root).open_session(started.message)
    header = reader.header
    assert header.status == 'CLOSED' and header.clock.session_domain == 'ROS_TIME'
    assert header.source.backend == 'mock' and header.source.transport == 'LIVE_ROS'
    assert header.source.descriptor['intrinsics_kind'] == 'NOMINAL_PROFILE'
    assert header.recording_config['resolved'] == rec.cfg                       # the EXACT effective config
    assert header.recording_config['file_sha256'] == sha256_bytes((tmp_path / 'rec.recording.yaml').read_bytes())
    assert header.counters['captures'] == {'attempted': 3, 'complete': 3, 'partial': 0, 'failed': 0}
    assert all(v > 0 for v in header.counters['received'].values())
    assert header.counters['writer_overruns'] == 0 and header.counters['write_errors'] == 0
    assert header.provenance['software']['trainit_data_ros'] == trainit_data_ros.__version__
    captures = list(reader.captures())
    for i, capture in enumerate(captures):
        assert capture.trigger.kind == 'MANUAL' and capture.trigger.origin == 'OPERATOR'
        assert capture.trigger.firing_index == i
        assert capture.sync.status == 'SYNCED' and capture.sync.policy.name == 'FIRST_AT_OR_AFTER_BOUNDARY'
        assert [o.stream for o in capture.observations] == ['color', 'depth']
        for obs in capture.observations:
            assert obs.data.frame_id == FRAME and obs.stamp.clock_domain == 'ROS_TIME'
            assert obs.stamp.to_ns() >= capture.trigger.issued.stamp.to_ns()     # first AT OR AFTER
            assert obs.receipt.host_monotonic_ns > 0 and obs.payload.sha256.startswith('sha256:')
        depth = capture.observations[1]
        assert depth.data.encoding == '16UC1' and depth.data.unit_scale_m == 0.001
        assert int(reader.load_payload(depth)[0, 0]) == 1000                    # the mock's plane, in mm
        assert capture.observations[0].data.calibration_id in header.calibration_ids
    cal = Store.open(store_root).get_calibration(captures[0].observations[1].data.calibration_id, reader.vocab)
    assert cal.depth.registered_to == 'color' and cal.extrinsics[0].parent_frame == 'base_link'
    assert cal.extrinsics[0].provenance.kind == 'NOMINAL_DESCRIPTION'
    assert verify_session(Store.open(store_root), started.message).ok
    code, doc = cli_verify(store_root, started.message)
    assert code == 0 and doc['ok'] and doc['result']['findings'] == []


def test_d0a_interval_trigger_on_the_session_clock(graph, tmp_path, store_root):
    mock_camera(graph)
    static_tf(graph)
    rec = start_recorder(graph, tmp_path, store_root, recording(
        triggers=[{'id': 'every_200ms', 'kind': 'INTERVAL', 'period_ms': 200, 'max_captures': 5}]))
    client = graph.helper('test_client')
    started = graph.call(client, f'{SERVICE}/start_session')
    assert wait_for(lambda: rec.firings['every_200ms'] >= 5 and not rec.coordinator.pending, 10.0)
    time.sleep(0.5)
    assert rec.firings['every_200ms'] == 5                   # max_captures honoured
    refused = graph.call(client, f'{SERVICE}/capture')
    assert not refused.success and 'no MANUAL trigger' in refused.message
    closed = graph.call(client, f'{SERVICE}/close_session')
    assert closed.success, closed.message
    reader = session_of(store_root, started)
    captures = list(reader.captures())
    assert [c.trigger.firing_index for c in captures] == [0, 1, 2, 3, 4]
    assert {(c.trigger.kind, c.trigger.origin) for c in captures} == {('INTERVAL', 'TIMER')}
    assert all(c.outcome == 'COMPLETE' for c in captures)
    boundaries = [c.trigger.issued.stamp.to_ns() for c in captures]
    gaps = np.diff(boundaries) / 1e6
    assert all(150 < g < 300 for g in gaps), gaps            # the period, on the session clock
    assert verify_session(Store.open(store_root), started.message).ok


def test_the_recorder_is_slot_scoped(graph, tmp_path, store_root):
    rec = start_recorder(graph, tmp_path, store_root, recording())
    names = {f'{ns.rstrip("/")}/{n}' for n, ns in rec.get_node_names_and_namespaces()}
    assert '/r1/trainit_data_recorder' in names
    services = {name for name, _ in rec.get_service_names_and_types()}
    assert {f'{SERVICE}/start_session', f'{SERVICE}/capture', f'{SERVICE}/close_session'} <= services
    topics = {name for name, _ in rec.get_topic_names_and_types()}
    assert '/r1/diagnostics' in topics


def test_a_launch_node_name_remap_does_not_rename_the_tf_node(graph, tmp_path, store_root):
    """A launch's `name=` is the GLOBAL argument `-r __node:=...`: it must not give the recorder's TF node the
    recorder's name (seen live in D0.3: two nodes of one name, a rosout warning)."""
    import rclpy
    from ros_graph import DOMAIN_ID
    from trainit_data_ros.recorder_node import Recorder
    path = tmp_path / 'rec.recording.yaml'
    path.write_text(yaml.safe_dump(recording()))
    ctx = rclpy.Context()
    rclpy.init(context=ctx, domain_id=DOMAIN_ID, args=['--ros-args', '-r', '__node:=launched_recorder'])
    rec = None
    try:
        rec = Recorder(context=ctx, namespace=SLOT, parameter_overrides=graph.params(
            {'config_file': str(path), 'store_root': str(store_root)}))
        observer = graph.helper('test_observer')

        def names():
            return sorted(f'{ns.rstrip("/")}/{n}' for n, ns in observer.get_node_names_and_namespaces()
                          if n.startswith('launched_recorder'))
        assert wait_for(lambda: len(names()) == 2, 10.0), names()
        assert names() == ['/r1/launched_recorder', '/r1/launched_recorder_tf']
    finally:
        try:
            if rec is not None:
                rec.shutdown()
                rec.destroy_node()
        finally:
            rclpy.shutdown(context=ctx)          # never leak the second context into the shared domain


def test_diagnostics_report_the_counters(graph, tmp_path, store_root):
    from diagnostic_msgs.msg import DiagnosticArray
    mock_camera(graph)
    static_tf(graph)
    start_recorder(graph, tmp_path, store_root, recording())
    client = graph.helper('test_client')
    seen = []
    client.create_subscription(DiagnosticArray, f'/{SLOT}/diagnostics', seen.append, 10)
    graph.call(client, f'{SERVICE}/start_session')
    graph.call(client, f'{SERVICE}/capture')
    assert wait_for(lambda: any('captures_complete' in {v.key for v in m.status[0].values} and
                                {v.key: v.value for v in m.status[0].values}['captures_complete'] == '1'
                                for m in seen), 5.0)
    keys = {v.key for v in seen[-1].status[0].values}
    assert {'session_state', 'captures_attempted', 'captures_partial', 'captures_failed', 'queue_overruns',
            'write_failures', 'calibration_changes', 'sync_failures', 'last_skew_ns', 'queue_high_water',
            'received/overhead_camera/color', 'received/overhead_camera/depth'} <= keys
    graph.call(client, f'{SERVICE}/close_session')


# ---- replay -------------------------------------------------------------------------------------------

def test_a_replayed_session_is_recorded_again_with_replay_provenance(graph, tmp_path, store_root):
    from trainit_data_ros.replay_node import Replay
    mock_camera(graph)
    static_tf(graph)
    rec = start_recorder(graph, tmp_path, store_root, recording())
    client = graph.helper('test_client')
    original_id = graph.call(client, f'{SERVICE}/start_session').message
    assert streams_flowing(rec)
    for _ in range(3):
        assert graph.call(client, f'{SERVICE}/capture').success
    assert graph.call(client, f'{SERVICE}/close_session').success
    graph.reset()                                          # the live source, its TF and the recorder are gone
    client = graph.helper('test_client')
    graph.add(Replay(context=graph.ctx, parameter_overrides=graph.params(
        {'session_id': original_id, 'store_root': str(store_root), 'rate_hz': 10.0})))
    replay_rec = start_recorder(graph, tmp_path, store_root, recording(
        source={'backend': 'replay', 'replay_of': original_id}), name='replay')
    replay_id = graph.call(client, f'{SERVICE}/start_session').message
    assert streams_flowing(replay_rec)
    answers = [graph.call(client, f'{SERVICE}/capture') for _ in range(3)]
    assert all(a.success for a in answers), [a.message for a in answers]
    assert graph.call(client, f'{SERVICE}/close_session').success

    store = Store.open(store_root)
    original, replayed = store.open_session(original_id), store.open_session(replay_id)
    assert replayed.header.source.backend == 'replay' and replayed.header.source.replay_of == original_id
    assert 're-stamped' in replayed.header.source.descriptor['supplied_by']
    assert replayed.header.source.descriptor['intrinsics_kind'] == original.header.source.descriptor['intrinsics_kind']
    assert replayed.header.schema == original.header.schema
    originals = {o.stream: original.load_payload(o).tobytes() for c in original.captures() for o in c.observations}
    for capture in replayed.captures():
        assert capture.outcome == 'COMPLETE'
        for obs in capture.observations:
            assert replayed.load_payload(obs).tobytes() == originals[obs.stream]      # values bit-exact
    assert list(replayed.header.calibration_ids) == list(original.header.calibration_ids)  # same content, same id
    assert verify_session(store, replay_id).ok
    code, _ = cli_verify(store_root, replay_id)
    assert code == 0


def test_performance_baseline_at_the_reference_rate(graph, tmp_path, store_root):
    """Observability, not a benchmark: the mock at 30 Hz (848x480 rgb8 + 16UC1), an INTERVAL trigger at 10 Hz for
    8 s. Records the numbers (set TRAINIT_D02_PERF_OUT to keep them) and fails only on an architectural
    problem: an overrun or a write failure at this modest rate."""
    mock_camera(graph, rate_hz=30.0)
    static_tf(graph)
    rec = start_recorder(graph, tmp_path, store_root, recording(
        triggers=[{'id': 'every_100ms', 'kind': 'INTERVAL', 'period_ms': 100, 'max_captures': 80}]))
    client = graph.helper('test_client')
    started = graph.call(client, f'{SERVICE}/start_session')
    assert wait_for(lambda: rec.firings['every_100ms'] >= 80 and not rec.coordinator.pending, 20.0)
    closed = graph.call(client, f'{SERVICE}/close_session', timeout=60.0)
    assert closed.success, closed.message
    performance = json.loads(closed.message)['performance']
    if os.environ.get('TRAINIT_D02_PERF_OUT'):
        with open(os.environ['TRAINIT_D02_PERF_OUT'], 'w') as handle:
            json.dump(performance, handle, indent=2, sort_keys=True)
    writer = performance['writer']
    # Accounting must close whatever the host load: every firing is committed, dropped or lost -- and counted.
    assert sum(performance['outcomes'].values()) + writer['overruns'] + writer['lost'] == 80
    assert writer['write_failures'] == 0
    assert verify_session(Store.open(store_root), started.message).ok


def test_a_moving_camera_pose_is_looked_up_at_the_observation_stamp(graph, tmp_path, store_root):
    fake_camera(graph, rate_hz=15.0)
    moving_tf(graph)
    doc = recording(sync={'policy': 'FIRST_AT_OR_AFTER_BOUNDARY', 'policy_version': 1, 'timeout_ms': 1000})
    doc['sensors'][0].pop('extrinsic_parent_frame')
    doc['sensors'][0]['pose_parent_frame'] = 'world'
    rec = start_recorder(graph, tmp_path, store_root, doc)
    client = graph.helper('test_client')
    started = graph.call(client, f'{SERVICE}/start_session')
    assert streams_flowing(rec)
    time.sleep(0.5)                                       # the TF buffer holds a history
    for _ in range(3):
        assert graph.call(client, f'{SERVICE}/capture').message.split()[1] == 'COMPLETE'
        time.sleep(0.2)                                    # "latest" would now be ~200 ms later than the stamp
    assert graph.call(client, f'{SERVICE}/close_session').success
    reader = Store.open(store_root).open_session(started.message)
    for capture in reader.captures():
        for obs in capture.observations:
            expected = (obs.stamp.to_ns() % 1_000_000_000_000) / 1e9
            assert obs.data.pose_at_stamp.parent_frame == 'world'
            assert abs(obs.data.pose_at_stamp.translation[0] - expected) < 1e-6


def test_a_live_camera_info_change_is_a_new_calibration(graph, tmp_path, store_root):
    camera = fake_camera(graph, fx=lambda tick: 8.0 if tick < 25 else 8.5, rate_hz=20.0)
    rec = start_recorder(graph, tmp_path, store_root, recording(
        sensors=[{'sensor_id': 'overhead_camera', 'profile': 'realsense_d455', 'streams': {'color': {}, 'depth': {}}}]))
    client = graph.helper('test_client')
    started = graph.call(client, f'{SERVICE}/start_session')
    assert streams_flowing(rec)
    assert graph.call(client, f'{SERVICE}/capture').success
    assert wait_for(lambda: camera.ticks['tick'] > 30, 10.0)
    assert graph.call(client, f'{SERVICE}/capture').success
    closed = graph.call(client, f'{SERVICE}/close_session')
    assert closed.success
    reader = Store.open(store_root).open_session(started.message)
    first, second = list(reader.captures())
    assert first.observations[0].data.calibration_id != second.observations[0].data.calibration_id
    assert len(reader.header.calibration_ids) == 4            # colour + depth, before and after
    assert reader.header.counters['calibration_changes'] == 2  # per stream, never merged


def test_replay_keeps_values_order_and_intra_capture_offsets(graph, tmp_path, store_root):
    from trainit_data_ros.replay_node import Replay
    fake_camera(graph, vary=True, depth_lag_ms=3.0, rate_hz=20.0)
    doc = recording(sensors=[{'sensor_id': 'overhead_camera', 'profile': 'realsense_d455',
                              'streams': {'color': {}, 'depth': {}}}])
    rec = start_recorder(graph, tmp_path, store_root, doc)
    client = graph.helper('test_client')
    original_id = graph.call(client, f'{SERVICE}/start_session').message
    assert streams_flowing(rec)
    for _ in range(3):
        assert graph.call(client, f'{SERVICE}/capture').success
        time.sleep(0.15)
    assert graph.call(client, f'{SERVICE}/close_session').success
    graph.reset()
    client = graph.helper('test_client')

    replay = graph.add(Replay(context=graph.ctx, parameter_overrides=graph.params(
        {'session_id': original_id, 'store_root': str(store_root), 'rate_hz': 4.0})))
    replay_doc = recording(source={'backend': 'replay', 'replay_of': original_id},
                           sensors=doc['sensors'])
    replay_rec = start_recorder(graph, tmp_path, store_root, replay_doc, name='replay')
    replay_id = graph.call(client, f'{SERVICE}/start_session').message
    assert streams_flowing(replay_rec), (replay.published, replay.errors, dict(replay_rec.coordinator.received))
    for _ in range(6):
        assert graph.call(client, f'{SERVICE}/capture').success
        time.sleep(0.3)
    assert graph.call(client, f'{SERVICE}/close_session').success

    store = Store.open(store_root)
    original, replayed = store.open_session(original_id), store.open_session(replay_id)
    by_value = {}
    for capture in original.captures():
        colour, depth = capture.observations
        by_value[original.load_payload(colour).tobytes()] = (original.load_payload(depth).tobytes(),
                                                             depth.stamp.to_ns() - colour.stamp.to_ns())
    seen = set()
    for capture in replayed.captures():
        colour, depth = capture.observations
        key = replayed.load_payload(colour).tobytes()
        assert key in by_value                                    # a real original capture, bit-exact
        if capture.sync.status not in ('SYNCED', 'WITHIN_TOLERANCE'):
            # The boundary fell inside the 3 ms colour -> depth gap of one replayed capture: FIRST_AT_OR_AFTER then
            # takes colour from the next capture; the policy records it honestly as OUT_OF_TOLERANCE / PARTIAL.
            assert capture.outcome == 'PARTIAL'
            continue
        depth_bytes, offset = by_value[key]
        assert replayed.load_payload(depth).tobytes() == depth_bytes      # colour and depth of the SAME capture
        assert depth.stamp.to_ns() - colour.stamp.to_ns() == offset == 3_000_000
        seen.add(key)
    assert len(seen) >= 2                                         # not one capture replayed over and over
    assert verify_session(store, replay_id).ok
