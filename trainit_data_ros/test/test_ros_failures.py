"""The D0.2 failure suite on a real ROS graph: every failure is recorded honestly, none becomes COMPLETE.

INCOMPATIBLE_CLOCKS cannot arise from ROS headers (a header stamp carries no clock domain: every received
stamp is read on the session clock); a publisher on another clock shows up as CLOCK_MISMATCH here, and
the policy's refusal to compare domains is proven in test_coordinator.py.
"""
import errno
import os
import signal
import subprocess
import sys
import time

import pytest
import yaml

rclpy = pytest.importorskip('rclpy', reason='the acquisition tests need a sourced ROS 2 Humble')

from ros_graph import DOMAIN_ID, Graph, fake_camera, sim_clock, static_tf, wait_for  # noqa: E402
from test_ros_acquisition import SERVICE, SLOT, mock_camera, recording, start_recorder  # noqa: E402
from trainit_data.integrity import recover_session, verify_session  # noqa: E402
from trainit_data.store import Store  # noqa: E402


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


def no_tf(**overrides):
    doc = recording(**overrides)
    doc['sensors'][0].pop('extrinsic_parent_frame')
    return doc


def one_capture(graph, store_root, rec, *, settle=True):
    client = graph.helper('test_client')
    started = graph.call(client, f'{SERVICE}/start_session')
    assert started.success, started.message
    if settle:
        time.sleep(0.4)
    answer = graph.call(client, f'{SERVICE}/capture')
    closed = graph.call(client, f'{SERVICE}/close_session')
    assert closed.success, closed.message
    reader = Store.open(store_root).open_session(started.message)
    [capture] = list(reader.captures())
    assert verify_session(Store.open(store_root), started.message).ok
    return answer, capture, reader


def reasons(capture):
    return {m.stream.split('/')[1]: m.reason for m in capture.missing}


@pytest.mark.parametrize('camera, outcome, expected', [
    (dict(infos=False), 'FAILED', {'color': 'NO_CAMERA_INFO', 'depth': 'NO_CAMERA_INFO'}),
    (dict(depth_frame='camera_depth_optical_frame'), 'PARTIAL', {'depth': 'FRAME_MISMATCH'}),
    (dict(stamp_offset_s=5.0), 'FAILED', {'color': 'CLOCK_MISMATCH', 'depth': 'CLOCK_MISMATCH'}),
    (dict(depth=False), 'PARTIAL', {'depth': 'TIMEOUT'}),                     # missing depth = sync timeout
    (dict(fx=lambda tick: 0.0), 'FAILED',                                     # REP-104 uncalibrated CameraInfo
     {'color': 'CALIBRATION_MISMATCH', 'depth': 'CALIBRATION_MISMATCH'}),
], ids=['missing_camera_info', 'frame_mismatch', 'clock_mismatch', 'missing_depth', 'uncalibrated_camera_info'])
def test_contract_failures_are_recorded(graph, tmp_path, store_root, camera, outcome, expected):
    fake_camera(graph, **camera)
    rec = start_recorder(graph, tmp_path, store_root, no_tf(
        sync={'policy': 'FIRST_AT_OR_AFTER_BOUNDARY', 'policy_version': 1, 'timeout_ms': 400}))
    answer, capture, _ = one_capture(graph, store_root, rec)
    assert capture.outcome == outcome and reasons(capture) == expected
    assert answer.success and answer.message.split()[1] == outcome          # written, and says so
    if outcome == 'PARTIAL':
        assert capture.sync.status == 'SINGLE_STREAM'


def test_no_camera_at_all_is_failed_within_the_timeout(graph, tmp_path, store_root):
    rec = start_recorder(graph, tmp_path, store_root, no_tf(
        sync={'policy': 'FIRST_AT_OR_AFTER_BOUNDARY', 'policy_version': 1, 'timeout_ms': 300}))
    answer, capture, _ = one_capture(graph, store_root, rec, settle=False)
    assert capture.outcome == 'FAILED' and set(reasons(capture).values()) == {'TIMEOUT'} and capture.sync is None
    assert answer.success


def test_a_stopped_sim_clock_is_clock_unavailable(graph, tmp_path, store_root):
    fake_camera(graph)
    rec = start_recorder(graph, tmp_path, store_root, no_tf(clock={'sim_source': 'isaac'}), use_sim_time=True)
    _, capture, reader = one_capture(graph, store_root, rec)              # no /clock: the session clock reads 0
    assert capture.outcome == 'FAILED' and set(reasons(capture).values()) == {'CLOCK_UNAVAILABLE'}
    assert reader.header.clock.session_domain == 'SIM_TIME' and reader.header.clock.sim_source == 'isaac'


def test_a_frozen_sim_clock_never_hangs_a_capture(graph, tmp_path, store_root):
    sim_clock(graph, 10.0)
    fake_camera(graph, use_sim_time=True, stamp_offset_s=-1.0)          # samples before the boundary only
    rec = start_recorder(graph, tmp_path, store_root, no_tf(
        clock={'sim_source': 'isaac'}, sync={'policy': 'FIRST_AT_OR_AFTER_BOUNDARY', 'policy_version': 1,
                                             'timeout_ms': 300}), use_sim_time=True)
    assert wait_for(lambda: rec.get_clock().now().nanoseconds > 0, 5.0)
    t0 = time.monotonic()
    answer, capture, _ = one_capture(graph, store_root, rec)
    assert capture.outcome == 'FAILED' and set(reasons(capture).values()) == {'TIMEOUT'}
    assert capture.trigger.issued.stamp.sec == 10                         # the boundary is on the sim clock
    assert time.monotonic() - t0 < 5.0                                     # bounded by the HOST clock


def test_a_wall_clock_camera_under_a_sim_clock_session_is_never_synced(graph, tmp_path, store_root):
    sim_clock(graph, 10.0)
    fake_camera(graph)                                                    # wall-clock stamps
    rec = start_recorder(graph, tmp_path, store_root, no_tf(clock={'sim_source': 'isaac'}), use_sim_time=True)
    assert wait_for(lambda: rec.get_clock().now().nanoseconds > 0, 5.0)
    _, capture, _ = one_capture(graph, store_root, rec)
    assert capture.outcome == 'FAILED' and set(reasons(capture).values()) == {'CLOCK_MISMATCH'}


def test_writer_queue_overrun_is_counted_and_never_claimed(graph, tmp_path, store_root, monkeypatch):
    from trainit_data import writer as core_writer
    real = core_writer.SessionWriter.append_capture

    def slow(self, **kwargs):
        time.sleep(0.25)
        return real(self, **kwargs)
    monkeypatch.setattr(core_writer.SessionWriter, 'append_capture', slow)
    mock_camera(graph)
    static_tf(graph)
    rec = start_recorder(graph, tmp_path, store_root, recording(
        triggers=[{'id': 'fast', 'kind': 'INTERVAL', 'period_ms': 40, 'max_captures': 30},
                  {'id': 'manual', 'kind': 'MANUAL'}],
        writer={'queue_size': 1, 'max_pending': 16}))
    client = graph.helper('test_client')
    started = graph.call(client, f'{SERVICE}/start_session')
    assert wait_for(lambda: rec.firings['fast'] >= 30, 10.0)
    assert wait_for(lambda: rec.worker.overruns > 0, 5.0)
    answers = [graph.call(client, f'{SERVICE}/capture') for _ in range(4)]
    closed = graph.call(client, f'{SERVICE}/close_session', timeout=60.0)
    assert closed.success, closed.message
    header = Store.open(store_root).open_session(started.message).header
    lines = header.index['lines']
    assert header.counters['writer_overruns'] > 0
    assert lines + header.counters['writer_overruns'] == 30 + 4             # every firing: written or counted
    assert all(a.success == ('DROPPED' not in a.message) for a in answers)  # a dropped capture never succeeds
    assert verify_session(Store.open(store_root), started.message).ok


def test_a_write_failure_is_recorded_and_reported(graph, tmp_path, store_root, monkeypatch):
    from trainit_data import _fsio
    real = _fsio.write_fully

    def enospc(fd, data):
        if os.readlink(f'/proc/self/fd/{fd}').endswith('overhead_camera__depth.png.tmp'):
            real(fd, data[:len(data) // 2])
            raise OSError(errno.ENOSPC, 'No space left on device')
        real(fd, data)
    mock_camera(graph)
    static_tf(graph)
    rec = start_recorder(graph, tmp_path, store_root, recording())
    monkeypatch.setattr(_fsio, 'write_fully', enospc)
    answer, capture, reader = one_capture(graph, store_root, rec)
    assert not answer.success and answer.message.startswith('WRITE_FAILED')
    assert capture.outcome == 'PARTIAL' and reasons(capture) == {'depth': 'WRITE_FAILED'}
    assert reader.header.counters['write_errors'] == 1
    assert reader.header.warnings[-1]['code'] == 'WRITE_FAILED'          # the partial .tmp moved to orphans/


def test_a_killed_recorder_leaves_a_recoverable_session(graph, tmp_path, store_root):
    mock_camera(graph)
    static_tf(graph)
    config = tmp_path / 'kill.recording.yaml'
    config.write_text(yaml.safe_dump(recording()))
    env = dict(os.environ, ROS_DOMAIN_ID=str(DOMAIN_ID), PYTHONUNBUFFERED='1')
    proc = subprocess.Popen([sys.executable, '-m', 'trainit_data_ros.recorder_node', '--ros-args',
                             '-r', f'__ns:=/{SLOT}', '-p', f'config_file:={config}', '-p', f'store_root:={store_root}'],
                            env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    try:
        client = graph.helper('test_client')
        started = graph.call(client, f'{SERVICE}/start_session')
        assert started.success, started.message
        time.sleep(1.0)
        for _ in range(2):
            assert graph.call(client, f'{SERVICE}/capture').success      # written (whatever its outcome)
    finally:
        proc.send_signal(signal.SIGKILL)
        proc.wait(timeout=10)
    store = Store.open(store_root)
    assert store.open_session(started.message).header.status == 'ACTIVE'
    report = recover_session(store, started.message)
    assert report['committed_captures'] == 2
    header = store.open_session(started.message).header
    counts = header.counters['captures']
    assert header.status == 'INTERRUPTED' and counts['complete'] + counts['partial'] + counts['failed'] == 2
    assert verify_session(store, started.message).ok
