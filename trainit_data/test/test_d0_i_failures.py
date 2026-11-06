"""D0-I -- the failure suite (D0_DESIGN.md §21.2): where each case is proven, and the end-to-end cases that were only
proven at the record level before D0.9.

Every failure is recorded honestly -- never dropped, never COMPLETE -- and ``verify`` finds what was damaged.

Where each case is proven (ROS = trainit_data_ros/test, a recorder on a real graph):
- missing CameraInfo: ROS test_ros_failures.py::test_contract_failures_are_recorded[missing_camera_info].
- missing calibration: test_integrity.py::test_a_missing_calibration_and_an_extra_file_are_found;
  test_crash_windows.py::test_verify_store_checks_calibration_files.
- bad calibration: test_calibration.py::test_an_uncalibrated_camera_matrix_is_refused (+ a stored one stays readable);
  ROS [uncalibrated_camera_info]; test_crash_windows.py::test_the_writer_refuses_a_calibration_of_another_size_or_frame.
- CLOCK_MISMATCH / CLOCK_UNAVAILABLE: ROS [clock_mismatch], the wall-clock camera under a sim clock, the stopped and
  the frozen sim clock.
- INCOMPATIBLE_CLOCKS: test_records.py (record); ROS test_coordinator.py (coordinator -> real writer); HERE (written,
  verified, re-read).
- missing / dropped depth: ROS [missing_depth]; HERE (depth that stops mid-session).
- sync timeout: ROS [missing_depth], test_no_camera_at_all_is_failed_within_the_timeout.
- interrupted capture + recover: ROS test_a_killed_recorder_leaves_a_recoverable_session (SIGKILL);
  test_integrity.py, test_crash_windows.py.
- disk full (ENOSPC): test_write_failures.py (4 cases); ROS test_a_write_failure_is_recorded_and_reported.
- corrupted payload: test_integrity.py::test_one_corrupted_byte_is_found_and_named; test_crash_windows.py.
- version overwrite refused: test_d0_6_versions.py::test_a_version_is_never_overwritten.

Every value is invented and customer-neutral.
"""
import numpy as np

from _helpers import FRAME, SENSOR, SENSORS, calibration, depth_mm, rgb, stamp, sync, trigger
from trainit_data.integrity import verify_session
from trainit_data.writer import ImageSample

OTHER = 'side_camera'


def test_incompatible_clocks_are_written_partial_verified_and_read_back(store, vocab):
    """A sensor stamped on another clock is never compared: its stream is recorded as missing with reason
    INCOMPATIBLE_CLOCKS (the recorder's representation, test_coordinator.py), the capture is PARTIAL with sync status
    INCOMPATIBLE_CLOCKS and no skew, and the stored record says so after a re-read."""
    sensors = [{'sensor_id': SENSOR, 'streams': [{'stream': 'color', 'modality': 'rgb'}]},
               {'sensor_id': OTHER, 'streams': [{'stream': 'color', 'modality': 'rgb'}]}]
    writer = store.create_session(source={'transport': 'LIVE_ROS', 'backend': 'mock'},
                                  clock={'session_domain': 'ROS_TIME'}, sensors=sensors)
    at = stamp(100)
    foreign = stamp(7, 0, 'SIM_TIME', sim_source='isaac')
    writer.append_capture(
        trigger=trigger(at), outcome='PARTIAL',
        sync=sync('INCOMPATIBLE_CLOCKS', None, reference=f'{SENSOR}/color'),
        samples=[ImageSample(SENSOR, 'color', at, rgb(1), 'rgb8', FRAME)],
        missing=[{'stream': f'{OTHER}/color', 'reason': 'INCOMPATIBLE_CLOCKS',
                  'detail': f'stamped {foreign.sec} s on SIM_TIME (isaac); the session clock is ROS_TIME'}])
    header = writer.close()
    report = verify_session(store, header.session_id)
    assert report.ok, report.to_dict()
    [capture] = store.open_session(header.session_id).captures()
    assert capture.outcome == 'PARTIAL' and capture.sync.status == 'INCOMPATIBLE_CLOCKS'
    assert capture.sync.max_skew_ns is None                     # never a skew between two clocks
    assert [o.stream_key for o in capture.observations] == [f'{SENSOR}/color']
    assert [(m.stream, m.reason) for m in capture.missing] == [(f'{OTHER}/color', 'INCOMPATIBLE_CLOCKS')]
    assert header.counters['captures']['partial'] == 1


def test_depth_that_stops_mid_session_is_partial_from_then_on(store, vocab):
    """Colour keeps arriving, depth stops after the first capture: the later captures are PARTIAL with the reason,
    the counters say so, and the session verifies."""
    color_cal = store.put_calibration(calibration(vocab, 'color'))
    depth_cal = store.put_calibration(calibration(vocab, 'depth'))
    writer = store.create_session(source={'transport': 'LIVE_ROS', 'backend': 'mock'},
                                  clock={'session_domain': 'ROS_TIME'}, sensors=SENSORS)
    for i in range(3):
        at = stamp(100 + i)
        color = ImageSample(SENSOR, 'color', at, rgb(i), 'rgb8', FRAME, calibration_id=color_cal)
        if i == 0:
            depth = ImageSample(SENSOR, 'depth', at, depth_mm(i), '16UC1', FRAME, unit_scale_m=0.001,
                                calibration_id=depth_cal)
            writer.append_capture(trigger=trigger(at, i), outcome='COMPLETE', sync=sync(), samples=[color, depth])
        else:
            writer.append_capture(trigger=trigger(at, i), outcome='PARTIAL', sync=sync('SINGLE_STREAM', None),
                                  samples=[color], missing=[{'stream': f'{SENSOR}/depth', 'reason': 'TIMEOUT',
                                                             'detail': 'no depth frame at or after the boundary'}])
    header = writer.close(received={f'{SENSOR}/color': 90, f'{SENSOR}/depth': 30})
    assert header.counters['captures'] == {'attempted': 3, 'complete': 1, 'partial': 2, 'failed': 0}
    report = verify_session(store, header.session_id)
    assert report.ok, report.to_dict()
    captures = list(store.open_session(header.session_id).captures())
    assert [c.outcome for c in captures] == ['COMPLETE', 'PARTIAL', 'PARTIAL']
    assert all(c.missing[0].reason == 'TIMEOUT' and c.missing[0].stream == f'{SENSOR}/depth' for c in captures[1:])
    assert np.array_equal(store.open_session(header.session_id).load_payload(captures[2].observations[0]), rgb(2))
