"""CalibrationRecord identity (acceptance 8): same canonical content -> same id; any change -> another id."""
import copy
import json

import pytest

from _helpers import calibration, calibration_content
from trainit_data.calibration import CalibrationRecord
from trainit_data.errors import CALIBRATION_CONFLICT, CALIBRATION_NOT_FOUND, HASH_MISMATCH, INVALID_VALUE, DataError


def _id(vocab, content):
    return CalibrationRecord.build(content, vocab).calibration_id


def test_same_content_same_id(vocab):
    a = calibration(vocab)
    b = calibration(vocab)
    assert a.calibration_id == b.calibration_id and a.sha256 == b.sha256
    assert a.calibration_id == 'cal-' + a.sha256[7:23]


def test_non_semantic_differences_do_not_change_the_id(vocab):
    content = calibration_content()
    as_floats = copy.deepcopy(content)
    as_floats['intrinsics']['K'] = [float(v) for v in content['intrinsics']['K']]
    as_floats['intrinsics']['D'] = [-0.0, 0.0, 0.0, 0.0, 0.0]                   # -0.0 is 0.0
    reordered = dict(reversed(list(copy.deepcopy(content).items())))       # key order is not content
    assert _id(vocab, content) == _id(vocab, as_floats) == _id(vocab, reordered)


@pytest.mark.parametrize('mutate', [
    lambda c: c['intrinsics']['K'].__setitem__(0, 4.0001),
    lambda c: c['intrinsics']['D'].__setitem__(4, 1e-9),
    lambda c: c['intrinsics'].__setitem__('distortion_model', 'rational_polynomial'),
    lambda c: c.__setitem__('optical_frame', 'other_optical_frame'),
    lambda c: c.__setitem__('stream', 'depth'),
    lambda c: c['intrinsics_provenance'].__setitem__('kind', 'DEVICE_REPORTED'),
    lambda c: c['intrinsics_provenance'].__setitem__('notes', 'other'),
    lambda c: c['extrinsics'][0]['translation'].__setitem__(2, 0.5000001),
    lambda c: c['extrinsics'][0]['provenance'].__setitem__('kind', 'SYNTHETIC_SCENE'),
    lambda c: c.__setitem__('depth', {'invalid': 'nan_or_inf'}),
    lambda c: c.__setitem__('extensions', {'trainit.test.note': 1}),
    lambda c: c.__setitem__('schema', 'trainit-calibration/1.1'),
])
def test_any_content_change_changes_the_id(vocab, mutate):
    content = calibration_content()
    changed = copy.deepcopy(content)
    mutate(changed)
    assert _id(vocab, changed) != _id(vocab, content)


def test_a_stored_record_must_hash_to_its_id(vocab):
    stored = calibration(vocab).to_dict()
    assert CalibrationRecord.from_dict(copy.deepcopy(stored), vocab).calibration_id == stored['calibration_id']
    stored['intrinsics']['K'][0] = 5.0                      # edited after the fact
    with pytest.raises(DataError) as err:
        CalibrationRecord.from_dict(stored, vocab)
    assert err.value.code == HASH_MISMATCH


def test_build_takes_content_only_and_validates(vocab):
    with pytest.raises(DataError):
        CalibrationRecord.build(dict(calibration_content(), calibration_id='cal-0000000000000000'), vocab)
    bad = calibration_content()
    bad['extrinsics'][0]['rotation_xyzw'] = [1.0, 1.0, 0.0, 0.0]               # not a unit quaternion
    with pytest.raises(DataError) as err:
        CalibrationRecord.build(bad, vocab)
    assert err.value.code == INVALID_VALUE


def test_store_keeps_calibrations_immutable(store, vocab):
    record = calibration(vocab)
    assert store.put_calibration(record) == record.calibration_id
    assert store.put_calibration(record) == record.calibration_id               # idempotent
    assert store.get_calibration(record.calibration_id, vocab) == record
    path = store.path(store.calibration_rel(record.calibration_id))
    doc = json.loads(path.read_text())
    doc['sha256'] = 'sha256:' + '1' * 64
    path.write_text(json.dumps(doc))
    with pytest.raises(DataError) as err:
        store.put_calibration(record)                       # never overwritten
    assert err.value.code == CALIBRATION_CONFLICT
    with pytest.raises(DataError) as err:
        store.get_calibration('cal-0123456789abcdef', vocab)
    assert err.value.code == CALIBRATION_NOT_FOUND


@pytest.mark.parametrize('K', [
    [0.0] * 9,                                                  # an all-zero CameraInfo (some drivers publish one)
    [0.0, 0.0, 8.0, 0.0, 0.0, 6.0, 0.0, 0.0, 1.0],              # REP-104: K[0] == 0 means "uncalibrated"
    [-4.0, 0.0, 8.0, 0.0, 4.0, 6.0, 0.0, 0.0, 1.0],             # a negative focal length
    [4.0, 0.0, 8.0, 0.0, 4.0, 6.0, 0.0, 0.0, 0.0],              # not a camera matrix (last row 0 0 0)
], ids=['all_zero', 'rep104_uncalibrated', 'negative_fx', 'bad_last_row'])
def test_an_uncalibrated_camera_matrix_is_refused(vocab, K):
    """D0-I "bad calibration": a record never holds a camera matrix nobody can project with."""
    bad = calibration_content()
    bad['intrinsics']['K'] = K
    with pytest.raises(DataError) as err:
        CalibrationRecord.build(bad, vocab)
    assert err.value.code == INVALID_VALUE and 'K' in str(err.value)


def test_a_stored_uncalibrated_record_stays_readable_and_verify_flags_it(store, vocab):
    """Immutability (architecture review, D0.9): the rule refuses NEW records only. A record written before 0.9.0
    with an uncalibrated K is still read, and ``verify`` reports it as a WARNING, never an ERROR."""
    import numpy as np
    from _helpers import FRAME, SENSOR, SENSORS, stamp, sync, trigger
    from trainit_data import _strict as s
    from trainit_data.integrity import verify_session
    from trainit_data.writer import ImageSample
    old = calibration_content()
    old['intrinsics']['K'] = [0.0] * 9
    old['schema'] = 'trainit-calibration/1.0'
    record = CalibrationRecord._parse(s.Fields(old, 'calibration'), vocab)      # how a 0.8 writer built it
    cal_id = store.put_calibration(record)
    assert store.get_calibration(cal_id, vocab).intrinsics.calibrated is False      # readable
    writer = store.create_session(source={'transport': 'LIVE_ROS', 'backend': 'mock'},
                                  clock={'session_domain': 'ROS_TIME'}, sensors=[dict(SENSORS[0], streams=[
                                      SENSORS[0]['streams'][0]])])
    at = stamp()
    writer.append_capture(trigger=trigger(at), outcome='COMPLETE', sync=sync('SINGLE_STREAM', None),
                          samples=[ImageSample(SENSOR, 'color', at, np.zeros((6, 8, 3), np.uint8), 'rgb8', FRAME,
                                               calibration_id=cal_id)])
    report = verify_session(store, writer.close().session_id)
    assert report.ok, report.to_dict()
    warnings = [f for f in report.findings if f.severity == 'WARNING' and f.code == INVALID_VALUE]
    assert len(warnings) == 1 and cal_id in warnings[0].detail


@pytest.mark.parametrize('K', [
    [4.0, 0.25, 4.0, 0.0, 4.0, 3.0, 0.0, 0.0, 1.0],              # a skewed camera stays legal
    [4.0, 0.0, 4.0, 1e-12, 4.0, 3.0, 0.0, 0.0, 0.9999999],       # a driver's float noise on the structure
], ids=['skew', 'float_noise'])
def test_a_calibrated_camera_matrix_is_accepted(vocab, K):
    content = calibration_content()
    content['intrinsics']['K'] = K
    assert CalibrationRecord.build(content, vocab).intrinsics.calibrated
