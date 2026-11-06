"""Record contracts: round trips, strict parsing, schema evolution, extensions, vocabulary refusal, clocks.

Acceptance 1 (round trip of each record), 3 (unknown field), 4 (extensions lossless), 5 (unsupported major),
6 (vocabulary refusal), 14 (INCOMPATIBLE_CLOCKS).
"""
import copy

import pytest

from _helpers import calibration_content, stamp, sync
from test_canonical import golden_capture_dict
from trainit_data._canonical import canonical_json, sha256_json
from trainit_data.calibration import CalibrationRecord
from trainit_data.capture import Capture
from trainit_data.errors import (INCOMPATIBLE_CLOCKS, INVALID_VALUE, MALFORMED_RECORD, MISSING_FIELD,
                                 SCHEMA_MAJOR_UNSUPPORTED, UNKNOWN_FIELD, UNKNOWN_VOCAB, DataError)
from trainit_data.observation import Observation
from trainit_data.session_record import SessionHeader
from trainit_data.timestamps import Timestamp, compare_timestamps, skew_ns


def session_dict(vocab, **overrides):
    doc = {'schema': 'trainit-session/1.0', 'session_id': 'ses-20261110T143000Z-9e8d7c6b', 'status': 'CLOSED',
           'opened': {'wall_utc_ns': 1}, 'closed': {'wall_utc_ns': 2, 'by': 'writer'},
           'source': {'transport': 'LIVE_ROS', 'backend': 'real',
                      'descriptor': {'intrinsics_kind': 'DEVICE_REPORTED', 'stamp_origin': 'DRIVER_GLOBAL_TIME',
                                     'device': {'serial': '<serial>'}, 'supplied_by': 'example launch'},
                      'backend_details': {'realsense': {'align_depth': True, 'nested': [1, {'x': 2.5}]}}},
           'vocab': {'schema': vocab.schema, 'sha256': vocab.sha256},
           'clock': {'session_domain': 'ROS_TIME', 'use_sim_time': False},
           'sensors': [{'sensor_id': 'overhead_camera', 'profile': {'model': 'example_rgbd'},
                        'streams': [{'stream': 'color', 'modality': 'rgb'}, {'stream': 'depth', 'modality': 'depth'}]}],
           'rights': {'classification': 'INTERNAL'}, 'calibration_ids': ['cal-0123456789abcdef'],
           'recording_config': {'resolved': {'triggers': [{'id': 'manual', 'kind': 'MANUAL'}]},
                                'resolved_sha256': 'sha256:' + '0' * 64},
           'provenance': {'software': {'trainit_data': '0.1.0'}, 'robot': {'robot_id': 'example_robot'}},
           'warnings': [{'code': 'EXAMPLE', 'detail': 'x'}],
           'counters': {'captures': {'attempted': 3, 'complete': 2, 'partial': 1, 'failed': 0}, 'write_errors': 0,
                        'writer_overruns': 0, 'calibration_changes': 0, 'received': {'overhead_camera/color': 90}},
           'index': {'path': 'sessions/ses-20261110T143000Z-9e8d7c6b/captures.jsonl', 'sha256': 'sha256:' + 'f' * 64,
                     'lines': 3}}
    doc['recording_config']['resolved_sha256'] = sha256_json(doc['recording_config']['resolved'])
    doc.update(overrides)
    return doc


def roundtrip(cls, value, vocab, **kw):
    record = cls.from_dict(copy.deepcopy(value), vocab, **kw) if vocab is not None else cls.from_dict(value, **kw)
    return record, record.to_dict()


# ---- acceptance 1: every record round-trips to identical canonical bytes ------------------------------

def test_capture_round_trips(vocab):
    doc = golden_capture_dict()
    _, out = roundtrip(Capture, doc, vocab)
    assert canonical_json(out) == canonical_json(doc)


def test_session_header_round_trips(vocab):
    doc = session_dict(vocab)
    header, out = roundtrip(SessionHeader, doc, vocab)
    assert canonical_json(out) == canonical_json(doc)
    assert header.sealed
    assert header.stream_modalities() == {'overhead_camera/color': 'rgb', 'overhead_camera/depth': 'depth'}


def test_the_acquisition_provenance_block_is_carried_opaque_and_other_blocks_stay_refused(vocab):
    """D0.4: the source topology lives in provenance (typed by its producer), never in a device schema here."""
    acquisition = {'schema': 'trainit-acquisition/1.0', 'sensors': {'overhead_camera': {
        'relation': 'REMOTE_SENSOR_SOURCE', 'source_info': {'device': {'serial_number': '<serial>'}}}}}
    doc = session_dict(vocab, schema='trainit-session/1.1',
                       provenance={'software': {'trainit_data': '0.4.0'}, 'acquisition': acquisition})
    header, out = roundtrip(SessionHeader, doc, vocab)
    assert header.provenance['acquisition'] == acquisition and canonical_json(out) == canonical_json(doc)
    with pytest.raises(DataError) as err:
        SessionHeader.from_dict(session_dict(vocab, provenance={'topology': {}}), vocab)
    assert err.value.code == UNKNOWN_FIELD
    old, _ = roundtrip(SessionHeader, session_dict(vocab), vocab)          # a 1.0 session is still read
    assert old.schema == 'trainit-session/1.0'


def test_calibration_round_trips(vocab):
    record = CalibrationRecord.build(calibration_content('depth'), vocab)
    again = CalibrationRecord.from_dict(record.to_dict(), vocab)
    assert again == record and canonical_json(again.to_dict()) == canonical_json(record.to_dict())


def test_timestamp_round_trips(vocab):
    sim = {'sec': 412, 'nanosec': 250000000, 'clock_domain': 'SIM_TIME', 'origin': 'RENDER_TICK', 'sim_source': 'isaac'}
    assert Timestamp.from_dict(sim, vocab).to_dict() == sim


# ---- acceptance 3: unknown ordinary fields are refused, at any depth -----------------------------------

@pytest.mark.parametrize('path', [(), ('trigger',), ('trigger', 'issued'), ('sync', 'policy'), ('observations', 0),
                                  ('observations', 0, 'data'), ('observations', 0, 'payload'), ('missing', 0)])
def test_unknown_field_refused(vocab, path):
    doc = golden_capture_dict()
    target = doc
    for key in path:
        target = target[key]
    target['surprise'] = 1
    with pytest.raises(DataError) as err:
        Capture.from_dict(doc, vocab)
    assert err.value.code == UNKNOWN_FIELD and 'surprise' in err.value.message


def test_missing_and_null_fields_refused(vocab):
    doc = golden_capture_dict()
    del doc['trigger']
    with pytest.raises(DataError) as err:
        Capture.from_dict(doc, vocab)
    assert err.value.code == MISSING_FIELD
    doc = golden_capture_dict()
    doc['observations'][0]['receipt'] = None
    with pytest.raises(DataError) as err:
        Capture.from_dict(doc, vocab)
    assert err.value.code == INVALID_VALUE


def test_booleans_are_not_integers(vocab):
    doc = golden_capture_dict()
    doc['seq'] = True
    with pytest.raises(DataError) as err:
        Capture.from_dict(doc, vocab)
    assert err.value.code == INVALID_VALUE


# ---- acceptance 4 / 5: schema evolution ---------------------------------------------------------------

def test_newer_minor_with_extensions_is_read_and_preserved_byte_for_byte(vocab):
    doc = golden_capture_dict()
    doc['schema'] = 'trainit-capture/1.7'
    doc['extensions'] = {'trainit.d1.joint_state_ref': {'series': 'ser-01', 'window_ns': [1, 2]},
                         'vendor.example.flag': True, 'trainit.test.text': 'ünïcode', 'trainit.test.large': 2 ** 70}
    doc['observations'][0]['extensions'] = {'trainit.test.float': 0.1}
    record = Capture.from_dict(copy.deepcopy(doc), vocab)
    assert canonical_json(record.to_dict()) == canonical_json(doc)
    assert record.schema == 'trainit-capture/1.7'      # never silently downgraded


def test_a_newer_minor_still_refuses_fields_outside_extensions(vocab):
    doc = golden_capture_dict()
    doc['schema'] = 'trainit-capture/1.7'
    doc['new_optional_field'] = 1
    with pytest.raises(DataError) as err:
        Capture.from_dict(doc, vocab)
    assert err.value.code == UNKNOWN_FIELD


@pytest.mark.parametrize('schema', ['trainit-capture/2.0', 'trainit-capture/0.9'])
def test_unsupported_major_refused(vocab, schema):
    doc = golden_capture_dict()
    doc['schema'] = schema
    with pytest.raises(DataError) as err:
        Capture.from_dict(doc, vocab)
    assert err.value.code == SCHEMA_MAJOR_UNSUPPORTED


def test_unsupported_data_block_major_refused(vocab):
    doc = golden_capture_dict()
    doc['observations'][0]['data']['version'] = '2.0'
    with pytest.raises(DataError) as err:
        Capture.from_dict(doc, vocab)
    assert err.value.code == SCHEMA_MAJOR_UNSUPPORTED


def test_wrong_record_kind_refused(vocab):
    doc = golden_capture_dict()
    doc['schema'] = 'trainit-session/1.0'
    with pytest.raises(DataError) as err:
        Capture.from_dict(doc, vocab)
    assert err.value.code == MALFORMED_RECORD


@pytest.mark.parametrize('key', ['nonamespace', 'Trainit.X', 'trainit.'])
def test_extension_keys_are_namespaced(vocab, key):
    doc = golden_capture_dict()
    doc['extensions'] = {key: 1}
    with pytest.raises(DataError) as err:
        Capture.from_dict(doc, vocab)
    assert err.value.code == INVALID_VALUE


# ---- acceptance 6: vocabulary refusal -----------------------------------------------------------------

@pytest.mark.parametrize('path, value', [
    (('trigger', 'kind'), 'TELEPATHY'),
    (('trigger', 'origin'), 'NOBODY'),
    (('sync', 'policy', 'name'), 'NEAREST_WITHIN_WINDOW'),    # a future policy: not a row of 1.0
    (('sync', 'policy', 'version'), 2),
    (('missing', 0, 'reason'), 'BAD_LUCK'),
    (('observations', 0, 'modality'), 'thermal'),
    (('observations', 0, 'stamp', 'origin'), 'GUESSED'),
])
def test_values_outside_the_vocabulary_are_refused(vocab, path, value):
    doc = golden_capture_dict()
    target = doc
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value
    with pytest.raises(DataError) as err:
        Capture.from_dict(doc, vocab)
    assert err.value.code == UNKNOWN_VOCAB
    assert err.value.context['vocab_sha256'] == vocab.sha256


def test_session_backend_and_transport_are_vocabulary(vocab):
    for key, value in (('backend', 'gazebo'), ('transport', 'CARRIER_PIGEON')):
        doc = session_dict(vocab)
        doc['source'][key] = value
        with pytest.raises(DataError) as err:
            SessionHeader.from_dict(doc, vocab)
        assert err.value.code == UNKNOWN_VOCAB


def test_trigger_context_namespaces_are_vocabulary(vocab):
    doc = golden_capture_dict()
    doc['trigger']['context'] = {'trainit.task_step': {'run_id': 'run-1', 'task_id': 'pick_one', 'step_id': 'approach'}}
    Capture.from_dict(copy.deepcopy(doc), vocab)
    doc['trigger']['context'] = {'trainit.robot_state': {'q': [0.1]}}
    with pytest.raises(DataError) as err:
        Capture.from_dict(doc, vocab)
    assert err.value.code == UNKNOWN_VOCAB


# ---- capture rules ------------------------------------------------------------------------------------

def _two_stream_capture(at_color, at_depth, sync_result, outcome='COMPLETE', missing=()):
    doc = golden_capture_dict()
    color = doc['observations'][0]
    color['stamp'] = at_color.to_dict()
    depth = copy.deepcopy(color)
    depth.update(observation_id=color['observation_id'].replace('/color', '/depth'), stream='depth', modality='depth',
                 stamp=at_depth.to_dict())
    depth['payload'] = dict(color['payload'], path=color['payload']['path'].replace('color.png', 'depth.png'),
                            media_type='image/png;bits=16')
    depth['data'] = dict(color['data'], encoding='16UC1', unit_scale_m=0.001)
    doc.update(observations=[color, depth], sync=sync_result, outcome=outcome, missing=list(missing))
    return doc


def test_recorded_skew_must_match_the_stamps(vocab):
    a, b = stamp(10, 0), stamp(10, 3_000_000)
    Capture.from_dict(_two_stream_capture(a, b, sync('WITHIN_TOLERANCE', 3_000_000, 16_666_667)), vocab)
    for bad in (sync('SYNCED', 0), sync('WITHIN_TOLERANCE', 1, 16_666_667), sync('OUT_OF_TOLERANCE', 5_000_000, 10)):
        with pytest.raises(DataError) as err:
            Capture.from_dict(_two_stream_capture(a, b, bad), vocab)
        assert err.value.code == INVALID_VALUE


def test_observations_in_two_clock_domains_are_never_synced(vocab):
    a, b = stamp(10, 0), stamp(10, 0, 'SIM_TIME', sim_source='isaac')
    with pytest.raises(DataError) as err:
        Capture.from_dict(_two_stream_capture(a, b, sync('SYNCED', 0)), vocab)
    assert err.value.code == INVALID_VALUE and 'INCOMPATIBLE_CLOCKS' in err.value.message
    with pytest.raises(DataError):           # ...and never COMPLETE, even when labelled incompatible
        Capture.from_dict(_two_stream_capture(a, b, sync('INCOMPATIBLE_CLOCKS', None)), vocab)
    # A PARTIAL capture keeps both observations and says why.
    doc = _two_stream_capture(a, b, sync('INCOMPATIBLE_CLOCKS', None), outcome='PARTIAL',
                              missing=[{'stream': 'other_camera/color', 'reason': 'TIMEOUT'}])
    assert Capture.from_dict(doc, vocab).sync.status == 'INCOMPATIBLE_CLOCKS'


@pytest.mark.parametrize('outcome, observations, missing', [
    ('COMPLETE', 1, 1), ('COMPLETE', 0, 0), ('PARTIAL', 1, 0), ('PARTIAL', 0, 1), ('FAILED', 1, 1), ('FAILED', 0, 0)])
def test_outcome_rules(vocab, outcome, observations, missing):
    doc = golden_capture_dict()
    doc['outcome'] = outcome
    doc['observations'] = doc['observations'][:observations]
    doc['missing'] = doc['missing'][:missing]
    with pytest.raises(DataError) as err:
        Capture.from_dict(doc, vocab)
    assert err.value.code == INVALID_VALUE


def test_a_failed_capture_is_representable_with_reasons(vocab):
    doc = golden_capture_dict()
    doc.update(outcome='FAILED', observations=[],
               missing=[{'stream': 'overhead_camera/color', 'reason': 'TIMEOUT'},
                        {'stream': 'overhead_camera/depth', 'reason': 'NO_CAMERA_INFO'}])
    del doc['sync']
    capture = Capture.from_dict(doc, vocab)
    assert [m.reason for m in capture.missing] == ['TIMEOUT', 'NO_CAMERA_INFO']


def test_a_stream_cannot_be_both_observed_and_missing(vocab):
    doc = golden_capture_dict()
    doc['missing'] = [{'stream': 'overhead_camera/color', 'reason': 'TIMEOUT'}]
    with pytest.raises(DataError):
        Capture.from_dict(doc, vocab)


def test_ids_must_be_consistent(vocab):
    doc = golden_capture_dict()
    doc['seq'] = 41
    with pytest.raises(DataError):
        Capture.from_dict(doc, vocab)
    doc = golden_capture_dict()
    doc['observations'][0]['observation_id'] = 'ses-20261110T143000Z-9e8d7c6b/000001/overhead_camera/color'
    with pytest.raises(DataError):
        Capture.from_dict(doc, vocab)


def test_depth_declares_its_unit_and_rgb_does_not(vocab):
    doc = golden_capture_dict()
    doc['observations'][0]['data']['unit_scale_m'] = 0.001
    with pytest.raises(DataError):
        Capture.from_dict(doc, vocab)
    depth = _two_stream_capture(stamp(), stamp(), sync('SYNCED', 0))
    del depth['observations'][1]['data']['unit_scale_m']
    with pytest.raises(DataError):
        Capture.from_dict(depth, vocab)


def test_observation_media_type_follows_the_modality(vocab):
    doc = golden_capture_dict()['observations'][0]
    doc['payload']['media_type'] = 'application/x-npy'
    with pytest.raises(DataError) as err:
        Observation.from_dict(doc, vocab)
    assert err.value.code == INVALID_VALUE


# ---- acceptance 14: clocks ----------------------------------------------------------------------------

def test_timestamps_compare_only_within_one_clock(vocab):
    a, b = stamp(10, 5), stamp(10, 7)
    assert compare_timestamps(a, b) == -1 and compare_timestamps(b, a) == 1 and compare_timestamps(a, a) == 0
    assert skew_ns(a, b) == 2
    pairs = [(stamp(10), stamp(10, domain='SIM_TIME', sim_source='isaac')),
             (stamp(10, domain='SIM_TIME', sim_source='isaac'), stamp(10, domain='SIM_TIME', sim_source='mock')),
             (stamp(10), stamp(10, domain='HOST_MONOTONIC'))]
    for x, y in pairs:
        for fn in (compare_timestamps, skew_ns):
            with pytest.raises(DataError) as err:
                fn(x, y)
            assert err.value.code == INCOMPATIBLE_CLOCKS


def test_timestamps_have_no_ordering_operators():
    with pytest.raises(TypeError):
        stamp(1) < stamp(2)


@pytest.mark.parametrize('value', [
    {'sec': 1, 'nanosec': 1_000_000_000, 'clock_domain': 'ROS_TIME', 'origin': 'HEADER'},
    {'sec': -1, 'nanosec': 0, 'clock_domain': 'ROS_TIME', 'origin': 'HEADER'},
    {'sec': 1, 'nanosec': 0, 'clock_domain': 'SIM_TIME', 'origin': 'HEADER'},                     # no sim_source
    {'sec': 1, 'nanosec': 0, 'clock_domain': 'ROS_TIME', 'origin': 'HEADER', 'sim_source': 'isaac'},
    {'sec': 1, 'nanosec': 0, 'clock_domain': 'GPS_TIME', 'origin': 'HEADER'},                     # closed set
])
def test_invalid_timestamps_refused(vocab, value):
    with pytest.raises(DataError):
        Timestamp.from_dict(value, vocab)


def test_session_lifecycle_rules(vocab):
    active = session_dict(vocab, status='ACTIVE')
    with pytest.raises(DataError):           # an ACTIVE header has no sealed parts yet
        SessionHeader.from_dict(active, vocab)
    for key in ('closed', 'counters', 'index'):
        del active[key]
    SessionHeader.from_dict(active, vocab)
    with pytest.raises(DataError):           # only recover seals INTERRUPTED
        SessionHeader.from_dict(session_dict(vocab, status='INTERRUPTED'), vocab)
    tampered = session_dict(vocab)
    tampered['recording_config']['resolved']['triggers'][0]['id'] = 'other'
    with pytest.raises(DataError):           # the configuration snapshot is hash-checked
        SessionHeader.from_dict(tampered, vocab)


def test_all_streams_present_but_out_of_tolerance_is_partial(vocab):
    a, b = stamp(10, 0), stamp(10, 40_000_000)
    out = sync('OUT_OF_TOLERANCE', 40_000_000, 16_666_667)
    assert Capture.from_dict(_two_stream_capture(a, b, out, outcome='PARTIAL'), vocab).outcome == 'PARTIAL'
    with pytest.raises(DataError):
        Capture.from_dict(_two_stream_capture(a, b, out, outcome='COMPLETE'), vocab)
