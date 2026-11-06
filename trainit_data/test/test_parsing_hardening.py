"""Parsing hardening (D0.1 review): hostile or ambiguous input is refused with a stable code, never a raw crash,
and no record can be changed through the dictionaries it hands out."""
import copy
import pickle

import pytest

from _helpers import calibration, calibration_content
from test_canonical import golden_capture_dict
from test_records import session_dict
from trainit_data._canonical import canonical_json, parse_json, sha256_json
from trainit_data.calibration import CalibrationRecord
from trainit_data.capture import Capture
from trainit_data.errors import (HASH_MISMATCH, INVALID_VALUE, MALFORMED_RECORD, SCHEMA_MAJOR_UNSUPPORTED,
                                 UNKNOWN_FIELD, UNKNOWN_VOCAB, DataError)
from trainit_data.session_record import SessionHeader
from trainit_data.timestamps import Timestamp
from trainit_data.vocab import Vocabulary


def test_duplicate_keys_are_refused():
    with pytest.raises(DataError) as err:
        parse_json(b'{"a": 1, "a": 2}', 'dup.json')
    assert err.value.code == MALFORMED_RECORD


def test_deep_nesting_is_a_malformed_record_not_a_crash():
    with pytest.raises(DataError) as err:
        parse_json(b'[' * 100000 + b']' * 100000, 'deep.json')
    assert err.value.code == MALFORMED_RECORD


def test_a_huge_integer_in_a_real_field_is_invalid_not_a_crash(vocab):
    content = calibration_content()
    content['intrinsics']['K'][0] = 1 << 1400
    with pytest.raises(DataError) as err:
        CalibrationRecord.build(content, vocab)
    assert err.value.code == INVALID_VALUE


@pytest.mark.parametrize('bad', [{'x': (1, 2)}, {1: 'int key'}, {'x': float('nan')}, {'x': object()}])
def test_python_built_opaque_content_must_be_plain_json(vocab, bad):
    content = calibration_content()
    content['intrinsics_provenance']['ref'] = bad
    with pytest.raises(DataError) as err:
        CalibrationRecord.build(content, vocab)
    assert err.value.code == INVALID_VALUE


def test_records_hand_out_copies_never_their_own_state(vocab):
    content = calibration_content(extensions={'trainit.test.block': {'x': 1}})
    content['intrinsics_provenance']['ref'] = {'serial': 'a'}
    record = CalibrationRecord.build(content, vocab)
    for out in (record.to_dict(), record.content()):
        out['extensions']['trainit.test.block']['x'] = 99
        out['intrinsics_provenance']['ref']['serial'] = 'b'
    assert sha256_json(record.content()) == record.sha256
    assert CalibrationRecord.from_dict(record.to_dict(), vocab) == record
    capture = Capture.from_dict(golden_capture_dict(), vocab)
    before = canonical_json(capture.to_dict())
    out = capture.to_dict()
    out['extensions']['trainit.test.values'].append(3)
    out['trigger']['reason']['note'] = 'changed'
    assert canonical_json(capture.to_dict()) == before
    header = SessionHeader.from_dict(session_dict(vocab), vocab)
    out = header.to_dict()
    out['source']['backend_details']['realsense']['nested'][1]['x'] = 0
    out['counters']['captures']['complete'] = 0
    assert header.to_dict() == session_dict(vocab)


def test_sim_source_is_a_simulator_not_any_backend(vocab):
    base = {'sec': 1, 'nanosec': 0, 'clock_domain': 'SIM_TIME', 'origin': 'RENDER_TICK'}
    assert Timestamp.from_dict(dict(base, sim_source='isaac'), vocab).sim_source == 'isaac'
    for source in ('realsense', 'mock'):
        with pytest.raises(DataError) as err:
            Timestamp.from_dict(dict(base, sim_source=source), vocab)
        assert err.value.code == UNKNOWN_VOCAB


@pytest.mark.parametrize('record', ['session', 'calibration'])
def test_strictness_and_evolution_hold_for_every_record(vocab, record):
    if record == 'session':
        doc, parse = session_dict(vocab), lambda d: SessionHeader.from_dict(d, vocab)
        name = 'trainit-session'
    else:
        doc, parse = calibration(vocab).to_dict(), lambda d: CalibrationRecord.from_dict(d, vocab)
        name = 'trainit-calibration'
    unknown = dict(copy.deepcopy(doc), surprise=1)
    with pytest.raises(DataError) as err:
        parse(unknown)
    assert err.value.code == UNKNOWN_FIELD
    newer_major = dict(copy.deepcopy(doc), schema=f'{name}/2.0')
    with pytest.raises(DataError) as err:
        parse(newer_major)
    assert err.value.code == SCHEMA_MAJOR_UNSUPPORTED
    if record == 'session':
        extended = dict(copy.deepcopy(doc), schema=f'{name}/1.4', extensions={'trainit.test.x': [1, {'y': 'ü'}]})
        assert canonical_json(parse(copy.deepcopy(extended)).to_dict()) == canonical_json(extended)
    else:
        content = calibration_content(schema=f'{name}/1.4', extensions={'trainit.test.x': [1, {'y': 'ü'}]})
        built = CalibrationRecord.build(content, vocab)
        assert canonical_json(CalibrationRecord.from_dict(built.to_dict(), vocab).to_dict()) == \
            canonical_json(built.to_dict())


def test_active_and_interrupted_headers_round_trip(vocab):
    active = session_dict(vocab, status='ACTIVE')
    for key in ('closed', 'counters', 'index'):
        del active[key]
    interrupted = session_dict(vocab, status='INTERRUPTED', closed={'wall_utc_ns': 3, 'by': 'recover'})
    del interrupted['counters']['writer_overruns']                  # unknown after a crash: absent
    for doc in (active, interrupted):
        assert canonical_json(SessionHeader.from_dict(copy.deepcopy(doc), vocab).to_dict()) == canonical_json(doc)


def test_complete_is_never_out_of_tolerance(vocab):
    doc = golden_capture_dict()
    doc.update(outcome='COMPLETE', missing=[], sync={'policy': {'name': 'FIRST_AT_OR_AFTER_BOUNDARY', 'version': 1},
                                                     'reference': 'overhead_camera/color', 'status': 'OUT_OF_TOLERANCE',
                                                     'max_skew_ns': 0, 'tolerance_ns': 0})
    with pytest.raises(DataError):
        Capture.from_dict(doc, vocab)


def test_sessions_are_not_capped_at_a_million_captures(vocab):
    doc = golden_capture_dict()
    wide = 'ses-20261110T143000Z-9e8d7c6b/1000000'
    doc.update(capture_id=wide, seq=1_000_000)
    doc['observations'][0]['observation_id'] = f'{wide}/overhead_camera/color'
    assert Capture.from_dict(doc, vocab).seq == 1_000_000


def test_data_errors_survive_pickling():
    error = pickle.loads(pickle.dumps(DataError(HASH_MISMATCH, 'x', context={'path': 'p'})))
    assert (error.code, error.context, str(error)) == (HASH_MISMATCH, {'path': 'p'}, 'HASH_MISMATCH: x')


@pytest.mark.parametrize('dtype', ['>u2', 'float128', 'int', 'l', 'object'])
def test_encoding_dtypes_are_explicit_and_platform_independent(vocab, dtype):
    doc = vocab.to_dict()
    doc['encodings']['rgb8']['dtype'] = dtype
    with pytest.raises(DataError) as err:
        Vocabulary.from_dict(doc)
    assert err.value.code == INVALID_VALUE


def test_a_vocabulary_must_keep_the_rows_the_core_records(vocab):
    doc = vocab.to_dict()
    doc['missing_reasons'].remove('WRITE_FAILED')
    with pytest.raises(DataError) as err:
        Vocabulary.from_dict(doc)
    assert err.value.code == INVALID_VALUE
    with pytest.raises(ValueError):
        vocab.require('schema', 't')              # not a category


def test_the_block_version_follows_the_vocabulary_row(vocab):
    doc = vocab.to_dict()
    doc['data_blocks']['image']['version'] = '2.0'
    newer = Vocabulary.from_dict(doc)
    capture = golden_capture_dict()
    with pytest.raises(DataError) as err:
        Capture.from_dict(capture, newer)
    assert err.value.code == SCHEMA_MAJOR_UNSUPPORTED
