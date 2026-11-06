"""SessionWriter (acceptance 10): a synthetic RGB + 16UC1 + 32FC1 session, bit-exact, hashed, sealed, immutable."""
import hashlib
import json

import numpy as np
import pytest

import trainit_data

from _helpers import (FRAME, SENSOR, calibration, depth_m, depth_mm, open_session, rgb, rgbd, snapshot, stamp,
                      sync, trigger)
from trainit_data.errors import (CALIBRATION_MISMATCH, CALIBRATION_NOT_FOUND, ENCODING_REFUSED, INVALID_VALUE,
                                 SESSION_FAULTED, SESSION_SEALED, DataError)
from trainit_data.integrity import verify_session
from trainit_data.writer import ImageSample


def _write_session(store, vocab, captures=3):
    color_cal = store.put_calibration(calibration(vocab, 'color'))
    depth_cal = store.put_calibration(calibration(vocab, 'depth'))
    writer = open_session(store, recording_config={'resolved': {'triggers': [{'id': 'manual', 'kind': 'MANUAL'}]}})
    expected = []
    for i in range(captures):
        at = stamp(100 + i)
        depth = '16UC1' if i % 2 == 0 else '32FC1'
        samples = rgbd(at, seed=i, color_cal=color_cal, depth_cal=depth_cal, depth=depth)
        capture = writer.append_capture(trigger=trigger(at, i), outcome='COMPLETE', sync=sync(), samples=samples)
        expected.append((capture, samples))
    return writer, expected, (color_cal, depth_cal)


def test_rgb_16uc1_32fc1_round_trip_bit_exact_with_recorded_sha256(store, vocab):
    writer, expected, cals = _write_session(store, vocab)
    header = writer.close()
    assert header.status == 'CLOSED' and header.closed['by'] == 'writer'
    assert header.counters['captures'] == {'attempted': 3, 'complete': 3, 'partial': 0, 'failed': 0}
    assert list(header.calibration_ids) == list(cals)
    reader = store.open_session(header.session_id)
    stored = list(reader.captures())
    assert [c.to_dict() for c in stored] == [c.to_dict() for c, _ in expected]
    for capture, samples in expected:
        for observation, sample in zip(capture.observations, samples):
            data = store.path(observation.payload.path).read_bytes()
            assert observation.payload.sha256 == 'sha256:' + hashlib.sha256(data).hexdigest()
            assert observation.payload.bytes == len(data)
            array = reader.load_payload(observation)
            assert array.dtype == sample.array.dtype and array.shape == sample.array.shape
            assert array.tobytes() == sample.array.tobytes()        # bit-exact, NaN / +-inf included
    media = {(o.data.encoding, o.payload.media_type) for c, _ in expected for o in c.observations}
    assert media == {('rgb8', 'image/png'), ('16UC1', 'image/png;bits=16'), ('32FC1', 'application/x-npy')}
    assert verify_session(store, header.session_id).ok


def test_units_and_frames_are_declared_not_converted(store, vocab):
    writer, expected, _ = _write_session(store, vocab, captures=2)
    writer.close()
    depth16, depth32 = expected[0][0].observations[1].data, expected[1][0].observations[1].data
    assert (depth16.encoding, depth16.unit_scale_m) == ('16UC1', 0.001)
    assert (depth32.encoding, depth32.unit_scale_m) == ('32FC1', 1.0)
    assert depth16.frame_id == FRAME


def test_the_sealed_index_hash_covers_every_line(store, vocab):
    writer, _, _ = _write_session(store, vocab)
    header = writer.close()
    index = store.path(header.index['path']).read_bytes()
    assert header.index['sha256'] == 'sha256:' + hashlib.sha256(index).hexdigest()
    assert header.index['lines'] == index.count(b'\n') == 3
    assert all(json.loads(line)['schema'] == 'trainit-capture/1.0' for line in index.splitlines())


def test_after_seal_every_write_is_refused(store, vocab):
    writer, _, _ = _write_session(store, vocab, captures=1)
    writer.close()
    before = snapshot(store.root)
    at = stamp(200)
    for attempt in (lambda: writer.append_capture(trigger=trigger(at), outcome='COMPLETE', sync=sync(),
                                                  samples=rgbd(at)),
                    writer.close):
        with pytest.raises(DataError) as err:
            attempt()
        assert err.value.code == SESSION_SEALED
    assert snapshot(store.root) == before
    assert not (store.session_dir(writer.session_id) / '.active').exists()


def test_the_active_session_has_a_marker_and_no_sealed_parts(store, vocab):
    writer = open_session(store)
    session_json = json.loads((store.session_dir(writer.session_id) / 'session.json').read_text())
    assert session_json['status'] == 'ACTIVE' and 'index' not in session_json and 'counters' not in session_json
    assert session_json['provenance']['software']['trainit_data'] == trainit_data.__version__
    assert (store.session_dir(writer.session_id) / '.active').exists()
    writer.close()


@pytest.mark.parametrize('sample, code', [
    (ImageSample(SENSOR, 'color', stamp(), rgb().astype(np.uint16), 'rgb8', FRAME), ENCODING_REFUSED),
    (ImageSample(SENSOR, 'depth', stamp(), depth_mm().astype(np.float32), '16UC1', FRAME, unit_scale_m=0.001),
     ENCODING_REFUSED),
    (ImageSample(SENSOR, 'depth', stamp(), depth_m(), '32FC1', FRAME, unit_scale_m=1.0,
                 media_type='image/png;bits=16'), ENCODING_REFUSED),
    (ImageSample(SENSOR, 'depth', stamp(), depth_m(), 'rgb8', FRAME, unit_scale_m=1.0), ENCODING_REFUSED),
    (ImageSample(SENSOR, 'depth', stamp(), depth_mm(), '16UC1', FRAME), INVALID_VALUE),      # unit not declared
    (ImageSample(SENSOR, 'thermal', stamp(), depth_mm(), '16UC1', FRAME, unit_scale_m=0.001), INVALID_VALUE),
    (ImageSample(SENSOR, 'color', stamp(), rgb(), 'rgb8', FRAME, calibration_id='cal-0123456789abcdef'),
     CALIBRATION_NOT_FOUND),
])
def test_invalid_samples_are_refused_before_any_byte_is_written(store, vocab, sample, code):
    writer = open_session(store)
    before = snapshot(store.root)
    with pytest.raises(DataError) as err:
        writer.append_capture(trigger=trigger(stamp()), outcome='COMPLETE',
                              sync=sync('SINGLE_STREAM', None, reference=sample.stream_key), samples=[sample])
    assert err.value.code == code
    assert snapshot(store.root) == before                     # no payload, no index line
    writer.close()


def test_a_calibration_that_does_not_describe_the_image_is_refused(store, vocab):
    other = store.put_calibration(calibration(vocab, 'depth'))          # the depth record on the colour stream
    writer = open_session(store)
    with pytest.raises(DataError) as err:
        writer.append_capture(trigger=trigger(stamp()), outcome='COMPLETE', sync=sync('SINGLE_STREAM', None),
                              samples=[ImageSample(SENSOR, 'color', stamp(), rgb(), 'rgb8', FRAME,
                                                   calibration_id=other)])
    assert err.value.code == CALIBRATION_MISMATCH
    writer.close()


def test_the_trigger_is_on_the_session_clock(store, vocab):
    writer = open_session(store)
    sim = stamp(1, domain='SIM_TIME', origin='RENDER_TICK', sim_source='isaac')
    with pytest.raises(DataError) as err:
        writer.append_capture(trigger=trigger(sim), outcome='COMPLETE', sync=sync(), samples=rgbd(stamp()))
    assert err.value.code == INVALID_VALUE
    with pytest.raises(DataError):            # an observation off the session clock is never COMPLETE / SYNCED
        writer.append_capture(trigger=trigger(stamp()), outcome='COMPLETE', sync=sync(), samples=rgbd(sim))
    writer.close()


def test_partial_and_failed_captures_are_recorded_not_dropped(store, vocab):
    writer = open_session(store)
    at = stamp(300)
    partial = writer.append_capture(trigger=trigger(at, 0), outcome='PARTIAL', sync=sync('SINGLE_STREAM', None),
                                    samples=rgbd(at)[:1],
                                    missing=[{'stream': f'{SENSOR}/depth', 'reason': 'TIMEOUT', 'detail': '1000 ms'}])
    failed = writer.append_capture(trigger=trigger(at, 1), outcome='FAILED',
                                   missing=[{'stream': f'{SENSOR}/color', 'reason': 'NO_CAMERA_INFO'},
                                            {'stream': f'{SENSOR}/depth', 'reason': 'NO_CAMERA_INFO'}])
    header = writer.close()
    assert (partial.outcome, failed.outcome) == ('PARTIAL', 'FAILED')
    assert header.counters['captures'] == {'attempted': 2, 'complete': 0, 'partial': 1, 'failed': 1}
    assert [c.outcome for c in store.open_session(header.session_id).captures()] == ['PARTIAL', 'FAILED']
    assert verify_session(store, header.session_id).ok


def test_appending_does_not_reread_the_index(store, vocab, monkeypatch):
    """O(1) per append: the index hash is incremental; nothing re-reads captures.jsonl while writing."""
    writer = open_session(store)
    opened = []
    real_open = open

    def spy(path, *args, **kwargs):
        opened.append(str(path))
        return real_open(path, *args, **kwargs)
    monkeypatch.setattr('builtins.open', spy)
    for i in range(5):
        at = stamp(400 + i)
        writer.append_capture(trigger=trigger(at, i), outcome='COMPLETE', sync=sync(), samples=rgbd(at, i))
    header = writer.close()
    assert not [p for p in opened if p.endswith('captures.jsonl')]
    assert header.index['lines'] == 5


def test_released_writer_leaves_an_active_session(store, vocab):
    writer = open_session(store)
    at = stamp()
    writer.append_capture(trigger=trigger(at), outcome='COMPLETE', sync=sync(), samples=rgbd(at))
    writer.release()
    with pytest.raises(DataError) as err:
        writer.append_capture(trigger=trigger(at, 1), outcome='COMPLETE', sync=sync(), samples=rgbd(at))
    assert err.value.code == SESSION_FAULTED
    assert store.open_session(writer.session_id).header.status == 'ACTIVE'


def test_a_context_manager_seals_on_success_and_leaves_active_on_error(store, vocab):
    with open_session(store) as writer:
        at = stamp()
        writer.append_capture(trigger=trigger(at), outcome='COMPLETE', sync=sync(), samples=rgbd(at))
    assert store.open_session(writer.session_id).header.status == 'CLOSED'
    with pytest.raises(RuntimeError):
        with open_session(store) as second:
            raise RuntimeError('caller failure')
    assert store.open_session(second.session_id).header.status == 'ACTIVE'
