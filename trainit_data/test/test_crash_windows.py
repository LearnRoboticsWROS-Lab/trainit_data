"""Crash windows and races found by the D0.1 integrity review, each reproduced and closed."""
import dataclasses
import errno
import json
import os

import pytest

from _helpers import SENSOR, calibration, calibration_content, open_session, rgb, rgbd, snapshot, stamp, sync, trigger
from trainit_data import _fsio
from trainit_data._canonical import sha256_bytes
from trainit_data.errors import (CALIBRATION_MISMATCH, HASH_MISMATCH, ORPHAN_FILE, PAYLOAD_INVALID,
                                 RECOVERY_UNSAFE, SESSION_FAULTED, SESSION_IN_USE, SESSION_INCONSISTENT,
                                 SESSION_NOT_FOUND, SESSION_SEALED, STALE_ACTIVE_MARKER, WRITE_FAILED, DataError)
from trainit_data.integrity import recover_session, verify_session, verify_store
from trainit_data.writer import ImageSample


def _one_capture(store, close=True):
    writer = open_session(store)
    at = stamp()
    writer.append_capture(trigger=trigger(at), outcome='COMPLETE', sync=sync(), samples=rgbd(at))
    if close:
        writer.close()
    return writer


# ---- B1: recover never seals over a writer that sealed meanwhile ---------------------------------------

def test_recover_after_a_concurrent_seal_is_refused_and_leaves_no_marker(store, vocab):
    writer = _one_capture(store, close=False)
    with pytest.raises(DataError) as err:
        recover_session(store, writer.session_id)
    assert err.value.code == SESSION_IN_USE
    header = writer.close()                       # the writer seals while recover was being retried
    before = snapshot(store.root)
    with pytest.raises(DataError) as err:
        recover_session(store, writer.session_id)
    assert err.value.code == SESSION_SEALED
    assert snapshot(store.root) == before         # no INTERRUPTED overwrite, no stale `.active`
    assert store.open_session(writer.session_id).header == header


def test_a_lock_on_a_marker_replaced_meanwhile_is_not_a_lock(tmp_path, monkeypatch):
    marker = tmp_path / '.active'
    marker.write_bytes(b'')
    real_flock = _fsio.fcntl.flock

    def flock_after_replacement(fd, flags):
        os.unlink(marker)                         # the sealing writer removes it ...
        marker.write_bytes(b'')                   # ... and someone creates a new one
        return real_flock(fd, flags)
    monkeypatch.setattr(_fsio.fcntl, 'flock', flock_after_replacement)
    with pytest.raises(BlockingIOError):
        _fsio.SessionLock(marker, create=False)


# ---- M1 / M2: write-path accounting --------------------------------------------------------------------

def test_a_renamed_payload_whose_directory_fsync_failed_is_quarantined(store, vocab, monkeypatch):
    writer = open_session(store)
    real = _fsio.fsync_dir
    calls = {'n': 0}

    def failing_fsync_dir(path):
        if str(path).endswith('/000001') and calls['n'] == 0:
            calls['n'] += 1
            raise OSError(errno.EIO, 'I/O error')
        return real(path)
    monkeypatch.setattr(_fsio, 'fsync_dir', failing_fsync_dir)
    at = stamp()
    with pytest.raises(DataError) as err:
        writer.append_capture(trigger=trigger(at), outcome='COMPLETE', sync=sync(), samples=rgbd(at))
    assert err.value.code == WRITE_FAILED
    monkeypatch.undo()
    header = writer.close()
    assert header.counters['write_errors'] == 1
    assert {m['from'] for m in header.warnings[-1]['context']['moved']} == {'payload/000001/overhead_camera__color.png'}
    assert verify_session(store, writer.session_id).ok


def test_a_payload_directory_failure_counts_its_write_errors(store, vocab, monkeypatch):
    writer = open_session(store)
    real_mkdir = type(store.root).mkdir

    def failing_mkdir(self, *args, **kwargs):
        if self.name == '000001':
            raise OSError(errno.ENOSPC, 'No space left on device')
        return real_mkdir(self, *args, **kwargs)
    monkeypatch.setattr(type(store.root), 'mkdir', failing_mkdir)
    at = stamp()
    with pytest.raises(DataError):
        writer.append_capture(trigger=trigger(at), outcome='COMPLETE', sync=sync(), samples=rgbd(at))
    monkeypatch.undo()
    header = writer.close()
    assert header.counters['write_errors'] == 2 and header.counters['captures']['failed'] == 1
    assert verify_session(store, writer.session_id).ok


def test_an_unexpected_error_mid_capture_faults_the_writer(store, vocab, monkeypatch):
    writer = open_session(store)
    monkeypatch.setattr('trainit_data._codecs.encode', lambda *a: (_ for _ in ()).throw(KeyboardInterrupt()))
    at = stamp()
    with pytest.raises(KeyboardInterrupt):
        writer.append_capture(trigger=trigger(at), outcome='COMPLETE', sync=sync(), samples=rgbd(at))
    monkeypatch.undo()
    with pytest.raises(DataError) as err:        # the seq is never reused over a half-written directory
        writer.append_capture(trigger=trigger(at), outcome='COMPLETE', sync=sync(), samples=rgbd(at))
    assert err.value.code == SESSION_FAULTED
    writer.release()
    recover_session(store, writer.session_id)
    assert verify_session(store, writer.session_id).ok


# ---- M3 / m6: session creation --------------------------------------------------------------------------

def test_an_interrupted_creation_commits_nothing_and_is_reported_not_fatal(store, vocab, monkeypatch):
    real = _fsio.write_new_file

    def no_header(path, data):
        if str(path).endswith('session.json'):
            raise OSError(errno.ENOSPC, 'No space left on device')
        return real(path, data)
    monkeypatch.setattr(_fsio, 'write_new_file', no_header)
    with pytest.raises(DataError) as err:
        open_session(store)
    assert err.value.code == WRITE_FAILED
    monkeypatch.undo()
    [session_id] = store.list_sessions()
    report = verify_store(store)
    assert report.ok and [(f.code, f.severity) for f in report.findings] == [(SESSION_NOT_FOUND, 'WARNING')]
    with pytest.raises(DataError) as err:
        recover_session(store, session_id)
    assert err.value.code == SESSION_NOT_FOUND
    # The lock of the failed creation was released: nothing holds the directory.
    _fsio.SessionLock(store.session_dir(session_id) / '.active', create=False).release(remove=False)


# ---- recover re-run, collisions, refusals ---------------------------------------------------------------

def test_recover_can_be_rerun_after_an_interrupted_recover(store, vocab):
    writer = _one_capture(store, close=False)
    writer.release()
    session_dir = store.session_dir(writer.session_id)
    (session_dir / 'orphans').mkdir()
    (session_dir / 'orphans' / 'captures.jsonl.torn').write_bytes(b'{"half')     # cut by the first attempt
    (session_dir / 'recovery.json').write_bytes(b'{"partial')                      # its report, never finished
    (session_dir / 'session.json.tmp').write_bytes(b'{"partial header')
    report = recover_session(store, writer.session_id)
    assert {q['from'] for q in report['quarantined']} == {'recovery.json', 'session.json.tmp'}
    assert report['earlier_torn_tails'] == ['orphans/captures.jsonl.torn']
    header = store.open_session(writer.session_id).header
    assert header.status == 'INTERRUPTED' and header.counters['captures']['attempted'] == 2
    assert verify_session(store, writer.session_id).ok


def test_quarantine_never_overwrites(tmp_path):
    (tmp_path / 'payload').mkdir()
    for content in (b'first', b'second'):
        (tmp_path / 'payload' / 'x.png').write_bytes(content)
        _fsio.quarantine(tmp_path, 'payload/x.png')
    assert (tmp_path / 'orphans/payload/x.png').read_bytes() == b'first'
    assert (tmp_path / 'orphans/payload/x.png.1').read_bytes() == b'second'


def test_recover_refuses_unexpected_entries(store, vocab):
    writer = _one_capture(store, close=False)
    writer.release()
    (store.session_dir(writer.session_id) / 'notes.txt').write_text('a person put this here')
    before = snapshot(store.root)
    with pytest.raises(DataError) as err:
        recover_session(store, writer.session_id)
    assert err.value.code == RECOVERY_UNSAFE and snapshot(store.root) == before


# ---- verify gaps ---------------------------------------------------------------------------------------

def test_verify_store_checks_calibration_files(store, vocab):
    record = calibration(vocab)
    store.put_calibration(record)
    path = store.path(store.calibration_rel(record.calibration_id))
    doc = json.loads(path.read_text())
    doc['optical_frame'] = 'edited'
    path.write_text(json.dumps(doc))
    findings = verify_store(store).findings
    assert [(f.code, f.artifact) for f in findings] == [(HASH_MISMATCH, f'calibrations/{path.name}')]
    path.write_text(json.dumps(record.to_dict()))
    path.rename(path.with_name('cal-0000000000000000.json'))
    assert [f.code for f in verify_store(store).findings] == [HASH_MISMATCH]


@pytest.mark.parametrize('tamper, code, severity', [
    (lambda h, d: h.update(calibration_ids=['cal-0123456789abcdef']), SESSION_INCONSISTENT, 'ERROR'),
    (lambda h, d: h['counters']['captures'].update(attempted=0, complete=0), SESSION_INCONSISTENT, 'ERROR'),
    (lambda h, d: (d / '.active').write_bytes(b''), STALE_ACTIVE_MARKER, 'WARNING'),
    (lambda h, d: h.update(status='INTERRUPTED', closed={'wall_utc_ns': 1, 'by': 'recover'}),
     'MALFORMED_RECORD', 'ERROR'),
])
def test_sealed_header_consistency_checks(store, vocab, tamper, code, severity):
    cal = store.put_calibration(calibration(vocab))
    writer = open_session(store)
    at = stamp()
    writer.append_capture(trigger=trigger(at), outcome='COMPLETE', sync=sync(),
                          samples=rgbd(at, color_cal=cal))
    writer.close()
    session_dir = store.session_dir(writer.session_id)
    header = json.loads((session_dir / 'session.json').read_text())
    tamper(header, session_dir)
    (session_dir / 'session.json').write_text(json.dumps(header))
    findings = [(f.code, f.severity) for f in verify_session(store, writer.session_id).findings]
    assert (code, severity) in findings


def test_a_symlinked_payload_is_invalid(store, vocab, tmp_path):
    writer = _one_capture(store)
    payload = next((store.session_dir(writer.session_id) / 'payload' / '000001').glob('*color.png'))
    outside = tmp_path / 'outside.png'
    outside.write_bytes(payload.read_bytes())
    payload.unlink()
    payload.symlink_to(outside)
    findings = verify_session(store, writer.session_id).findings
    assert (PAYLOAD_INVALID, 'ERROR') in [(f.code, f.severity) for f in findings]


def test_a_payload_that_hashes_but_does_not_decode_is_invalid(store, vocab):
    writer = _one_capture(store)
    reader = store.open_session(writer.session_id)
    observation = next(reader.captures()).observations[0]
    path = store.path(observation.payload.path)
    path.write_bytes(b'not a png at all')
    forged = dataclasses.replace(observation, payload=dataclasses.replace(
        observation.payload, sha256=sha256_bytes(path.read_bytes()), bytes=path.stat().st_size))
    with pytest.raises(DataError) as err:
        reader.load_payload(forged)
    assert err.value.code == PAYLOAD_INVALID


def test_a_tampered_pinned_vocabulary_is_found(store, vocab):
    writer = _one_capture(store)
    [vocab_file] = (store.root / 'vocab').iterdir()
    doc = json.loads(vocab_file.read_text())
    doc['backends'].append('anything')
    vocab_file.write_text(json.dumps(doc))
    findings = verify_session(store, writer.session_id).findings
    assert [(f.code, f.severity) for f in findings] == [(HASH_MISMATCH, 'ERROR')]


@pytest.mark.parametrize('width, frame', [(9, None), (None, 'another_optical_frame')])
def test_the_writer_refuses_a_calibration_of_another_size_or_frame(store, vocab, width, frame):
    content = {}
    if width is not None:
        intr = calibration_content()['intrinsics']
        content['intrinsics'] = dict(intr, width=width)
    if frame is not None:
        content['optical_frame'] = frame
    cal = store.put_calibration(calibration(vocab, **content))
    writer = open_session(store)
    with pytest.raises(DataError) as err:
        writer.append_capture(trigger=trigger(stamp()), outcome='COMPLETE', sync=sync('SINGLE_STREAM', None),
                              samples=[ImageSample(SENSOR, 'color', stamp(), rgb(), 'rgb8',
                                                   'overhead_camera_optical_frame', calibration_id=cal)])
    assert err.value.code == CALIBRATION_MISMATCH
    writer.close()


def test_orphans_in_a_sealed_session_are_errors(store, vocab):
    writer = _one_capture(store)
    (store.session_dir(writer.session_id) / 'payload' / '000001' / 'late.png').write_bytes(b'x')
    assert (ORPHAN_FILE, 'ERROR') in [(f.code, f.severity) for f in verify_session(store, writer.session_id).findings]
