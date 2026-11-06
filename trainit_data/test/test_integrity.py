"""verify (acceptance 11) and recover (acceptance 12). verify inspects only; recover moves, never deletes."""
import json
import os

import pytest

from _helpers import calibration, open_session, rgbd, snapshot, stamp, sync, trigger
from trainit_data._fsio import SessionLock
from trainit_data.errors import (CALIBRATION_NOT_FOUND, HASH_MISMATCH, INDEX_CORRUPT, ORPHAN_FILE, PAYLOAD_MISSING,
                                 RECOVERY_UNSAFE, SESSION_IN_USE, SESSION_INCONSISTENT, SESSION_SEALED, DataError)
from trainit_data.integrity import recover_session, verify_session, verify_store


def _session(store, vocab, captures=3, seal=True):
    cal = store.put_calibration(calibration(vocab, 'color'))
    writer = open_session(store)
    for i in range(captures):
        at = stamp(100 + i)
        writer.append_capture(trigger=trigger(at, i), outcome='COMPLETE', sync=sync(),
                              samples=rgbd(at, i, color_cal=cal))
    if seal:
        writer.close()
    return writer


def _codes(report):
    return [(f.code, f.severity) for f in report.findings]


# ---- verify -------------------------------------------------------------------------------------------

def test_a_clean_session_verifies(store, vocab):
    writer = _session(store, vocab)
    report = verify_session(store, writer.session_id)
    assert report.ok and report.findings == [] and report.captures == 3 and report.observations == 6
    assert verify_store(store).ok


def test_one_corrupted_byte_is_found_and_named(store, vocab):
    writer = _session(store, vocab)
    capture = list(store.open_session(writer.session_id).captures())[1]
    target = capture.observations[1]                                   # the depth payload of capture 2
    path = store.path(target.payload.path)
    data = bytearray(path.read_bytes())
    data[len(data) // 2] ^= 0x01
    path.write_bytes(bytes(data))
    before = snapshot(store.root)
    report = verify_session(store, writer.session_id)
    assert not report.ok
    assert [(f.code, f.artifact, f.record) for f in report.findings] == [
        (HASH_MISMATCH, target.payload.path, target.observation_id)]
    assert snapshot(store.root) == before                              # verify never repairs
    with pytest.raises(DataError) as err:
        store.open_session(writer.session_id).load_payload(target)
    assert err.value.code == HASH_MISMATCH


def test_a_tampered_index_is_found(store, vocab):
    writer = _session(store, vocab)
    index = store.session_dir(writer.session_id) / 'captures.jsonl'
    lines = index.read_bytes().splitlines(keepends=True)
    record = json.loads(lines[0])
    record['trigger']['reason']['note'] = 'edited'
    lines[0] = json.dumps(record, sort_keys=True, separators=(',', ':')).encode() + b'\n'
    index.write_bytes(b''.join(lines))
    report = verify_session(store, writer.session_id)
    assert (HASH_MISMATCH, 'ERROR') in _codes(report)                  # the sealed index hash
    assert report.findings[0].artifact.endswith('captures.jsonl')


def test_a_garbled_line_missing_payload_and_counter_lie_are_found(store, vocab):
    writer = _session(store, vocab)
    session_dir = store.session_dir(writer.session_id)
    os.remove(next((session_dir / 'payload' / '000003').iterdir()))
    header = json.loads((session_dir / 'session.json').read_text())
    header['counters']['captures'].update(complete=2, partial=1)      # sums to attempted, but false
    (session_dir / 'session.json').write_text(json.dumps(header))
    with open(session_dir / 'captures.jsonl', 'r+b') as handle:
        handle.seek(10)
        handle.write(b'#')
    codes = {code for code, _ in _codes(verify_session(store, writer.session_id))}
    assert {INDEX_CORRUPT, PAYLOAD_MISSING, SESSION_INCONSISTENT, HASH_MISMATCH} <= codes


def test_a_missing_calibration_and_an_extra_file_are_found(store, vocab):
    writer = _session(store, vocab, captures=1)
    cal_id = store.open_session(writer.session_id).header.calibration_ids[0]
    os.rename(store.path(store.calibration_rel(cal_id)), store.root / 'moved_away.json')
    (store.session_dir(writer.session_id) / 'payload' / 'stray.bin').write_bytes(b'x')
    codes = _codes(verify_session(store, writer.session_id))
    assert (CALIBRATION_NOT_FOUND, 'ERROR') in codes and (ORPHAN_FILE, 'ERROR') in codes


# ---- recover ------------------------------------------------------------------------------------------

def _crash(store, vocab):
    """An ACTIVE session as a killed writer leaves it: 2 committed captures, then a payload written but its index
    line torn half-way, plus a never-renamed .tmp."""
    writer = _session(store, vocab, captures=2, seal=False)
    writer.release()                                               # the process "dies" (its lock is gone)
    session_dir = store.session_dir(writer.session_id)
    payload = session_dir / 'payload' / '000003'
    payload.mkdir()
    (payload / 'overhead_camera__color.png').write_bytes(b'\x89PNG committed but never indexed')
    (payload / 'overhead_camera__depth.png.tmp').write_bytes(b'half a write')
    with open(session_dir / 'captures.jsonl', 'ab') as handle:
        handle.write(b'{"schema":"trainit-capture/1.0","capture_id":"ses-')       # torn: no newline
    return writer.session_id, session_dir


def test_recover_truncates_the_torn_tail_moves_orphans_and_deletes_nothing(store, vocab):
    session_id, session_dir = _crash(store, vocab)
    before = snapshot(session_dir)
    report = recover_session(store, session_id)
    after = snapshot(session_dir)

    # Every byte that existed still exists somewhere in the session (the torn tail as a copy).
    torn = b'{"schema":"trainit-capture/1.0","capture_id":"ses-'
    assert after['orphans/captures.jsonl.torn'] == torn
    color = 'payload/000003/overhead_camera__color.png'
    assert after['orphans/' + color] == before[color]
    assert after['orphans/payload/000003/overhead_camera__depth.png.tmp'] == b'half a write'
    assert after['captures.jsonl'] == before['captures.jsonl'][:-len(torn)]
    for rel, data in before.items():
        if rel not in ('captures.jsonl', 'session.json', '.active'):
            assert data in after.values(), rel

    assert report['committed_captures'] == 2 and report['torn_tail']['bytes'] == len(torn)
    assert {q['from'] for q in report['quarantined']} == {
        'payload/000003/overhead_camera__color.png', 'payload/000003/overhead_camera__depth.png.tmp'}
    header = store.open_session(session_id).header
    assert header.status == 'INTERRUPTED' and header.closed['by'] == 'recover'
    assert header.counters['captures'] == {'attempted': 3, 'complete': 2, 'partial': 0, 'failed': 0}
    assert header.warnings[-1]['code'] == 'SESSION_INTERRUPTED'
    assert not (session_dir / '.active').exists()
    assert verify_session(store, session_id).ok


def test_recover_is_refused_on_a_sealed_session(store, vocab):
    writer = _session(store, vocab, captures=1)
    with pytest.raises(DataError) as err:
        recover_session(store, writer.session_id)
    assert err.value.code == SESSION_SEALED


def test_recover_is_refused_while_a_live_writer_holds_the_session(store, vocab):
    writer = _session(store, vocab, captures=1, seal=False)
    before = snapshot(store.root)
    with pytest.raises(DataError) as err:
        recover_session(store, writer.session_id)
    assert err.value.code == SESSION_IN_USE
    assert snapshot(store.root) == before
    writer.close()


def test_recover_refuses_to_guess_about_a_damaged_committed_record(store, vocab):
    session_id, session_dir = _crash(store, vocab)
    payload = next((session_dir / 'payload' / '000001').glob('*.png'))
    payload.write_bytes(payload.read_bytes()[:-1])                    # a COMMITTED payload is damaged
    before = snapshot(store.root)
    with pytest.raises(DataError) as err:
        recover_session(store, session_id)
    assert err.value.code == RECOVERY_UNSAFE
    assert err.value.context['findings'][0]['code'] == HASH_MISMATCH
    assert snapshot(store.root) == before                             # nothing changed


def test_recover_refuses_a_corrupt_middle_line(store, vocab):
    session_id, session_dir = _crash(store, vocab)
    index = session_dir / 'captures.jsonl'
    data = index.read_bytes()
    index.write_bytes(data[:5] + b'X' + data[6:])
    before = snapshot(store.root)
    with pytest.raises(DataError) as err:
        recover_session(store, session_id)
    assert err.value.code == RECOVERY_UNSAFE
    assert snapshot(store.root) == before


def test_recover_without_a_torn_tail_or_orphans(store, vocab):
    writer = _session(store, vocab, captures=2, seal=False)
    writer.release()
    report = recover_session(store, writer.session_id)
    assert 'torn_tail' not in report and report['quarantined'] == []
    assert store.open_session(writer.session_id).header.counters['captures']['attempted'] == 2
    assert verify_session(store, writer.session_id).ok


def test_an_active_session_with_a_torn_tail_verifies_with_warnings_only(store, vocab):
    session_id, _ = _crash(store, vocab)
    report = verify_session(store, session_id)
    assert report.ok and {sev for _, sev in _codes(report)} == {'WARNING'}


def test_the_session_lock_is_exclusive(tmp_path):
    marker = tmp_path / '.active'
    first = SessionLock(marker, create=True)
    with pytest.raises(BlockingIOError):
        SessionLock(marker, create=False)
    first.release(remove=False)
    SessionLock(marker, create=False).release(remove=True)
    assert not marker.exists()
