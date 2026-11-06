"""Write failures (acceptance 13) through controlled fault injection of ENOSPC.

``_fsio.write_fully`` is the single place bytes reach the OS; the tests replace it to fail with ENOSPC
after writing part of the data (a realistic short write), for chosen files only.
"""
import errno

import pytest

from _helpers import SENSOR, open_session, rgbd, snapshot, stamp, sync, trigger
from trainit_data import _fsio
from trainit_data.errors import SESSION_FAULTED, WRITE_FAILED, DataError
from trainit_data.integrity import recover_session, verify_session


def _inject_enospc(monkeypatch, should_fail):
    real = _fsio.write_fully

    def faulty(fd, data):
        import os
        target = os.readlink(f'/proc/self/fd/{fd}')
        if should_fail(target):
            real(fd, data[:len(data) // 2])                 # a short write, then the disk is full
            raise OSError(errno.ENOSPC, 'No space left on device')
        real(fd, data)
    monkeypatch.setattr(_fsio, 'write_fully', faulty)


def test_a_payload_enospc_is_recorded_honestly_and_raised(store, vocab, monkeypatch):
    writer = open_session(store)
    at = stamp(1)
    writer.append_capture(trigger=trigger(at, 0), outcome='COMPLETE', sync=sync(), samples=rgbd(at, 0))
    _inject_enospc(monkeypatch, lambda path: path.endswith('overhead_camera__depth.png.tmp'))
    at = stamp(2)
    with pytest.raises(DataError) as err:
        writer.append_capture(trigger=trigger(at, 1), outcome='COMPLETE', sync=sync(), samples=rgbd(at, 1))
    assert err.value.code == WRITE_FAILED
    assert err.value.context['outcome'] == 'PARTIAL' and err.value.context['failed_streams'] == [f'{SENSOR}/depth']

    captures = list(store.open_session(writer.session_id).captures())
    second = captures[1]
    assert second.outcome == 'PARTIAL'                       # never a false COMPLETE
    assert [(m.stream, m.reason) for m in second.missing] == [(f'{SENSOR}/depth', 'WRITE_FAILED')]
    assert 'ENOSPC' in second.missing[0].detail
    assert [o.stream for o in second.observations] == ['color']

    monkeypatch.undo()                                       # the disk has space again: the writer carries on
    at = stamp(3)
    writer.append_capture(trigger=trigger(at, 2), outcome='COMPLETE', sync=sync(), samples=rgbd(at, 2))
    header = writer.close()
    assert header.counters['captures'] == {'attempted': 3, 'complete': 2, 'partial': 1, 'failed': 0}
    assert header.counters['write_errors'] == 1
    moved = header.warnings[-1]['context']['moved']
    assert header.warnings[-1]['code'] == WRITE_FAILED
    assert moved == [{'from': 'payload/000002/overhead_camera__depth.png.tmp',
                      'to': 'orphans/payload/000002/overhead_camera__depth.png.tmp'}]
    assert (store.session_dir(writer.session_id) / moved[0]['to']).stat().st_size > 0      # kept, not deleted
    assert verify_session(store, writer.session_id).ok


def test_all_payloads_failing_is_a_failed_capture(store, vocab, monkeypatch):
    writer = open_session(store)
    _inject_enospc(monkeypatch, lambda path: '/payload/' in path)
    at = stamp(1)
    with pytest.raises(DataError) as err:
        writer.append_capture(trigger=trigger(at), outcome='COMPLETE', sync=sync(), samples=rgbd(at))
    assert err.value.context['outcome'] == 'FAILED'
    monkeypatch.undo()
    header = writer.close()
    assert header.counters['captures']['failed'] == 1 and header.counters['write_errors'] == 2
    assert verify_session(store, writer.session_id).ok


def test_an_index_enospc_faults_the_writer_and_recover_restores_consistency(store, vocab, monkeypatch):
    writer = open_session(store)
    at = stamp(1)
    writer.append_capture(trigger=trigger(at, 0), outcome='COMPLETE', sync=sync(), samples=rgbd(at, 0))
    _inject_enospc(monkeypatch, lambda path: path.endswith('captures.jsonl'))
    at = stamp(2)
    with pytest.raises(DataError) as err:
        writer.append_capture(trigger=trigger(at, 1), outcome='COMPLETE', sync=sync(), samples=rgbd(at, 1))
    assert err.value.code == WRITE_FAILED and err.value.context['capture_id'].endswith('/000002')
    monkeypatch.undo()

    for attempt in (lambda: writer.append_capture(trigger=trigger(at, 2), outcome='COMPLETE', sync=sync(),
                                                  samples=rgbd(at)), writer.close):
        with pytest.raises(DataError) as err:
            attempt()
        assert err.value.code == SESSION_FAULTED             # never sealed CLOSED over a torn index
    session_dir = store.session_dir(writer.session_id)
    committed = list(store.open_session(writer.session_id).captures())
    assert [c.seq for c in committed] == [1]                 # no false successful capture 2
    writer.release()

    before = snapshot(session_dir)
    report = recover_session(store, writer.session_id)
    assert report['torn_tail']['bytes'] > 0
    assert {q['from'] for q in report['quarantined']} == {
        'payload/000002/overhead_camera__color.png', 'payload/000002/overhead_camera__depth.png'}
    after = snapshot(session_dir)
    for rel in ('payload/000002/overhead_camera__color.png', 'payload/000002/overhead_camera__depth.png'):
        assert after['orphans/' + rel] == before[rel]         # moved, byte for byte
    header = store.open_session(writer.session_id).header
    assert header.status == 'INTERRUPTED'
    assert header.counters['captures'] == {'attempted': 2, 'complete': 1, 'partial': 0, 'failed': 0}
    assert verify_session(store, writer.session_id).ok


def test_a_failed_seal_leaves_the_session_recoverable(store, vocab, monkeypatch):
    writer = open_session(store)
    at = stamp(1)
    writer.append_capture(trigger=trigger(at), outcome='COMPLETE', sync=sync(), samples=rgbd(at))
    _inject_enospc(monkeypatch, lambda path: path.endswith('session.json.tmp'))
    with pytest.raises(DataError) as err:
        writer.close()
    assert err.value.code == WRITE_FAILED
    monkeypatch.undo()
    assert store.open_session(writer.session_id).header.status == 'ACTIVE'     # the old header is intact
    writer.release()
    report = recover_session(store, writer.session_id)
    assert [q['from'] for q in report['quarantined']] == ['session.json.tmp']
    assert verify_session(store, writer.session_id).ok
