"""WriterWorker: every draft resolved exactly once; written + overruns + lost == submitted; the thread never dies."""
import errno
import time

from _ros_helpers import depth_mm, rgb
from test_coordinator import B, COLOR, DEPTH, feed_infos, make, run, sample, trigger
from trainit_data.errors import WRITE_FAILED, DataError
from trainit_data.store import Store
from trainit_data_ros.writer_worker import STOPPED, WriterWorker


def _session(tmp_path, cfg):
    store = Store.open(tmp_path / 'TrainItData', create=True)
    sensors = [{'sensor_id': s['sensor_id'],
                'streams': [{'stream': n, 'modality': b['modality']} for n, b in s['streams'].items()]}
               for s in cfg['sensors']]
    writer = store.create_session(source={'transport': 'LIVE_ROS', 'backend': 'mock',
                                          'descriptor': cfg['source']['descriptor']}, clock=cfg['clock'],
                                  sensors=sensors)
    return store, writer


def _draft(c, clock, i=0):
    feed_infos(c)
    c.fire(trigger(B + i, i), B + i)
    c.on_sample(COLOR, sample(c, clock, B + i + 1, rgb(i), 'rgb8'))
    c.on_sample(DEPTH, sample(c, clock, B + i + 1, depth_mm(i), '16UC1'))
    return run(c, clock)


def test_an_unexpected_error_loses_one_capture_and_the_thread_survives(tmp_path, monkeypatch):
    c, clock, cfg = make()
    store, writer = _session(tmp_path, cfg)
    outcomes = []
    worker = WriterWorker(store, writer, 4, on_written=lambda d, o: outcomes.append(o))
    real = writer.append_capture
    calls = {'n': 0}

    def flaky(**kwargs):
        calls['n'] += 1
        if calls['n'] == 1:
            raise RuntimeError('codec exploded')
        return real(**kwargs)
    monkeypatch.setattr(writer, 'append_capture', flaky)
    first, second = _draft(c, clock, 0), _draft(c, clock, 1)
    assert worker.submit(first) and worker.submit(second)
    assert first.pending.done.wait(10) and second.pending.done.wait(10)
    assert isinstance(first.pending.result, DataError) and first.pending.result.code == WRITE_FAILED
    assert second.pending.result.outcome == 'COMPLETE' and outcomes == ['COMPLETE']
    assert (worker.written, worker.lost, worker.overruns) == (1, 1, 0)
    assert worker.drain(5) and worker.stop()
    writer.close()


def test_a_calibration_that_cannot_be_written_is_lost_not_silent(tmp_path, monkeypatch):
    c, clock, cfg = make()
    store, writer = _session(tmp_path, cfg)
    worker = WriterWorker(store, writer, 4)

    def enospc(record):
        raise OSError(errno.ENOSPC, 'No space left on device')
    monkeypatch.setattr(store, 'put_calibration', enospc)
    draft = _draft(c, clock)
    worker.submit(draft)
    assert draft.pending.done.wait(10)
    assert draft.pending.result.code == WRITE_FAILED and worker.lost == 1 and worker.written == 0
    worker.stop()
    writer.close()


def test_nothing_is_queued_after_the_session_starts_closing(tmp_path):
    c, clock, cfg = make()
    store, writer = _session(tmp_path, cfg)
    worker = WriterWorker(store, writer, 4)
    assert worker.drain(5)
    late = _draft(c, clock)
    assert not worker.submit(late) and late.pending.result == STOPPED and worker.overruns == 1
    assert worker.stop()
    writer.close()


def test_stop_never_blocks_forever_on_a_full_queue_with_a_hung_writer(tmp_path, monkeypatch):
    c, clock, cfg = make()
    store, writer = _session(tmp_path, cfg)
    monkeypatch.setattr(writer, 'append_capture', lambda **kw: time.sleep(3))
    worker = WriterWorker(store, writer, 1)
    for i in range(3):
        worker.submit(_draft(c, clock, i))
    started = time.monotonic()
    assert worker.stop(timeout_s=0.5) is False and time.monotonic() - started < 2.0
