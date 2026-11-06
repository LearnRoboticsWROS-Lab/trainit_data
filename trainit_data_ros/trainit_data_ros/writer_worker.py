"""The writer worker: the ONLY thread that touches the data store during a session (no ROS imports).

Encoding PNG / .npy and fsyncing never happen in a ROS callback. Assembled captures go through a
BOUNDED queue; when it is full the capture is DROPPED -- never written, never reported as written -- and
counted as a writer overrun (persisted in the session counters at close, and in the diagnostics). One
encoded payload is in memory at a time (trainit_data's SessionWriter streams).

Every submitted draft is resolved exactly once (``pending.done`` set, ``pending.result`` = the committed
Capture, a DataError, or a ``DROPPED`` / ``LOST`` text), whatever happens -- the thread never dies -- so
``written + overruns + lost == submitted`` always holds and the session can say so at close.
"""
import queue
import threading
import time

from trainit_data.errors import SESSION_FAULTED, WRITE_FAILED, DataError

DROPPED = 'DROPPED: writer queue full (QUEUE_OVERRUN)'
STOPPED = 'DROPPED: the session was closing (QUEUE_OVERRUN)'


class WriterWorker:
    def __init__(self, store, writer, queue_size: int, on_written=None):
        self.store = store
        self.writer = writer
        self.on_written = on_written or (lambda draft, outcome: None)   # outcome AS COMMITTED
        self.queue = queue.Queue(maxsize=queue_size)
        self.overruns = 0               # drafts never queued (full queue, or the session closing)
        self.lost = 0                   # drafts queued but not indexed (refused / failed before the index line)
        self.write_failures = 0         # WRITE_FAILED (a payload or the index): the capture may still be indexed
        self.high_water = 0
        self.written = 0
        self.latency_ms = {'count': 0, 'sum': 0.0, 'max': 0.0, 'last': None}
        self.faulted = None             # the DataError that faulted the session, if any
        self.last_loss = None
        self._accepting = True
        self._lock = threading.Lock()
        self._thread = threading.Thread(target=self._run, name='trainit_data_writer', daemon=True)
        self._thread.start()

    def _resolve(self, draft, result):
        draft.pending.result = result
        draft.pending.done.set()

    def submit(self, draft) -> bool:
        """Queue an assembled capture; False (the overrun counted, the draft resolved DROPPED) if it cannot be."""
        with self._lock:
            if not self._accepting:
                self.overruns += 1
                self._resolve(draft, STOPPED)
                return False
            try:
                self.queue.put_nowait(draft)
            except queue.Full:
                self.overruns += 1
                self._resolve(draft, DROPPED)
                return False
            self.high_water = max(self.high_water, self.queue.qsize())
            return True

    def _run(self):
        while True:
            draft = self.queue.get()
            try:
                if draft is None:
                    return
                result = None
                try:
                    result = self._write(draft)
                finally:
                    self._resolve(draft, result if result is not None else 'LOST: the writer failed unexpectedly')
            finally:
                self.queue.task_done()

    def _write(self, draft):
        if self.faulted is not None:
            self._lose(self.faulted)
            return self.faulted
        started = time.monotonic()
        try:
            for record in draft.calibrations:
                self.store.put_calibration(record)
            capture = self.writer.append_capture(trigger=draft.pending.trigger, outcome=draft.outcome,
                                                 samples=draft.samples, missing=draft.missing, sync=draft.sync)
        except DataError as exc:
            committed = exc.code == WRITE_FAILED and 'capture_id' in exc.context and self.writer.state != 'FAULTED'
            with self._lock:
                if exc.code == WRITE_FAILED:
                    self.write_failures += 1
                if exc.code == SESSION_FAULTED or self.writer.state == 'FAULTED':
                    self.faulted = exc
            if committed:
                # The capture WAS indexed (honestly downgraded, a payload WRITE_FAILED): count it as written.
                self._count_written(draft, exc.context.get('outcome', draft.outcome), started)
            else:
                self._lose(exc)
            return exc
        except Exception as exc:  # noqa: BLE001 -- an OSError of a calibration file, or anything unexpected
            error = DataError(WRITE_FAILED, f'the capture could not be written: {exc!r}', cause=exc)
            with self._lock:
                self.write_failures += 1
                if self.writer.state == 'FAULTED':
                    self.faulted = error
            self._lose(error)
            return error
        self._count_written(draft, capture.outcome, started)
        return capture

    def _count_written(self, draft, outcome, started):
        elapsed = (time.monotonic() - started) * 1000.0
        with self._lock:
            self.written += 1
            stats = self.latency_ms
            stats['count'] += 1
            stats['sum'] += elapsed
            stats['max'] = max(stats['max'], elapsed)
            stats['last'] = elapsed
        try:
            self.on_written(draft, outcome)
        except Exception:  # noqa: BLE001 -- accounting must never undo a committed write
            pass

    def _lose(self, error):
        with self._lock:
            self.lost += 1
            self.last_loss = error

    def stats(self) -> dict:
        with self._lock:
            stats = self.latency_ms
            mean = stats['sum'] / stats['count'] if stats['count'] else None
            return {'queue_depth': self.queue.qsize(), 'queue_high_water': self.high_water,
                    'queue_size': self.queue.maxsize, 'overruns': self.overruns, 'lost': self.lost,
                    'write_failures': self.write_failures, 'written': self.written,
                    'write_latency_ms': {'mean': mean, 'max': stats['max'] or None, 'last': stats['last']},
                    'faulted': self.faulted is not None}

    def drain(self, timeout_s: float) -> bool:
        """Stop accepting, then wait until every queued capture is resolved (or the timeout passes)."""
        with self._lock:
            self._accepting = False
        deadline = time.monotonic() + timeout_s
        while self.queue.unfinished_tasks and time.monotonic() < deadline:
            time.sleep(0.01)
        return not self.queue.unfinished_tasks

    def stop(self, timeout_s: float = 30.0) -> bool:
        """End the thread after the queued work; never blocks forever (returns False if it did not end)."""
        with self._lock:
            self._accepting = False
        try:
            self.queue.put(None, timeout=timeout_s)
        except queue.Full:
            return False
        self._thread.join(timeout=timeout_s)
        return not self._thread.is_alive()
