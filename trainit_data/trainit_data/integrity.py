"""verify and recover (PUBLIC).

``verify_session`` / ``verify_store`` INSPECT and never change a byte. They report findings with stable
codes and the exact artifact (store-relative path) and record (observation id, index line) concerned.

``recover_session`` makes an interrupted ACTIVE session consistent, non-destructively (D0_DESIGN.md §12):

1. analyse first, change nothing: every COMPLETE index line must be a valid capture of this session, in
   sequence, whose payloads exist with their recorded sha256. If not, recovery cannot establish safety and
   refuses (``RECOVERY_UNSAFE``) -- a damaged committed record is not a crash artifact and is never guessed
   about;
2. a trailing fragment without its newline (a torn write) is COPIED to ``orphans/`` and only then cut off
   the index;
3. every payload-area file no committed line references (a ``.tmp``, a payload whose line was never
   written) and a stray ``session.json.tmp`` are MOVED to ``orphans/``;
4. ``recovery.json`` is written, then the header is sealed ``INTERRUPTED`` with recomputed counters and
   the index hash, then the ``.active`` marker is removed.

A session whose live writer still holds its lock is refused (``SESSION_IN_USE``); a sealed one too
(``SESSION_SEALED``). Nothing is ever deleted.
"""
import os
import time
from dataclasses import dataclass, field

from . import __version__
from . import _codecs, _fsio
from . import _strict as s
from ._canonical import parse_json, pretty_json, sha256_file, sha256_json
from .errors import (CALIBRATION_MISMATCH, CALIBRATION_NOT_FOUND, HASH_MISMATCH, INDEX_CORRUPT, INDEX_TORN,
                     INVALID_VALUE, MALFORMED_RECORD, ORPHAN_FILE, PAYLOAD_INVALID, PAYLOAD_MISSING, RECOVERY_UNSAFE,
                     SESSION_IN_USE, SESSION_INCONSISTENT, SESSION_NOT_FOUND, SESSION_SEALED, STALE_ACTIVE_MARKER,
                     WRITE_FAILED, DataError)
from .reader import index_lines, parse_index_line, read_header
from .session_record import SessionHeader
from .synthetic import TRIGGER_KIND, session_problems

RECOVERY_SCHEMA = 'trainit-recovery/1.0'
_KNOWN_ENTRIES = {'session.json', 'captures.jsonl', 'payload', 'orphans', '.active', 'recovery.json'}


@dataclass(frozen=True)
class Finding:
    code: str
    severity: str                       # ERROR | WARNING
    artifact: str                       # store-relative path
    detail: str
    record: str = None                  # observation id / capture id / index line

    def to_dict(self):
        out = {'code': self.code, 'severity': self.severity, 'artifact': self.artifact, 'detail': self.detail}
        s.put(out, 'record', self.record)
        return out


@dataclass
class VerifyReport:
    session_id: str
    status: str = None
    findings: list = field(default_factory=list)
    captures: int = 0
    observations: int = 0
    payload_bytes: int = 0

    @property
    def ok(self) -> bool:
        return not any(f.severity == 'ERROR' for f in self.findings)

    def to_dict(self):
        return {'session_id': self.session_id, 'status': self.status, 'ok': self.ok, 'captures': self.captures,
                'observations': self.observations, 'payload_bytes': self.payload_bytes,
                'findings': [f.to_dict() for f in self.findings]}


class _Analysis:
    """One pass over a session's index and payload area, shared by verify and recover."""

    def __init__(self, store, session_id, header, vocab):
        self.store, self.session_id, self.header, self.vocab = store, session_id, header, vocab
        self.base = f'sessions/{session_id}'
        self.dir = store.session_dir(session_id)
        self.findings = []
        self.lines = 0
        self.counts = {'complete': 0, 'partial': 0, 'failed': 0}
        self.write_failed = 0
        self.calibrations = {}          # id -> first use, in order
        self.referenced = set()
        self.torn = None                # (offset, bytes) of a trailing fragment
        self.observations = 0
        self.payload_bytes = 0
        self._cal_cache = {}
        self._uncalibrated = set()
        # D0.7: the captures the synthetic-generation rules need (all of an INGEST session with a generation block,
        # else only generated frames: a live session keeps streaming)
        self._keep_all = header.source.transport == 'INGEST' and 'randomization' in header.provenance
        self.generated = []

    def add(self, code, severity, artifact, detail, record=None):
        self.findings.append(Finding(code, severity, artifact, detail, record))

    def run(self):
        index_rel = f'{self.base}/captures.jsonl'
        if not (self.dir / 'captures.jsonl').is_file():
            self.add(INDEX_CORRUPT, 'ERROR', index_rel, 'the index file is missing')
            return self
        for number, offset, raw, complete in index_lines(self.dir / 'captures.jsonl'):
            where = f'{index_rel}:{number}'
            if not complete:
                self.torn = (offset, raw)
                self.add(INDEX_TORN, 'ERROR' if self.header.sealed else 'WARNING', index_rel,
                         f'the index ends with an incomplete line of {len(raw)} bytes at offset {offset}', where)
                break
            self.lines += 1
            try:
                capture = parse_index_line(raw, self.vocab, where)
            except DataError as exc:
                self.add(INDEX_CORRUPT, 'ERROR', index_rel, exc.message, where)
                continue
            self._check_capture(capture, number, index_rel, where)
        self._scan_payload_area()
        return self

    def _check_capture(self, capture, number, index_rel, where):
        if capture.session_id != self.session_id or capture.seq != number:
            self.add(INDEX_CORRUPT, 'ERROR', index_rel, f'line {number} holds capture {capture.capture_id} '
                     f'(expected {self.session_id}/{number:06d})', where)
        for problem in self.header.capture_problems(capture):
            self.add(SESSION_INCONSISTENT, 'ERROR', index_rel, problem, where)
        self.counts[capture.outcome.lower()] += 1
        if self._keep_all or capture.trigger.kind == TRIGGER_KIND:
            self.generated.append(capture)
        self.write_failed += sum(1 for m in capture.missing if m.reason == 'WRITE_FAILED')
        for obs in capture.observations:
            self.observations += 1
            if obs.payload is not None:
                self._check_payload(obs, capture.seq)
            if getattr(obs.data, 'calibration_id', None) is not None:
                self._check_calibration(obs)

    def _check_payload(self, obs, seq):
        ref = obs.payload
        extension = _codecs.CODECS.get(ref.media_type, ('?',))[0]
        expected = f'{self.base}/payload/{seq:06d}/{obs.sensor_id}__{obs.stream}{extension}'
        if ref.path != expected:
            self.add(INDEX_CORRUPT, 'ERROR', ref.path, f'payload path is not the deterministic {expected}',
                     obs.observation_id)
        self.referenced.add(ref.path)
        path = self.store.path(ref.path)
        if path.is_symlink():
            self.add(PAYLOAD_INVALID, 'ERROR', ref.path, 'a payload is a regular file of the store, never a symlink',
                     obs.observation_id)
            return
        if not path.is_file():
            self.add(PAYLOAD_MISSING, 'ERROR', ref.path, 'the referenced payload does not exist', obs.observation_id)
            return
        actual, size = sha256_file(path)
        self.payload_bytes += size
        if actual != ref.sha256 or size != ref.bytes:
            self.add(HASH_MISMATCH, 'ERROR', ref.path, f'recorded {ref.sha256} ({ref.bytes} B), file is {actual} '
                     f'({size} B)', obs.observation_id)
        elif getattr(obs.data, 'block', None) == 'instance_image':
            # an instance mask is exactly what its block says (ids, pixel counts, box extents)
            try:
                mask = _codecs.decode(path.read_bytes(), ref.media_type, where=ref.path)
                problems = obs.data.mask_problems(mask)
            except DataError as exc:
                problems = [exc.message]
            for problem in problems:
                self.add(PAYLOAD_INVALID, 'ERROR', ref.path, problem, obs.observation_id)

    def _check_calibration(self, obs):
        cal_id = obs.data.calibration_id
        self.calibrations.setdefault(cal_id, obs.observation_id)
        rel = self.store.calibration_rel(cal_id)
        if cal_id not in self._cal_cache:
            try:
                self._cal_cache[cal_id] = self.store.get_calibration(cal_id, self.vocab)
            except DataError as exc:
                self._cal_cache[cal_id] = exc
        record = self._cal_cache[cal_id]
        if isinstance(record, DataError):
            code = record.code if record.code in (CALIBRATION_NOT_FOUND, HASH_MISMATCH) else MALFORMED_RECORD
            self.add(code, 'ERROR', rel, record.message, obs.observation_id)
            return
        if not record.intrinsics.calibrated and cal_id not in self._uncalibrated:
            self._uncalibrated.add(cal_id)
            self.add(INVALID_VALUE, 'WARNING', rel, f'calibration {cal_id} holds an uncalibrated camera matrix K '
                     f'{list(record.intrinsics.K)} (written before trainit_data 0.9.0, which refuses one): readable, '
                     'but no projection can use it', obs.observation_id)
        if ((record.sensor_id, record.stream) != (obs.sensor_id, obs.stream)
                or (record.intrinsics.width, record.intrinsics.height) != (obs.data.width, obs.data.height)
                or record.optical_frame != obs.data.frame_id):
            self.add(CALIBRATION_MISMATCH, 'ERROR', rel, f'calibration {cal_id} does not describe this observation '
                     '(sensor / stream / size / optical frame)', obs.observation_id)

    def _scan_payload_area(self):
        payload_dir = self.dir / 'payload'
        severity = 'ERROR' if self.header.sealed else 'WARNING'
        self.orphans = []
        self.last_seq_dir = max((int(p.name) for p in payload_dir.iterdir() if p.is_dir() and p.name.isdigit()),
                                default=0) if payload_dir.is_dir() else 0
        if payload_dir.is_dir():
            for path in sorted(p for p in payload_dir.rglob('*') if p.is_file() or p.is_symlink()):
                rel = f'{self.base}/{path.relative_to(self.dir).as_posix()}'
                if rel not in self.referenced:
                    self.orphans.append(path.relative_to(self.dir).as_posix())
                    self.add(ORPHAN_FILE, severity, rel, 'no committed capture references this file')
        for entry in sorted(p.name for p in self.dir.iterdir()):
            if entry not in _KNOWN_ENTRIES:
                self.add(ORPHAN_FILE, severity if entry != 'session.json.tmp' else 'WARNING', f'{self.base}/{entry}',
                         'unexpected entry in the session directory')


def _recovery_record(value, where):
    f = s.Fields(value, where)
    f.req('schema', s.schema('trainit-recovery', 1))
    for name in ('session_id', 'recovered_wall_utc_ns', 'trainit_data', 'committed_captures', 'quarantined', 'index'):
        f.req(name, lambda v, w: v)
    f.opt('torn_tail', s.opaque)
    f.opt('earlier_torn_tails', s.list_of(s.string))
    f.done()


def verify_session(store, session_id: str) -> VerifyReport:
    """Inspect one session; never modifies anything."""
    report = VerifyReport(session_id)
    base = f'sessions/{session_id}'
    session_dir = store.session_dir(session_id)
    if not (session_dir / 'session.json').is_file():
        raise DataError(SESSION_NOT_FOUND, f'no session {session_id} in {store.root}',
                        context={'session_id': session_id})
    try:
        header, vocab = read_header(store, session_id)
    except DataError as exc:
        report.findings.append(Finding(exc.code, 'ERROR', f'{base}/session.json', exc.message))
        return report
    report.status = header.status
    analysis = _Analysis(store, session_id, header, vocab).run()
    report.findings.extend(analysis.findings)
    for problem in session_problems(header, analysis.generated):
        report.findings.append(Finding(SESSION_INCONSISTENT, 'ERROR', f'{base}/captures.jsonl', problem))
    report.captures, report.observations = analysis.lines, analysis.observations
    report.payload_bytes = analysis.payload_bytes
    if header.sealed:
        _check_sealed(store, header, analysis, report)
    return report


def _check_sealed(store, header: SessionHeader, analysis: '_Analysis', report: VerifyReport):
    base = analysis.base
    index_path = analysis.dir / 'captures.jsonl'
    if index_path.is_file():
        actual, _ = sha256_file(index_path)
        if actual != header.index['sha256']:
            report.findings.append(Finding(HASH_MISMATCH, 'ERROR', f'{base}/captures.jsonl',
                                           f'sealed index sha256 {header.index["sha256"]}, file is {actual}'))
    if header.index['lines'] != analysis.lines:
        report.findings.append(Finding(SESSION_INCONSISTENT, 'ERROR', f'{base}/session.json',
                                       f'header says {header.index["lines"]} index lines, the index has '
                                       f'{analysis.lines}'))
    recorded = header.counters['captures']
    for outcome, count in analysis.counts.items():
        if recorded[outcome] != count:
            report.findings.append(Finding(SESSION_INCONSISTENT, 'ERROR', f'{base}/session.json',
                                           f'counters.captures.{outcome} = {recorded[outcome]}, the index has {count}'))
    if recorded['attempted'] < analysis.lines or header.counters['write_errors'] < analysis.write_failed:
        report.findings.append(Finding(SESSION_INCONSISTENT, 'ERROR', f'{base}/session.json',
                                       'attempted / write_errors are below what the index records'))
    if list(header.calibration_ids) != list(analysis.calibrations):
        report.findings.append(Finding(SESSION_INCONSISTENT, 'ERROR', f'{base}/session.json',
                                       f'calibration_ids {list(header.calibration_ids)} != used '
                                       f'{list(analysis.calibrations)}'))
    if (analysis.dir / '.active').exists():
        report.findings.append(Finding(STALE_ACTIVE_MARKER, 'WARNING', f'{base}/.active',
                                       'the session is sealed but still carries its .active marker'))
    if header.status == 'INTERRUPTED':
        rel = f'{base}/recovery.json'
        try:
            _recovery_record(parse_json(store.path(rel).read_bytes(), rel), rel)
        except (DataError, OSError) as exc:
            report.findings.append(Finding(MALFORMED_RECORD, 'ERROR', rel, f'recovery report unreadable: {exc}'))


@dataclass
class StoreReport:
    root: str
    store_id: str
    sessions: list = field(default_factory=list)
    findings: list = field(default_factory=list)
    datasets: list = field(default_factory=list)        # D0.6: DatasetReport per dataset

    @property
    def ok(self) -> bool:
        return (all(r.ok for r in self.sessions) and all(r.ok for r in self.datasets)
                and not any(f.severity == 'ERROR' for f in self.findings))

    def to_dict(self):
        return {'root': self.root, 'store_id': self.store_id, 'ok': self.ok,
                'sessions': [r.to_dict() for r in self.sessions], 'datasets': [r.to_dict() for r in self.datasets],
                'findings': [f.to_dict() for f in self.findings]}


def verify_store(store) -> StoreReport:
    """Inspect every calibration record, every session and every dataset of the store; never modifies anything."""
    report = StoreReport(str(store.root), store.store_id)
    for path in sorted((store.root / 'calibrations').glob('*.json')):
        rel = f'calibrations/{path.name}'
        try:
            raw = parse_json(path.read_bytes(), rel)
            content = {k: v for k, v in raw.items() if k not in ('calibration_id', 'sha256')}
            sha = sha256_json(content)
            if raw.get('sha256') != sha or raw.get('calibration_id') != 'cal-' + sha[7:23] \
                    or path.name != f'{raw.get("calibration_id")}.json':
                report.findings.append(Finding(HASH_MISMATCH, 'ERROR', rel, 'the file does not hash to its id'))
        except (DataError, AttributeError) as exc:
            report.findings.append(Finding(MALFORMED_RECORD, 'ERROR', rel, str(exc)))
    for session_id in store.list_sessions():
        if not (store.session_dir(session_id) / 'session.json').is_file():
            # The writer writes the header last: no header = an interrupted creation, nothing committed.
            report.findings.append(Finding(SESSION_NOT_FOUND, 'WARNING', f'sessions/{session_id}',
                                           'a session directory without session.json (its creation was interrupted '
                                           'before anything was committed); left untouched'))
            continue
        try:
            report.sessions.append(verify_session(store, session_id))
        except DataError as exc:
            report.findings.append(Finding(exc.code, 'ERROR', f'sessions/{session_id}', exc.message))
    from .dataset import list_datasets                  # D0.6; imported here: the dataset modules import this one
    from .dataset_integrity import verify_dataset
    verified = {r.session_id: r.ok for r in report.sessions}
    for dataset_id in list_datasets(store):
        try:
            report.datasets.append(verify_dataset(store, dataset_id, verified))
        except DataError as exc:
            report.findings.append(Finding(exc.code, 'ERROR', f'datasets/{dataset_id}', exc.message))
    return report


def recover_session(store, session_id: str) -> dict:
    """Seal an interrupted ACTIVE session INTERRUPTED, non-destructively. Returns the recovery report."""
    session_dir = store.session_dir(session_id)
    if not (session_dir / 'session.json').is_file():
        raise DataError(SESSION_NOT_FOUND, f'no session header for {session_id}; nothing was committed, nothing is '
                        'changed', context={'session_id': session_id})
    lock, created = _lock_for_recovery(session_dir / '.active', session_id)
    try:
        # Read the header only UNDER the lock: a writer that sealed meanwhile is seen as sealed.
        try:
            header, vocab = read_header(store, session_id)
        except DataError as exc:
            raise DataError(RECOVERY_UNSAFE, f'session {session_id}: the header is unreadable ({exc.code}); nothing '
                            'is changed', context={'session_id': session_id, 'detail': exc.to_dict()}) from None
        if header.sealed:
            raise DataError(SESSION_SEALED, f'session {session_id} is already sealed {header.status}',
                            context={'session_id': session_id, 'status': header.status})
        report = _recover_locked(store, session_id, header, vocab)
    except BaseException:
        lock.release(remove=created)    # a marker this call created on a session it did not recover goes again
        raise
    lock.release(remove=True)
    return report


def _lock_for_recovery(marker, session_id):
    """(lock, created): the session's `.active` lock, taken without waiting; ``SESSION_IN_USE`` if held."""
    try:
        try:
            return _fsio.SessionLock(marker, create=False), False
        except FileNotFoundError:
            return _fsio.SessionLock(marker, create=True), True
    except (BlockingIOError, FileExistsError):
        raise DataError(SESSION_IN_USE, f'session {session_id} is held by a live writer (or another recover); '
                        'nothing is changed', context={'session_id': session_id}) from None


def _recover_locked(store, session_id, header, vocab):
    analysis = _Analysis(store, session_id, header, vocab).run()
    blocking = [f for f in analysis.findings if f.severity == 'ERROR']
    if blocking:
        raise DataError(RECOVERY_UNSAFE, f'session {session_id}: committed records are damaged '
                        f'({len(blocking)} finding(s)); recovery would have to guess, nothing is changed',
                        context={'session_id': session_id, 'findings': [f.to_dict() for f in blocking]})
    session_dir = analysis.dir
    stray = [p.name for p in session_dir.iterdir() if p.name not in _KNOWN_ENTRIES]
    unsafe = [name for name in stray if name != 'session.json.tmp']
    if unsafe:
        raise DataError(RECOVERY_UNSAFE, f'session {session_id}: unexpected entries {unsafe}; nothing is changed',
                        context={'session_id': session_id, 'entries': unsafe})
    orphans_dir = session_dir / 'orphans'
    earlier_tails = sorted(p.name for p in orphans_dir.glob('captures.jsonl.torn*')) if orphans_dir.is_dir() else []
    try:
        torn_tail = None
        if analysis.torn is not None:
            offset, raw = analysis.torn
            saved = _save_torn_tail(session_dir, raw)                 # a durable copy first ...
            os.truncate(session_dir / 'captures.jsonl', offset)        # ... then cut the fragment off the index
            torn_tail = {'offset': offset, 'bytes': len(raw), 'saved_as': saved}
        _fsync_file(session_dir / 'captures.jsonl')                     # hash what is on disk, not in the cache
        quarantined = []
        leftovers = [n for n in stray if n == 'session.json.tmp']
        if (session_dir / 'recovery.json').exists():
            leftovers.append('recovery.json')       # an earlier recover attempt that did not finish
        for rel in analysis.orphans + leftovers:
            size = os.lstat(session_dir / rel).st_size
            quarantined.append({'from': rel, 'to': _fsio.quarantine(session_dir, rel), 'bytes': size})
        index_sha, _ = sha256_file(session_dir / 'captures.jsonl')
        now = time.time_ns()
        recovery = {'schema': RECOVERY_SCHEMA, 'session_id': session_id, 'recovered_wall_utc_ns': now,
                    'trainit_data': __version__, 'committed_captures': analysis.lines, 'quarantined': quarantined,
                    'index': {'sha256': index_sha, 'lines': analysis.lines}}
        s.put(recovery, 'torn_tail', torn_tail)
        if earlier_tails:
            recovery['earlier_torn_tails'] = [f'orphans/{name}' for name in earlier_tails]
        _fsio.write_new_file(session_dir / 'recovery.json', pretty_json(recovery))
        document = header.to_dict()
        # Only what the files prove: attempts = the highest seq that left a trace (an index line, a torn
        # line, a payload directory). Overruns / calibration changes are unknown after a crash: omitted.
        attempted = max(analysis.lines + (1 if torn_tail or earlier_tails else 0), analysis.last_seq_dir)
        document.update(
            status='INTERRUPTED', closed={'wall_utc_ns': now, 'by': 'recover'},
            calibration_ids=list(analysis.calibrations),
            counters={'captures': dict(analysis.counts, attempted=attempted), 'write_errors': analysis.write_failed},
            index={'path': f'sessions/{session_id}/captures.jsonl', 'sha256': index_sha, 'lines': analysis.lines})
        document['warnings'] = list(header.warnings) + [{
            'code': 'SESSION_INTERRUPTED', 'detail': 'the writer stopped without sealing; sealed by recover, see '
            'recovery.json (counters recomputed from the files)'}]
        sealed = SessionHeader.from_dict(document, vocab)
        _fsio.replace_file(session_dir / 'session.json', pretty_json(sealed.to_dict()))
    except OSError as exc:
        raise DataError(WRITE_FAILED, f'session {session_id}: recovery could not write; the session stays ACTIVE '
                        'and recover can be re-run', context={'session_id': session_id}, cause=exc) from None
    return recovery


def _fsync_file(path):
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _save_torn_tail(session_dir, raw: bytes) -> str:
    orphans = session_dir / 'orphans'
    _fsio.makedirs_durable(orphans)
    name, suffix = 'captures.jsonl.torn', 0
    while (orphans / name).exists():
        suffix += 1
        name = f'captures.jsonl.torn.{suffix}'
    _fsio.write_new_file(orphans / name, raw)
    return f'orphans/{name}'


__all__ = ['verify_session', 'verify_store', 'recover_session', 'VerifyReport', 'StoreReport', 'Finding']
