"""SessionWriter: append-only, crash-safe persistence of one session (PUBLIC).

Session directory (store-relative ``sessions/<session_id>/``)::

    session.json                         the header: ACTIVE at creation, rewritten once when sealed
    captures.jsonl                       the index: one canonical Capture per line, append-only
    payload/<seq:06d>/<sensor>__<stream>.png|.npy
    .active                              present while ACTIVE; the live writer holds an exclusive lock on it
    orphans/                             quarantine: bytes no committed capture references (never deleted)
    recovery.json                        written by ``recover`` only

The ordering of one ``append_capture`` (D0_DESIGN.md §12):

1. validate EVERYTHING (vocabulary, declared streams, encodings, calibrations, the capture rules) with
   no I/O, so a refused call leaves no trace;
2. per payload: encode -> ``<name>.tmp`` -> fsync -> rename -> fsync the directory;
3. append the canonical index line and fsync it.

An index line therefore never points at a payload that was not committed. Write failures (ENOSPC, EIO):

- a payload that cannot be written becomes a missing stream with reason ``WRITE_FAILED``. The capture is
  recorded honestly as PARTIAL or FAILED, the counters say so, and ``WRITE_FAILED`` is raised AFTER the
  record is committed. The writer stays usable; the failed ``.tmp`` is quarantined into ``orphans/``
  when the session is closed;
- if the INDEX line cannot be written, the writer is FAULTED: it refuses further appends and refuses to
  seal (``SESSION_FAULTED``), and raises ``WRITE_FAILED``. ``trainit-data recover`` then makes the session
  consistent (a torn tail truncated and kept, the unindexed payloads moved to ``orphans/``) and seals it
  INTERRUPTED.

Nothing is ever deleted by the writer. One encoded payload is in memory at a time; an append costs O(1)
(the index hash is kept incrementally), so the writer streams captures of any session length.
"""
import errno
import hashlib
import os
import secrets
import time
from dataclasses import dataclass, field

import numpy as np

from . import __version__
from . import _codecs, _fsio
from . import _strict as s
from ._canonical import canonical_json, pretty_json, sha256_bytes, sha256_json
from .capture import SCHEMA as CAPTURE_SCHEMA
from .capture import Capture, capture_id_for
from .errors import (CALIBRATION_MISMATCH, ENCODING_REFUSED, INVALID_VALUE, SESSION_FAULTED, SESSION_SEALED,
                     WRITE_FAILED, DataError)
from .observation import InstanceImageBlock, PoseAtStamp, Quality, Receipt, modality_problem
from .session_record import SCHEMA as SESSION_SCHEMA
from .session_record import SessionHeader
from .timestamps import Timestamp
from .vocab import Vocabulary

_PLACEHOLDER_SHA = 'sha256:' + '0' * 64


@dataclass(frozen=True)
class ImageSample:
    """One image (rgb / depth) to persist as an observation of a declared stream. Nothing is converted:
    ``array`` must be exactly ``encoding`` (dtype + channels), and depth declares ``unit_scale_m``."""
    sensor_id: str
    stream: str
    stamp: Timestamp
    array: np.ndarray
    encoding: str
    frame_id: str
    unit_scale_m: float = None
    calibration_id: str = None
    media_type: str = None              # default: the first payload of the modality row that fits the array
    receipt: Receipt = None
    device_stamp: Timestamp = None
    quality: Quality = None
    pose_at_stamp: PoseAtStamp = None
    transforms: tuple = ()
    extensions: dict = field(default_factory=dict)
    block_fields: dict = None           # the fields a non-``image`` data block adds (``instance_image``: instances)

    @property
    def stream_key(self) -> str:
        return f'{self.sensor_id}/{self.stream}'


@dataclass(frozen=True)
class VectorSample:
    """One low-rate non-image reading (e.g. an IMU sample) for a stream whose modality uses the ``vector`` data
    block (test D0-J). Its values are written into the index line itself: no payload file."""
    sensor_id: str
    stream: str
    stamp: Timestamp
    frame_id: str
    values: tuple
    units: tuple
    receipt: Receipt = None
    device_stamp: Timestamp = None
    quality: Quality = None
    extensions: dict = field(default_factory=dict)

    @property
    def stream_key(self) -> str:
        return f'{self.sensor_id}/{self.stream}'


def _as_dict(value):
    return value.to_dict() if hasattr(value, 'to_dict') else value


def new_session_id(wall_ns: int = None) -> str:
    wall_ns = time.time_ns() if wall_ns is None else wall_ns
    stamp = time.strftime('%Y%m%dT%H%M%SZ', time.gmtime(wall_ns // 1_000_000_000))
    return f'ses-{stamp}-{secrets.token_hex(4)}'


def _os_detail(exc: OSError) -> str:
    name = errno.errorcode.get(exc.errno, 'EIO') if exc.errno else type(exc).__name__
    return f'{name}: {exc.strerror or exc}'


class SessionWriter:
    """The only writer of an ACTIVE session. Create it with ``Store.create_session(...)``."""

    def __init__(self, store, header: SessionHeader, vocab: Vocabulary, lock, index):
        self.store = store
        self.header = header
        self.vocab = vocab
        self._lock = lock
        self._index = index
        self._dir = store.session_dir(header.session_id)
        self._modalities = header.stream_modalities()
        self._index_hash = hashlib.sha256()
        self._lines = 0
        self._counts = {'attempted': 0, 'complete': 0, 'partial': 0, 'failed': 0}
        self._write_errors = 0
        self._calibrations = {}         # id -> record, in first-use order
        self._strays = []               # session-relative paths of bytes no committed capture references
        self._pending_calibrations = {}
        self.state = 'ACTIVE'           # ACTIVE | FAULTED | RELEASED | SEALED

    @property
    def session_id(self) -> str:
        return self.header.session_id

    # ---- creation ------------------------------------------------------------------------------------

    @classmethod
    def create(cls, store, *, source, clock, sensors, rights=None, provenance=None, recording_config=None,
               quality=None, warnings=(), extensions=None, vocabulary: Vocabulary = None) -> 'SessionWriter':
        """Open a new ACTIVE session.

        ``source`` / ``clock`` / ``sensors``: :class:`~trainit_data.session_record.Source`, ``Clock``,
        ``SensorDecl`` (or their dict forms). ``recording_config``: ``{'resolved': {...}, 'file_sha256'?}``;
        its content hash is computed here. ``vocabulary`` defaults to the shipped one and is pinned in the
        store: it is authoritative for this session.
        """
        vocab = vocabulary or Vocabulary.default()
        opened = time.time_ns()
        session_id = new_session_id(opened)
        provenance = {k: dict(v) for k, v in (provenance or {}).items()}
        provenance.setdefault('software', {})['trainit_data'] = __version__
        document = {
            'schema': SESSION_SCHEMA, 'session_id': session_id, 'status': 'ACTIVE',
            'opened': {'wall_utc_ns': opened}, 'source': _as_dict(source),
            'vocab': {'schema': vocab.schema, 'sha256': vocab.sha256}, 'clock': _as_dict(clock),
            'sensors': [_as_dict(d) for d in sensors], 'rights': dict(rights or {'classification': 'INTERNAL'}),
            'calibration_ids': [], 'provenance': provenance,
        }
        if recording_config is not None:
            config = dict(recording_config)
            config['resolved_sha256'] = sha256_json(config.get('resolved'))
            document['recording_config'] = config
        if quality is not None:
            document['quality'] = quality
        if warnings:
            document['warnings'] = list(warnings)
        if extensions:
            document['extensions'] = extensions
        header = SessionHeader.from_dict(document, vocab)          # validated before anything is written
        for sensor in header.sensors:                              # every declared stream must be recordable
            for decl in sensor.streams:
                problem = modality_problem(vocab, decl.modality)
                if problem is not None:
                    raise DataError(problem[0], f'stream {sensor.sensor_id}/{decl.stream} (modality '
                                    f'{decl.modality!r}): {problem[1]}', context={'modality': decl.modality})

        # The header is written LAST: it is the commit point. A crash before it leaves a directory with no
        # session.json (nothing committed, reported by verify as an incomplete creation); after it, the session
        # has its index and payload area and `recover` can always seal it.
        session_dir = store.session_dir(session_id)
        lock = None
        try:
            store.pin_vocabulary(vocab)
            session_dir.mkdir()
            _fsio.fsync_dir(session_dir.parent)
            lock = _fsio.SessionLock(session_dir / '.active', create=True)
            _fsio.write_new_file(session_dir / 'captures.jsonl', b'')
            (session_dir / 'payload').mkdir()
            _fsio.fsync_dir(session_dir)
            _fsio.write_new_file(session_dir / 'session.json', pretty_json(header.to_dict()))
            index = _fsio.AppendLog(session_dir / 'captures.jsonl')
        except OSError as exc:
            if lock is not None:
                lock.release(remove=False)
            raise DataError(WRITE_FAILED, f'session {session_id} could not be created', cause=exc,
                            context={'session_id': session_id}) from None
        return cls(store, header, vocab, lock, index)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        if exc_type is None and self.state == 'ACTIVE':
            self.close()
        else:
            self.release()              # left ACTIVE: `trainit-data recover` seals it INTERRUPTED

    # ---- appending -----------------------------------------------------------------------------------

    def _require_active(self):
        if self.state == 'SEALED':
            raise DataError(SESSION_SEALED, f'session {self.session_id} is sealed; it is immutable',
                            context={'session_id': self.session_id})
        if self.state in ('FAULTED', 'RELEASED'):
            why = 'lost a write' if self.state == 'FAULTED' else 'was released by its writer without sealing'
            raise DataError(SESSION_FAULTED, f'session {self.session_id} {why}; run '
                            f'`trainit-data recover {self.session_id}`', context={'session_id': self.session_id})

    def _plan_sample(self, sample, capture_id: str, seq: int, where: str) -> dict:
        """Validate one ImageSample or VectorSample (no I/O): its observation dict (an image with a placeholder payload
        hash)."""
        modality = self._modalities.get(sample.stream_key)
        if modality is None:
            s.fail(INVALID_VALUE, f'stream {sample.stream_key!r} is not declared by this session', where)
        row = self.vocab.modalities[modality]
        if isinstance(sample, VectorSample) or row['data_block'] == 'vector':
            return self._plan_vector(sample, row, modality, capture_id, where)
        if sample.encoding not in row['encodings']:
            raise DataError(ENCODING_REFUSED, f'{where}: encoding {sample.encoding!r} is not allowed for modality '
                            f'{modality!r} (vocabulary: {list(row["encodings"])})',
                            context={'where': where, 'encoding': sample.encoding})
        _codecs.check_array(sample.array, self.vocab.encodings[sample.encoding], sample.encoding)
        if row['data_block'] == 'instance_image':
            self._check_instance_mask(sample, where)
        media_type = sample.media_type or _codecs.default_media_type(sample.array, row['payloads'])
        if media_type not in row['payloads'] or not _codecs.accepts(sample.array, media_type):
            raise DataError(ENCODING_REFUSED, f'{where}: media type {media_type!r} is not a payload of modality '
                            f'{modality!r} that stores this array losslessly', context={'where': where})
        extension = _codecs.CODECS[media_type][0]
        height, width = sample.array.shape[:2]
        if sample.calibration_id is not None:
            self._check_calibration(sample, width, height, where)
        block = row['data_block']
        data = {'block': block, 'version': self.vocab.data_blocks[block]['version'], 'width': width, 'height': height,
                'encoding': sample.encoding, 'frame_id': sample.frame_id}
        if block == 'image':
            data['transforms'] = [_as_dict(t) for t in sample.transforms]
            s.put(data, 'unit_scale_m', sample.unit_scale_m)
            if sample.pose_at_stamp is not None:
                data['pose_at_stamp'] = _as_dict(sample.pose_at_stamp)
        elif sample.transforms or sample.unit_scale_m is not None or sample.pose_at_stamp is not None:
            s.fail(INVALID_VALUE, f'transforms / unit_scale_m / pose_at_stamp belong to the image block, not to '
                   f'{block!r}', where)
        s.put(data, 'calibration_id', sample.calibration_id)
        data.update(sample.block_fields or {})
        observation = {
            'observation_id': f'{capture_id}/{sample.sensor_id}/{sample.stream}', 'sensor_id': sample.sensor_id,
            'stream': sample.stream, 'modality': modality, 'stamp': _as_dict(sample.stamp), 'data': data,
            'payload': {'path': f'sessions/{self.session_id}/payload/{seq:06d}/{sample.sensor_id}__{sample.stream}'
                                f'{extension}', 'media_type': media_type, 'sha256': _PLACEHOLDER_SHA, 'bytes': 0}}
        for name in ('receipt', 'device_stamp', 'quality'):
            value = getattr(sample, name)
            if value is not None:
                observation[name] = _as_dict(value)
        if sample.extensions:
            observation['extensions'] = sample.extensions
        return observation

    def _plan_vector(self, sample, row, modality, capture_id: str, where: str) -> dict:
        """A ``vector`` observation: the values in the record, no payload (the block reader checks the rest)."""
        if not isinstance(sample, VectorSample) or row['data_block'] != 'vector':
            s.fail(INVALID_VALUE, f'stream {sample.stream_key!r} (modality {modality!r}, data block '
                   f'{row["data_block"]!r}) takes {"a VectorSample" if row["data_block"] == "vector" else "an image"}'
                   f', not a {type(sample).__name__}', where)
        observation = {
            'observation_id': f'{capture_id}/{sample.sensor_id}/{sample.stream}', 'sensor_id': sample.sensor_id,
            'stream': sample.stream, 'modality': modality, 'stamp': _as_dict(sample.stamp),
            'data': {'block': 'vector', 'version': self.vocab.data_blocks['vector']['version'],
                     'frame_id': sample.frame_id, 'values': list(sample.values),     # as given: the block
                     'units': list(sample.units)}}                                   # reader is the validator
        for name in ('receipt', 'device_stamp', 'quality'):
            value = getattr(sample, name)
            if value is not None:
                observation[name] = _as_dict(value)
        if sample.extensions:
            observation['extensions'] = sample.extensions
        return observation

    def _check_instance_mask(self, sample, where):
        """An instance mask must be exactly what its block says (ids, pixel counts, box extents), before any byte."""
        height, width = sample.array.shape[:2]
        fields = dict(sample.block_fields or {})
        document = {'block': 'instance_image', 'version': self.vocab.data_blocks['instance_image']['version'],
                    'width': width, 'height': height, 'encoding': sample.encoding, 'frame_id': sample.frame_id,
                    **fields}
        modality = self._modalities[sample.stream_key]
        block = InstanceImageBlock.from_dict(document, self.vocab.modalities[modality],
                                             self.vocab.data_blocks['instance_image'], f'{where}.data')
        problems = block.mask_problems(sample.array)
        if problems:
            s.fail(INVALID_VALUE, '; '.join(problems), where)

    def _check_calibration(self, sample, width, height, where):
        record = self._calibrations.get(sample.calibration_id)
        if record is None:
            record = self.store.get_calibration(sample.calibration_id, self.vocab)     # CALIBRATION_NOT_FOUND
        problems = []
        if (record.sensor_id, record.stream) != (sample.sensor_id, sample.stream):
            problems.append(f'it calibrates {record.sensor_id}/{record.stream}')
        if (record.intrinsics.width, record.intrinsics.height) != (width, height):
            problems.append(f'its size is {record.intrinsics.width}x{record.intrinsics.height}, the image is '
                            f'{width}x{height}')
        if record.optical_frame != sample.frame_id:
            problems.append(f'its optical frame is {record.optical_frame!r}, the image says {sample.frame_id!r}')
        if problems:
            raise DataError(CALIBRATION_MISMATCH, f'{where}: calibration {sample.calibration_id}: '
                            + '; '.join(problems), context={'where': where, 'calibration_id': sample.calibration_id})
        self._pending_calibrations[sample.calibration_id] = record

    def append_capture(self, *, trigger, outcome: str, samples=(), missing=(), sync=None,
                       extensions=None) -> Capture:
        """Persist one capture (one trigger firing) and return the committed record.

        ``outcome`` is what the caller obtained; if a payload cannot be written the committed record is
        downgraded (PARTIAL / FAILED, reason ``WRITE_FAILED``) and ``WRITE_FAILED`` is raised after it is
        committed, with ``context['capture_id']``.
        """
        self._require_active()
        seq = self._lines + 1
        capture_id = capture_id_for(self.session_id, seq)
        self._pending_calibrations = {}
        planned = [self._plan_sample(sample, capture_id, seq, f'capture.samples[{i}]')
                   for i, sample in enumerate(samples)]
        document = {'schema': CAPTURE_SCHEMA, 'capture_id': capture_id, 'seq': seq, 'trigger': _as_dict(trigger),
                    'outcome': outcome, 'observations': planned, 'missing': [_as_dict(m) for m in missing]}
        if sync is not None:
            document['sync'] = _as_dict(sync)
        if extensions:
            document['extensions'] = extensions
        problems = self.header.capture_problems(Capture.from_dict(document, self.vocab))   # before any byte
        if problems:
            s.fail(INVALID_VALUE, '; '.join(problems), 'capture')

        self._counts['attempted'] += 1
        try:
            written, failed = self._write_payloads(seq, samples, planned)
            if failed:
                document = self._downgrade(document, written, failed)
            capture = Capture.from_dict(document, self.vocab)
        except BaseException:
            # Not an OSError of one payload (those are recorded): this seq is spent and its files are strays.
            self.state = 'FAULTED'
            raise
        self._commit(capture, [o['payload']['path'] for o in written if 'payload' in o])
        if failed:
            raise DataError(WRITE_FAILED, f'capture {capture_id}: {len(failed)} payload(s) could not be written; '
                            f'recorded as {capture.outcome}', context={
                                'capture_id': capture_id, 'outcome': capture.outcome,
                                'failed_streams': [m['stream'] for m in failed]})
        return capture

    def _write_payloads(self, seq, samples, planned):
        if not any('payload' in o for o in planned):
            return list(planned), []            # vector readings only: everything is in the index line
        written, failed = [], []
        seq_dir = self._dir / 'payload' / f'{seq:06d}'
        try:
            seq_dir.mkdir()
            _fsio.fsync_dir(seq_dir.parent)
        except OSError as exc:
            self._write_errors += sum(1 for o in planned if 'payload' in o)     # a vector reading needs no file
            if seq_dir.exists():
                self._strays.append(f'payload/{seq:06d}')
            return [o for o in planned if 'payload' not in o], [
                {'stream': o['sensor_id'] + '/' + o['stream'], 'reason': 'WRITE_FAILED', 'detail': _os_detail(exc)}
                for o in planned if 'payload' in o]
        try:
            for sample, observation in zip(samples, planned):
                if 'payload' not in observation:
                    written.append(observation)     # a vector reading: nothing to write
                    continue
                payload = observation['payload']
                data, _ = _codecs.encode(sample.array, payload['media_type'])
                path = self.store.path(payload['path'])
                try:
                    _fsio.write_new_file(path, data)
                except OSError as exc:
                    self._write_errors += 1
                    if os.path.lexists(_fsio.tmp_path(path)):
                        self._strays.append(self._session_rel(payload['path']) + _fsio.TMP_SUFFIX)
                    if os.path.lexists(path):       # renamed, but its directory entry was not made durable
                        self._strays.append(self._session_rel(payload['path']))
                    failed.append({'stream': f'{sample.sensor_id}/{sample.stream}', 'reason': 'WRITE_FAILED',
                                   'detail': _os_detail(exc)})
                    continue
                payload['sha256'] = sha256_bytes(data)
                payload['bytes'] = len(data)
                written.append(observation)
        except BaseException:
            # Unexpected (not an OSError of one payload): the payload directory of this seq is unreferenced.
            self._strays.append(f'payload/{seq:06d}')
            raise
        return written, failed

    @staticmethod
    def _downgrade(document, written, failed):
        document = dict(document, observations=written, missing=document['missing'] + failed)
        document['outcome'] = 'PARTIAL' if written else 'FAILED'
        return document

    def _session_rel(self, store_rel: str) -> str:
        return store_rel[len(f'sessions/{self.session_id}/'):]

    def _commit(self, capture: Capture, payload_paths):
        line = canonical_json(capture.to_dict()) + b'\n'
        try:
            self._index.append(line)
        except OSError as exc:
            self.state = 'FAULTED'
            self._write_errors += 1
            self._strays.extend(self._session_rel(p) for p in payload_paths)
            raise DataError(WRITE_FAILED, f'capture {capture.capture_id}: the index line could not be written; the '
                            f'session is FAULTED, run `trainit-data recover {self.session_id}`',
                            context={'capture_id': capture.capture_id, 'session_id': self.session_id},
                            cause=exc) from None
        self._index_hash.update(line)
        self._lines += 1
        self._counts[capture.outcome.lower()] += 1
        for cal_id, record in self._pending_calibrations.items():
            if any(o.data.calibration_id == cal_id for o in capture.observations):
                self._calibrations.setdefault(cal_id, record)

    # ---- sealing -------------------------------------------------------------------------------------

    def close(self, *, received=None, writer_overruns: int = 0, calibration_changes: int = 0,
              warnings=()) -> SessionHeader:
        """Seal the session CLOSED: quarantine stray bytes, write the final header, drop the marker."""
        self._require_active()
        moved = []
        try:
            for rel in self._strays:
                moved.append({'from': rel, 'to': _fsio.quarantine(self._dir, rel)})
        except OSError as exc:
            self.state = 'FAULTED'
            raise DataError(WRITE_FAILED, f'session {self.session_id}: a stray file could not be quarantined; '
                            'run `trainit-data recover`', context={'session_id': self.session_id}, cause=exc) from None
        all_warnings = list(self.header.warnings) + [_as_dict(w) for w in warnings]
        if moved:
            all_warnings.append({'code': WRITE_FAILED, 'detail': f'{len(moved)} partially written file(s) moved '
                                 'to orphans/', 'context': {'moved': moved}})
        counters = {'captures': dict(self._counts), 'write_errors': self._write_errors,
                    'writer_overruns': writer_overruns, 'calibration_changes': calibration_changes}
        if received is not None:
            counters['received'] = dict(received)
        document = self.header.to_dict()
        document.update(status='CLOSED', closed={'wall_utc_ns': time.time_ns(), 'by': 'writer'},
                        calibration_ids=list(self._calibrations), counters=counters,
                        index={'path': f'sessions/{self.session_id}/captures.jsonl',
                               'sha256': 'sha256:' + self._index_hash.hexdigest(), 'lines': self._lines})
        if all_warnings:
            document['warnings'] = all_warnings
        header = SessionHeader.from_dict(document, self.vocab)
        try:
            _fsio.replace_file(self._dir / 'session.json', pretty_json(header.to_dict()))
        except OSError as exc:
            self.state = 'FAULTED'
            raise DataError(WRITE_FAILED, f'session {self.session_id}: the sealed header could not be written; run '
                            '`trainit-data recover`', context={'session_id': self.session_id}, cause=exc) from None
        self._index.close()
        self._lock.release(remove=True)
        self.header = header
        self.state = 'SEALED'
        return header

    def release(self):
        """Stop writing WITHOUT sealing (the session stays ACTIVE for ``recover``). Idempotent."""
        if self.state == 'SEALED':
            return
        self._index.close()
        self._lock.release(remove=False)
        if self.state == 'ACTIVE':
            self.state = 'RELEASED'     # this object can no longer write; the session itself is intact
