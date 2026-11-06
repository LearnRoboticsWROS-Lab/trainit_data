"""Capture: ONE trigger firing and the observations it grouped (PUBLIC).

``trainit-capture/1.0`` is an EVENT record, one line of a session's ``captures.jsonl``
(D0_SCHEMA_PROPOSAL.md §5). It holds the trigger record, the synchronization result (policy name, version
and parameters), the outcome and EVERY observation obtained. A PARTIAL or FAILED capture is recorded with a
reason per missing stream; it never disappears. This module is the data contract only: the live
CaptureCoordinator that fills it is slice D0.2.

Cross-field rules enforced on parse (so ``verify`` re-checks them on every line):
- outcome: COMPLETE = observations, nothing missing, nothing out of tolerance; PARTIAL = some observations
  and a stream missing or out of tolerance (D0_DESIGN.md §7.3); FAILED = no observation;
- stream keys are unique and an observed stream is never also missing;
- clocks are never compared across domains: observations in more than one clock domain require the sync
  status ``INCOMPATIBLE_CLOCKS`` and a non-COMPLETE outcome;
- a recorded skew must equal the skew of the recorded stamps against the reference stream (with a stream
  missing it may be larger: the missing sample may have been obtained but not persisted).
"""
import copy
import re
from dataclasses import dataclass, field

from . import _strict as s
from .errors import INVALID_VALUE
from .observation import Observation
from .timestamps import Timestamp

SCHEMA_NAME = 'trainit-capture'
SCHEMA_MAJOR = 1
SCHEMA = 'trainit-capture/1.0'

# CLOSED sets: the core implements their semantics (the rules above).
OUTCOMES = frozenset({'COMPLETE', 'PARTIAL', 'FAILED'})
SYNC_STATUSES = frozenset({'SYNCED', 'WITHIN_TOLERANCE', 'OUT_OF_TOLERANCE', 'SINGLE_STREAM', 'INCOMPATIBLE_CLOCKS'})

_CAPTURE_ID_RE = re.compile(r'^(ses-\d{8}T\d{6}Z-[0-9a-f]{8})/(\d{6,})$')   # 6 digits, more past 999999
_STREAM_KEY_RE = re.compile(r'^[a-z0-9]+(?:_[a-z0-9]+)*/[a-z0-9]+(?:_[a-z0-9]+)*$')
stream_key = s.matching(_STREAM_KEY_RE, 'stream key (<sensor_id>/<stream>)')


def capture_id_for(session_id: str, seq: int) -> str:
    return f'{session_id}/{seq:06d}'


@dataclass(frozen=True)
class Issued:
    stamp: Timestamp                    # the trigger boundary, on the session clock
    host_monotonic_ns: int = None
    wall_utc_ns: int = None

    @classmethod
    def from_dict(cls, value, vocab, where='issued'):
        f = s.Fields(value, where)
        issued = cls(stamp=f.req('stamp', lambda v, w: Timestamp.from_dict(v, vocab, w)),
                     host_monotonic_ns=f.opt('host_monotonic_ns', s.integer(0)),
                     wall_utc_ns=f.opt('wall_utc_ns', s.integer(0)))
        f.done()
        return issued

    def to_dict(self):
        out = {'stamp': self.stamp.to_dict()}
        s.put(out, 'host_monotonic_ns', self.host_monotonic_ns)
        s.put(out, 'wall_utc_ns', self.wall_utc_ns)
        return out


@dataclass(frozen=True)
class TriggerRecord:
    trigger_id: str
    kind: str                           # vocabulary: trigger_kinds
    firing_index: int                   # 0-based count of this trigger's firings in the session
    origin: str                         # vocabulary: trigger_origins
    issued: Issued
    reason: dict = field(default_factory=dict)      # kind-specific facts (a MANUAL note, an INTERVAL phase)
    context: dict = field(default_factory=dict)     # {namespace: {...ids...}}: the D1 join seam, ids only

    @classmethod
    def from_dict(cls, value, vocab, where='trigger'):
        f = s.Fields(value, where)
        trigger = cls(trigger_id=f.req('trigger_id', s.ident),
                      kind=f.req('kind', vocab.checker('trigger_kinds')),
                      firing_index=f.req('firing_index', s.integer(0)),
                      origin=f.req('origin', vocab.checker('trigger_origins')),
                      issued=f.req('issued', lambda v, w: Issued.from_dict(v, vocab, w)),
                      reason=f.opt('reason', s.opaque, {}),
                      context=f.opt('context', s.opaque, {}))
        f.done()
        for namespace, entry in trigger.context.items():
            vocab.require('context_namespaces', namespace, f'{where}.context')
            s.opaque(entry, f'{where}.context.{namespace}')
        return trigger

    def to_dict(self):
        out = {'trigger_id': self.trigger_id, 'kind': self.kind, 'firing_index': self.firing_index,
               'origin': self.origin, 'issued': self.issued.to_dict()}
        if self.reason:
            out['reason'] = copy.deepcopy(self.reason)
        if self.context:
            out['context'] = copy.deepcopy(self.context)
        return out


@dataclass(frozen=True)
class SyncPolicyRef:
    name: str                           # vocabulary: sync_policies (FIRST_AT_OR_AFTER_BOUNDARY v1 in D0)
    version: int
    params: dict = field(default_factory=dict)

    @classmethod
    def from_dict(cls, value, vocab, where='policy'):
        f = s.Fields(value, where)
        policy = cls(name=f.req('name', s.token), version=f.req('version', s.integer(1)),
                     params=f.opt('params', s.opaque, {}))
        f.done()
        vocab.require_sync_policy(policy.name, policy.version, where)
        return policy

    def to_dict(self):
        out = {'name': self.name, 'version': self.version}
        if self.params:
            out['params'] = copy.deepcopy(self.params)
        return out


@dataclass(frozen=True)
class SyncResult:
    policy: SyncPolicyRef
    reference: str                      # <sensor_id>/<stream>
    status: str
    tolerance_ns: int = None
    max_skew_ns: int = None

    @classmethod
    def from_dict(cls, value, vocab, where='sync'):
        f = s.Fields(value, where)
        sync = cls(policy=f.req('policy', lambda v, w: SyncPolicyRef.from_dict(v, vocab, w)),
                   reference=f.req('reference', stream_key),
                   status=f.req('status', s.one_of(SYNC_STATUSES)),
                   tolerance_ns=f.opt('tolerance_ns', s.integer(0)),
                   max_skew_ns=f.opt('max_skew_ns', s.integer(0)))
        f.done()
        if sync.status == 'INCOMPATIBLE_CLOCKS' and sync.max_skew_ns is not None:
            s.fail(INVALID_VALUE, 'no skew exists across clock domains', where)
        if sync.status in ('SYNCED', 'WITHIN_TOLERANCE', 'OUT_OF_TOLERANCE') and sync.max_skew_ns is None:
            s.fail(INVALID_VALUE, f'status {sync.status} records max_skew_ns', where)
        if sync.status == 'SYNCED' and sync.max_skew_ns != 0:
            s.fail(INVALID_VALUE, 'SYNCED means identical stamps (max_skew_ns 0)', where)
        if sync.status in ('WITHIN_TOLERANCE', 'OUT_OF_TOLERANCE'):
            if sync.tolerance_ns is None:
                s.fail(INVALID_VALUE, f'status {sync.status} records tolerance_ns', where)
            within = sync.max_skew_ns <= sync.tolerance_ns
            if within != (sync.status == 'WITHIN_TOLERANCE'):
                s.fail(INVALID_VALUE, f'max_skew_ns {sync.max_skew_ns} vs tolerance_ns {sync.tolerance_ns} '
                       f'contradicts status {sync.status}', where)
        return sync

    def to_dict(self):
        out = {'policy': self.policy.to_dict(), 'reference': self.reference, 'status': self.status}
        s.put(out, 'tolerance_ns', self.tolerance_ns)
        s.put(out, 'max_skew_ns', self.max_skew_ns)
        return out


@dataclass(frozen=True)
class Missing:
    stream: str                         # <sensor_id>/<stream>
    reason: str                         # vocabulary: missing_reasons
    detail: str = None

    @classmethod
    def from_dict(cls, value, vocab, where='missing'):
        f = s.Fields(value, where)
        missing = cls(stream=f.req('stream', stream_key), reason=f.req('reason', vocab.checker('missing_reasons')),
                      detail=f.opt('detail', s.text))
        f.done()
        return missing

    def to_dict(self):
        out = {'stream': self.stream, 'reason': self.reason}
        s.put(out, 'detail', self.detail)
        return out


@dataclass(frozen=True)
class Capture:
    schema: str
    capture_id: str                     # <session_id>/<seq:06d>
    seq: int                            # 1-based position in the session index
    trigger: TriggerRecord
    outcome: str
    observations: tuple = ()
    missing: tuple = ()
    sync: SyncResult = None             # required unless the capture FAILED
    extensions: dict = field(default_factory=dict)

    @property
    def session_id(self) -> str:
        return self.capture_id.split('/')[0]

    @classmethod
    def from_dict(cls, value, vocab, where='capture') -> 'Capture':
        f = s.Fields(value, where)
        capture = cls(
            schema=f.req('schema', s.schema(SCHEMA_NAME, SCHEMA_MAJOR)),
            capture_id=f.req('capture_id', s.matching(_CAPTURE_ID_RE, 'capture id (<session_id>/<seq:06d>)')),
            seq=f.req('seq', s.integer(1)),
            trigger=f.req('trigger', lambda v, w: TriggerRecord.from_dict(v, vocab, w)),
            outcome=f.req('outcome', s.one_of(OUTCOMES)),
            observations=f.req('observations', s.list_of(lambda v, w: Observation.from_dict(v, vocab, w))),
            missing=f.req('missing', s.list_of(lambda v, w: Missing.from_dict(v, vocab, w))),
            sync=f.opt('sync', lambda v, w: SyncResult.from_dict(v, vocab, w)),
            extensions=f.extensions())
        f.done()
        capture._check(where)
        return capture

    def _check(self, where):
        if int(_CAPTURE_ID_RE.match(self.capture_id).group(2)) != self.seq:
            s.fail(INVALID_VALUE, f'capture_id {self.capture_id!r} does not carry seq {self.seq}', where)
        for i, obs in enumerate(self.observations):
            if obs.observation_id != f'{self.capture_id}/{obs.sensor_id}/{obs.stream}':
                s.fail(INVALID_VALUE, f'observation_id {obs.observation_id!r} is not '
                       f'"{self.capture_id}/{obs.stream_key}"', f'{where}.observations[{i}]')
        observed = [o.stream_key for o in self.observations]
        missing = [m.stream for m in self.missing]
        if len(set(observed)) != len(observed) or len(set(missing)) != len(missing) or set(observed) & set(missing):
            s.fail(INVALID_VALUE, 'a stream appears twice (observed and / or missing)', where)
        if self.outcome == 'COMPLETE' and (not observed or missing):
            s.fail(INVALID_VALUE, 'COMPLETE = at least one observation and nothing missing', where)
        out_of_tolerance = self.sync is not None and self.sync.status == 'OUT_OF_TOLERANCE'
        if self.outcome == 'PARTIAL' and (not observed or not (missing or out_of_tolerance)):
            s.fail(INVALID_VALUE, 'PARTIAL = at least one observation, and a stream missing or out of tolerance',
                   where)
        if self.outcome == 'FAILED' and (observed or not missing):
            s.fail(INVALID_VALUE, 'FAILED = no observation and at least one missing stream (with its reason)',
                   where)
        if self.outcome == 'COMPLETE' and self.sync is not None and self.sync.status == 'OUT_OF_TOLERANCE':
            s.fail(INVALID_VALUE, 'a stream out of tolerance makes the capture PARTIAL, not COMPLETE', where)
        if self.sync is None:
            if self.outcome != 'FAILED':
                s.fail(INVALID_VALUE, 'only a FAILED capture may omit its sync result', where)
            return
        self._check_clocks(f'{where}.sync')

    def _check_clocks(self, where):
        sync = self.sync
        by_key = {o.stream_key: o for o in self.observations}
        if sync.reference not in by_key and sync.reference not in {m.stream for m in self.missing}:
            s.fail(INVALID_VALUE, f'reference {sync.reference!r} is not a stream of this capture', where)
        clocks = {o.stamp.clock for o in self.observations}
        if len(clocks) > 1:
            if sync.status != 'INCOMPATIBLE_CLOCKS' or self.outcome == 'COMPLETE':
                s.fail(INVALID_VALUE, 'observations span several clock domains: the status is INCOMPATIBLE_CLOCKS '
                       'and the capture is not COMPLETE (stamps are never compared across domains)', where)
            return
        if sync.status == 'INCOMPATIBLE_CLOCKS':
            if not any(m.reason in ('INCOMPATIBLE_CLOCKS', 'WRITE_FAILED') for m in self.missing):
                s.fail(INVALID_VALUE, 'INCOMPATIBLE_CLOCKS with one clock domain needs a stream missing for that '
                       'reason (or one whose payload could not be written)', where)
            return
        if not self.observations or sync.reference not in by_key:
            return                      # the samples the sync was computed on were not all persisted
        if sync.status == 'SINGLE_STREAM':
            if len(self.observations) != 1:
                s.fail(INVALID_VALUE, 'SINGLE_STREAM = exactly one observation', where)
            return
        ref = by_key[sync.reference].stamp.to_ns()
        skew = max(abs(o.stamp.to_ns() - ref) for o in self.observations)
        # With a stream missing, the recorded skew may include a sample that was obtained but not persisted.
        if skew > sync.max_skew_ns or (not self.missing and skew != sync.max_skew_ns):
            s.fail(INVALID_VALUE, f'max_skew_ns {sync.max_skew_ns} contradicts the recorded stamps (skew {skew} '
                   'against the reference)', where)

    def to_dict(self) -> dict:
        out = {'schema': self.schema, 'capture_id': self.capture_id, 'seq': self.seq,
               'trigger': self.trigger.to_dict(), 'outcome': self.outcome,
               'observations': [o.to_dict() for o in self.observations],
               'missing': [m.to_dict() for m in self.missing]}
        if self.sync is not None:
            out['sync'] = self.sync.to_dict()
        if self.extensions:
            out['extensions'] = copy.deepcopy(self.extensions)
        return out


__all__ = ['Capture', 'TriggerRecord', 'Issued', 'SyncResult', 'SyncPolicyRef', 'Missing', 'OUTCOMES',
           'SYNC_STATUSES', 'capture_id_for']
