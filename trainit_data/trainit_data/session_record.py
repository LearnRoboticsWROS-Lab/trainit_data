"""The session header ``session.json`` (``trainit-session/1.1``) (PUBLIC).

A session is ONE acquisition or ingest run (D0_DESIGN.md §5, D0_SCHEMA_PROPOSAL.md §6). It is NOT a
dataset and NOT a dataset version: those are slice D0.6 and reference sealed sessions by their index hash.

Lifecycle: ``ACTIVE`` (append-only: the writer adds index lines and payloads) -> ``CLOSED`` (sealed by
the writer) or ``INTERRUPTED`` (sealed by ``recover``). A sealed header carries ``closed``, the index hash
and line count, and the counters; an ACTIVE header carries none of them.

Typed here: identity, status, source (transport x backend, the backend's source descriptor), the pinned
vocabulary, the session clock, the declared sensors / streams, calibration ids, the recording
configuration snapshot, the provenance block names, rights, warnings, counters and the index. Opaque here
(stored, hashed, carried, validated by their producer): ``backend_details``, the resolved recording
configuration, and the content of each provenance block.
"""
import copy
from dataclasses import dataclass, field

from . import _strict as s
from ._canonical import sha256_json
from .errors import INVALID_VALUE
from .timestamps import CLOCK_DOMAINS

SCHEMA_NAME = 'trainit-session'
SCHEMA_MAJOR = 1
# 1.1 (D0.4) adds the optional provenance block `acquisition`. Evolution is backward-only inside a major
# (D0_SCHEMA_PROPOSAL.md §0): this reader reads 1.0 and 1.1; a 1.0-era reader refuses a 1.1 session that carries the
# block (UNKNOWN_FIELD) -- readers are upgraded with the writers, never the other way round.
SCHEMA = 'trainit-session/1.1'

# CLOSED sets: implemented lifecycle and rights semantics.
STATUSES = frozenset({'ACTIVE', 'CLOSED', 'INTERRUPTED'})
SEALED = frozenset({'CLOSED', 'INTERRUPTED'})
CLASSIFICATIONS = frozenset({'INTERNAL', 'CUSTOMER_RESTRICTED', 'PUBLIC_APPROVED'})
CLOSED_BY = frozenset({'writer', 'recover'})
# `acquisition` (D0.4): where the data was acquired and how it reached the recorder -- the source's own document
# (device, attachment, acquisition host) and the recorder's host and transport. Typed by its producer
# (trainit_data_ros), never by a device schema here.
PROVENANCE_BLOCKS = ('software', 'application', 'robot', 'scene', 'randomization', 'acquisition')


def _descriptor(vocab):
    def check(value, where):
        f = s.Fields(value, where)
        out = {}
        s.put(out, 'intrinsics_kind', f.opt('intrinsics_kind', vocab.checker('intrinsic_provenance')))
        s.put(out, 'registration_basis', f.opt('registration_basis', s.token))
        s.put(out, 'stamp_origin', f.opt('stamp_origin', vocab.checker('stamp_origins')))
        s.put(out, 'device', f.opt('device', s.str_map))
        s.put(out, 'supplied_by', f.opt('supplied_by', s.string))
        f.done()
        return out
    return check


@dataclass(frozen=True)
class Source:
    """Where the observations came from: transport x backend kind + the backend's source descriptor."""
    transport: str                      # vocabulary: transports (LIVE_ROS | INGEST)
    backend: str                        # vocabulary: backends
    adapter: str = None                 # INGEST: the adapter module that produced the records
    replay_of: str = None               # replay: the original session id
    descriptor: dict = field(default_factory=dict)
    backend_details: dict = field(default_factory=dict)     # OPAQUE to the core

    @classmethod
    def from_dict(cls, value, vocab, where='source'):
        f = s.Fields(value, where)
        source = cls(transport=f.req('transport', vocab.checker('transports')),
                     backend=f.req('backend', vocab.checker('backends')),
                     adapter=f.opt('adapter', s.token),
                     replay_of=f.opt('replay_of', s.session_id),
                     descriptor=f.opt('descriptor', _descriptor(vocab), {}),
                     backend_details=f.opt('backend_details', s.opaque, {}))
        f.done()
        return source

    def to_dict(self):
        out = {'transport': self.transport, 'backend': self.backend}
        s.put(out, 'adapter', self.adapter)
        s.put(out, 'replay_of', self.replay_of)
        if self.descriptor:
            out['descriptor'] = copy.deepcopy(self.descriptor)
        if self.backend_details:
            out['backend_details'] = copy.deepcopy(self.backend_details)
        return out


@dataclass(frozen=True)
class Clock:
    session_domain: str
    sim_source: str = None
    use_sim_time: bool = None

    @property
    def key(self) -> tuple:
        return (self.session_domain, self.sim_source)

    @classmethod
    def from_dict(cls, value, vocab, where='clock'):
        f = s.Fields(value, where)
        clock = cls(session_domain=f.req('session_domain', s.one_of(CLOCK_DOMAINS)),
                    sim_source=f.opt('sim_source', vocab.checker('sim_sources')),
                    use_sim_time=f.opt('use_sim_time', s.boolean))
        f.done()
        if (clock.session_domain == 'SIM_TIME') != (clock.sim_source is not None):
            s.fail(INVALID_VALUE, 'sim_source is given exactly when the session domain is SIM_TIME', where)
        return clock

    def to_dict(self):
        out = {'session_domain': self.session_domain}
        s.put(out, 'sim_source', self.sim_source)
        s.put(out, 'use_sim_time', self.use_sim_time)
        return out


@dataclass(frozen=True)
class StreamDecl:
    stream: str
    modality: str                       # vocabulary: modalities
    topic: str = None                   # LIVE_ROS: provenance only

    @classmethod
    def from_dict(cls, value, vocab, where='stream'):
        f = s.Fields(value, where)
        decl = cls(stream=f.req('stream', s.ident), modality=f.req('modality', vocab.checker('modalities')),
                   topic=f.opt('topic', s.token))
        f.done()
        return decl

    def to_dict(self):
        out = {'stream': self.stream, 'modality': self.modality}
        s.put(out, 'topic', self.topic)
        return out


def _profile(value, where):
    f = s.Fields(value, where)
    profile = {'model': f.req('model', s.token)}
    sha = f.opt('sha256', s.sha256)
    if sha is not None:
        profile['sha256'] = sha
    f.done()
    return profile


@dataclass(frozen=True)
class SensorDecl:
    sensor_id: str
    streams: tuple
    profile: dict = None                # {model, sha256?}: the hardware profile the sensor follows

    @classmethod
    def from_dict(cls, value, vocab, where='sensor'):
        f = s.Fields(value, where)
        decl = cls(sensor_id=f.req('sensor_id', s.ident),
                   streams=f.req('streams', s.list_of(lambda v, w: StreamDecl.from_dict(v, vocab, w), min_length=1)),
                   profile=f.opt('profile', _profile))
        f.done()
        names = [st.stream for st in decl.streams]
        if len(set(names)) != len(names):
            s.fail(INVALID_VALUE, 'a stream is declared twice', where)
        return decl

    def to_dict(self):
        out = {'sensor_id': self.sensor_id, 'streams': [st.to_dict() for st in self.streams]}
        if self.profile is not None:
            out['profile'] = copy.deepcopy(self.profile)
        return out


def _vocab_ref(value, where):
    f = s.Fields(value, where)
    ref = {'schema': f.req('schema', s.schema('trainit-vocab', 1)), 'sha256': f.req('sha256', s.sha256)}
    f.done()
    return ref


def _recording_config(value, where):
    f = s.Fields(value, where)
    config = {'resolved': f.req('resolved', s.opaque), 'resolved_sha256': f.req('resolved_sha256', s.sha256)}
    file_sha = f.opt('file_sha256', s.sha256)
    if file_sha is not None:
        config['file_sha256'] = file_sha
    f.done()
    if sha256_json(config['resolved']) != config['resolved_sha256']:
        s.fail(INVALID_VALUE, 'resolved_sha256 is not the content hash of `resolved`', where)
    return config


def _provenance(value, where):
    f = s.Fields(value, where)
    out = {}
    for name in PROVENANCE_BLOCKS:
        block = f.opt(name, s.str_map if name == 'software' else s.opaque)
        if block is not None:
            out[name] = block
    f.done()
    return out


def _rights(value, where):
    f = s.Fields(value, where)
    rights = {'classification': f.req('classification', s.one_of(CLASSIFICATIONS))}
    for name in ('owner_org', 'license', 'notes'):
        s.put(rights, name, f.opt(name, s.string))
    f.done()
    return rights


def _warning(value, where):
    f = s.Fields(value, where)
    warning = {'code': f.req('code', s.code)}
    s.put(warning, 'detail', f.opt('detail', s.text))
    context = f.opt('context', s.opaque)
    if context:
        warning['context'] = context
    f.done()
    return warning


def _counters(value, where):
    f = s.Fields(value, where)
    captures = f.req('captures', s.opaque)
    cf = s.Fields(captures, f'{where}.captures')
    counts = {k: cf.req(k, s.integer(0)) for k in ('attempted', 'complete', 'partial', 'failed')}
    cf.done()
    counters = {'captures': counts, 'write_errors': f.req('write_errors', s.integer(0))}
    for name in ('writer_overruns', 'calibration_changes'):    # absent = unknown (e.g. after a crash)
        s.put(counters, name, f.opt(name, s.integer(0)))
    received = f.opt('received', s.opaque)
    if received is not None:
        for key, count in received.items():
            s.integer(0)(count, f'{where}.received.{key}')
        counters['received'] = received
    f.done()
    if counts['complete'] + counts['partial'] + counts['failed'] > counts['attempted']:
        s.fail(INVALID_VALUE, 'complete + partial + failed exceeds attempted', where)
    return counters


def _index(value, where):
    f = s.Fields(value, where)
    index = {'path': f.req('path', s.relative_path), 'sha256': f.req('sha256', s.sha256),
             'lines': f.req('lines', s.integer(0))}
    f.done()
    return index


def _opened(value, where):
    f = s.Fields(value, where)
    out = {'wall_utc_ns': f.req('wall_utc_ns', s.integer(0))}
    f.done()
    return out


def _closed(value, where):
    f = s.Fields(value, where)
    out = {'wall_utc_ns': f.req('wall_utc_ns', s.integer(0)), 'by': f.req('by', s.one_of(CLOSED_BY))}
    f.done()
    return out


@dataclass(frozen=True)
class SessionHeader:
    schema: str
    session_id: str
    status: str
    opened: dict                        # {wall_utc_ns}
    source: Source
    vocab: dict                         # {schema, sha256}: the pinned, authoritative vocabulary
    clock: Clock
    sensors: tuple
    rights: dict
    calibration_ids: tuple = ()
    recording_config: dict = None
    provenance: dict = field(default_factory=dict)
    quality: dict = None
    warnings: tuple = ()
    closed: dict = None                 # sealed only: {wall_utc_ns, by}
    counters: dict = None               # sealed only
    index: dict = None                  # sealed only: {path, sha256, lines}
    extensions: dict = field(default_factory=dict)

    @property
    def sealed(self) -> bool:
        return self.status in SEALED

    def stream_modalities(self) -> dict:
        """{``sensor/stream``: modality} of every declared stream."""
        return {f'{d.sensor_id}/{st.stream}': st.modality for d in self.sensors for st in d.streams}

    def capture_problems(self, capture) -> list:
        """The rules a capture must meet IN THIS SESSION (the writer refuses them; verify reports them):
        its id belongs to the session, its streams are declared with their modality, the trigger boundary is on
        the session clock, and an observation off the session clock is never COMPLETE nor called synchronized."""
        problems = []
        if capture.session_id != self.session_id:
            problems.append(f'capture {capture.capture_id} does not belong to session {self.session_id}')
        declared = self.stream_modalities()
        for obs in capture.observations:
            if declared.get(obs.stream_key) != obs.modality:
                problems.append(f'{obs.stream_key} ({obs.modality}) is not a declared stream of the session')
        for missing in capture.missing:
            if missing.stream not in declared:
                problems.append(f'missing stream {missing.stream} is not declared by the session')
        if capture.trigger.issued.stamp.clock != self.clock.key:
            problems.append(f'the trigger stamp is not on the session clock {self.clock.session_domain}')
        foreign = [o.observation_id for o in capture.observations if o.stamp.clock != self.clock.key]
        if foreign and (capture.sync is None or capture.sync.status != 'INCOMPATIBLE_CLOCKS'
                        or capture.outcome == 'COMPLETE'):
            problems.append(f'{foreign} are not on the session clock {self.clock.session_domain}: the sync status '
                            'is INCOMPATIBLE_CLOCKS and the capture is not COMPLETE')
        return problems

    @classmethod
    def from_dict(cls, value, vocab, where='session') -> 'SessionHeader':
        f = s.Fields(value, where)
        header = cls(
            schema=f.req('schema', s.schema(SCHEMA_NAME, SCHEMA_MAJOR)),
            session_id=f.req('session_id', s.session_id),
            status=f.req('status', s.one_of(STATUSES)),
            opened=f.req('opened', _opened),
            source=f.req('source', lambda v, w: Source.from_dict(v, vocab, w)),
            vocab=f.req('vocab', _vocab_ref),
            clock=f.req('clock', lambda v, w: Clock.from_dict(v, vocab, w)),
            sensors=f.req('sensors', s.list_of(lambda v, w: SensorDecl.from_dict(v, vocab, w), min_length=1)),
            rights=f.req('rights', _rights),
            calibration_ids=f.req('calibration_ids', s.list_of(s.calibration_id, unique=True)),
            recording_config=f.opt('recording_config', _recording_config),
            provenance=f.opt('provenance', _provenance, {}),
            quality=f.opt('quality', s.opaque),
            warnings=f.opt('warnings', s.list_of(_warning), ()),
            closed=f.opt('closed', _closed),
            counters=f.opt('counters', _counters),
            index=f.opt('index', _index),
            extensions=f.extensions())
        f.done()
        if header.vocab['sha256'] != vocab.sha256:
            s.fail(INVALID_VALUE, 'the session pins another vocabulary than the one given to read it', where)
        ids = [d.sensor_id for d in header.sensors]
        if len(set(ids)) != len(ids):
            s.fail(INVALID_VALUE, 'a sensor is declared twice', where)
        sealed_parts = (header.closed, header.counters, header.index)
        if header.sealed and any(p is None for p in sealed_parts):
            s.fail(INVALID_VALUE, f'a {header.status} session carries closed, counters and index', where)
        if not header.sealed and any(p is not None for p in sealed_parts):
            s.fail(INVALID_VALUE, 'an ACTIVE session carries no closed / counters / index yet', where)
        if header.status == 'INTERRUPTED' and header.closed['by'] != 'recover':
            s.fail(INVALID_VALUE, 'only recover seals a session INTERRUPTED', where)
        if header.status == 'CLOSED' and header.closed['by'] != 'writer':
            s.fail(INVALID_VALUE, 'only the writer seals a session CLOSED', where)
        return header

    def to_dict(self) -> dict:
        out = {'schema': self.schema, 'session_id': self.session_id, 'status': self.status,
               'opened': dict(self.opened), 'source': self.source.to_dict(), 'vocab': dict(self.vocab),
               'clock': self.clock.to_dict(), 'sensors': [d.to_dict() for d in self.sensors],
               'rights': dict(self.rights), 'calibration_ids': list(self.calibration_ids)}
        for name in ('recording_config', 'quality', 'closed', 'counters', 'index'):
            value = getattr(self, name)
            if value is not None:
                out[name] = copy.deepcopy(value)
        if self.provenance:
            out['provenance'] = copy.deepcopy(self.provenance)
        if self.warnings:
            out['warnings'] = copy.deepcopy(list(self.warnings))
        if self.extensions:
            out['extensions'] = copy.deepcopy(self.extensions)
        return out


__all__ = ['SessionHeader', 'Source', 'Clock', 'SensorDecl', 'StreamDecl', 'STATUSES', 'CLASSIFICATIONS',
           'PROVENANCE_BLOCKS']
