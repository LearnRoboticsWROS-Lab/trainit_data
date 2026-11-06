"""``trainit-recording/1.0``: the recorder's runtime acquisition configuration (strict; no ROS imports).

The file says only what the user chose (D0_SCHEMA_PROPOSAL.md §11). :func:`resolve` fills every default
from the hardware profile and the camera contract, validates everything against the vocabulary, and
returns the EFFECTIVE configuration -- the ``resolved`` block the session persists, from which a later
reader recovers exactly what the recorder ran with::

    schema: trainit-recording/1.0
    source: {backend: mock, replay_of?: <session id>, descriptor?: {...}}   # descriptor: else the profile's
    clock: {sim_source?: isaac}                    # required exactly when the node runs on use_sim_time
    rights: {classification: INTERNAL, owner_org?: ..., license?: ..., notes?: ...}
    sensors:
      - sensor_id: overhead_camera
        profile: realsense_d455                    # trainit_hardware_layer camera profile
        namespace?: /camera                        # default: the profile's topics.namespace
        streams: {color: {}, depth: {}}            # contract streams; per stream: topic? info? expected_frame?
        source_info?: {topic?: <abs>, required?: false}   # the camera's latched source document (D0.4): default
                                                   # <namespace>/source_info; required: refuse a session without it
        extrinsic_parent_frame?: base_link         # record the TF parent -> optical frame (required if given)
        extrinsic_provenance?: NOMINAL_DESCRIPTION
        pose_parent_frame?: world                  # moving camera: pose_at_stamp at every observation stamp
    triggers: [{id: manual, kind: MANUAL}, {id: every_2s, kind: INTERVAL, period_ms: 2000, max_captures?: 500},
               {id: on_glue, kind: EVENT, process_channel: glue, edge: rising},
               {id: approach, kind: APPLICATION_STATE, task: pick_one, step?: approach, transition: enter,
                context?: {pick_attempt: {phase: pre_grasp}}}]     # D0.5: execution_triggers.py
    sync: {policy: FIRST_AT_OR_AFTER_BOUNDARY | FIRST_ALIGNED_AT_OR_AFTER_BOUNDARY, policy_version: 1,
           reference?: ..., tolerance_ms?: ...,
           timeout_ms?: 1000}
    writer: {queue_size?: 8, max_pending?: 4, buffer_frames?: 8}
    deployment?: {host_id?: <logical id>}          # DECLARED id of the recorder's host (its facts are OBSERVED)
    provenance?:                                   # NOT in a replay configuration (a replay inherits its original's)
      description: {package: <pkg>, file: <xacro / urdf in its share>, args?: {name: value}}
                                                   # the ROOT robot_state_publisher expands: the recorder hashes it
                                                   # with its whole include closure (xacro) at session start
      scene: {package: <pkg>, file: <stage file in its share>, prim?: <camera prim>}
                                                   # the simulator stage the operator is told to open (declared)
    extensions?: {...}

The store root is NOT here (``TRAINIT_DATA_ROOT`` at launch); TSA-specific concepts are not here either, and
neither is a dataset id: a session is an input to a dataset version, it does not own dataset identity
(ADR-0029 Decision 5); its ``rights`` are per session. MANUAL and INTERVAL since D0.2; EVENT and
APPLICATION_STATE since D0.5, from the runner's ExecutionEvent only (``execution_triggers.py``).

Camera topics are CELL-GLOBAL (ADR-0023 Decision 4): the namespace and every topic override are absolute, so
nothing resolves silently inside the recorder's robot slot. ``replay`` is the one backend resolved specially:
it is a data-source kind, not a device -- it inherits its original session's descriptor and extrinsic
provenance (the node reads them from the store).
"""
import copy

from trainit_data import _strict as s          # same repository, same version line (ADR-0029 Decision 1)
from trainit_data.errors import INVALID_VALUE, SOURCE_DESCRIPTOR_INVALID, DataError

from .contract import CAMERA_STREAMS, SOURCE_INFO, join
from .execution_triggers import trigger_fields

SCHEMA_NAME = 'trainit-recording'
SCHEMA_MAJOR = 1
IMPLEMENTED_TRIGGERS = ('MANUAL', 'INTERVAL', 'EVENT', 'APPLICATION_STATE')
IMPLEMENTED_POLICIES = {('FIRST_AT_OR_AFTER_BOUNDARY', 1), ('FIRST_ALIGNED_AT_OR_AFTER_BOUNDARY', 1)}
DEFAULT_POLICY = ('FIRST_AT_OR_AFTER_BOUNDARY', 1)      # what an authoring tool proposes by default
CLOCK_MISMATCH_MS = 1000            # the M3 DetectObject rule: a sample > 1 s after the boundary
DEFAULTS = {'timeout_ms': 1000, 'queue_size': 8, 'max_pending': 4, 'buffer_frames': 8,
            'extrinsic_provenance': 'NOMINAL_DESCRIPTION'}
# Upper bounds: every pending ~/capture may hold an executor thread, and buffers hold raw images.
LIMITS = {'queue_size': 64, 'max_pending': 16, 'buffer_frames': 64}


def _absolute(value, where):
    value = s.token(value, where)
    if not value.startswith('/'):
        s.fail(INVALID_VALUE, f'{value!r}: camera names are cell-global, give the absolute name (ADR-0023)', where)
    return value


def _source(vocab):
    def check(value, where):
        f = s.Fields(value, where)
        out = {'backend': f.req('backend', vocab.checker('backends'))}
        s.put(out, 'replay_of', f.opt('replay_of', s.session_id))
        s.put(out, 'descriptor', f.opt('descriptor', s.opaque))
        f.done()
        if (out['backend'] == 'replay') != ('replay_of' in out):
            s.fail(INVALID_VALUE, 'replay_of is given exactly when the backend is replay', where)
        return out
    return check


def _stream(value, where):
    f = s.Fields(value, where)
    out = {}
    for name in ('topic', 'info'):
        s.put(out, name, f.opt(name, _absolute))
    s.put(out, 'expected_frame', f.opt('expected_frame', s.token))
    f.done()
    return out


def _source_info(value, where):
    f = s.Fields(value, where)
    out = {}
    s.put(out, 'topic', f.opt('topic', _absolute))
    s.put(out, 'required', f.opt('required', s.boolean))
    f.done()
    return out


def _deployment(value, where):
    f = s.Fields(value, where)
    out = {}
    s.put(out, 'host_id', f.opt('host_id', s.ident))
    f.done()
    return out


def _sensor(vocab):
    def check(value, where):
        f = s.Fields(value, where)
        out = {'sensor_id': f.req('sensor_id', s.ident), 'profile': f.req('profile', s.token)}
        s.put(out, 'namespace', f.opt('namespace', _absolute))
        streams = f.req('streams', s.opaque)
        if not streams:
            s.fail(INVALID_VALUE, 'at least one stream', f'{where}.streams')
        out['streams'] = {}
        for name, block in streams.items():
            if name not in CAMERA_STREAMS:
                s.fail(INVALID_VALUE, f'{name!r} is not a camera-contract stream ({sorted(CAMERA_STREAMS)})',
                       f'{where}.streams')
            out['streams'][name] = _stream(block or {}, f'{where}.streams.{name}')
        s.put(out, 'source_info', f.opt('source_info', _source_info))
        s.put(out, 'extrinsic_parent_frame', f.opt('extrinsic_parent_frame', s.token))
        s.put(out, 'extrinsic_provenance', f.opt('extrinsic_provenance', vocab.checker('extrinsic_provenance')))
        s.put(out, 'pose_parent_frame', f.opt('pose_parent_frame', s.token))
        f.done()
        return out
    return check


def _trigger(vocab):
    def check(value, where):
        f = s.Fields(value, where)
        out = {'id': f.req('id', s.ident), 'kind': f.req('kind', vocab.checker('trigger_kinds'))}
        s.put(out, 'period_ms', f.opt('period_ms', s.integer(1)))
        s.put(out, 'max_captures', f.opt('max_captures', s.integer(1)))
        if out['kind'] not in IMPLEMENTED_TRIGGERS:
            s.fail(INVALID_VALUE, f'trigger kind {out["kind"]} is not implemented: {list(IMPLEMENTED_TRIGGERS)}',
                   where)
        trigger_fields(out, f, where)
        f.done()
        if (out['kind'] == 'INTERVAL') != ('period_ms' in out):
            s.fail(INVALID_VALUE, 'period_ms is given exactly for an INTERVAL trigger', where)
        if 'max_captures' in out and out['kind'] != 'INTERVAL':
            s.fail(INVALID_VALUE, 'max_captures applies to an INTERVAL trigger', where)
        return out
    return check


def _sync(value, where):
    f = s.Fields(value, where)
    out = {'policy': f.req('policy', s.token), 'policy_version': f.req('policy_version', s.integer(1))}
    s.put(out, 'reference', f.opt('reference', s.token))
    s.put(out, 'tolerance_ms', f.opt('tolerance_ms', s.positive_real))
    s.put(out, 'timeout_ms', f.opt('timeout_ms', s.integer(1)))
    f.done()
    return out


def _package_file(*extra):
    def check(value, where):
        f = s.Fields(value, where)
        out = {'package': f.req('package', s.ident), 'file': f.req('file', s.relative_path)}
        if 'args' in extra:
            s.put(out, 'args', f.opt('args', s.str_map))
        if 'prim' in extra:
            s.put(out, 'prim', f.opt('prim', s.token))
        f.done()
        return out
    return check


def _provenance(value, where):
    f = s.Fields(value, where)
    out = {}
    s.put(out, 'description', f.opt('description', _package_file('args')))
    s.put(out, 'scene', f.opt('scene', _package_file('prim')))
    f.done()
    return out


def _writer(value, where):
    f = s.Fields(value, where)
    out = {}
    for name in ('queue_size', 'max_pending', 'buffer_frames'):
        value = f.opt(name, s.integer(1))
        if value is not None and value > LIMITS[name]:
            s.fail(INVALID_VALUE, f'at most {LIMITS[name]}', f'{where}.{name}')
        s.put(out, name, value)
    f.done()
    return out


def _clock(vocab):
    def check(value, where):
        f = s.Fields(value, where)
        out = {}
        s.put(out, 'sim_source', f.opt('sim_source', vocab.checker('sim_sources')))
        f.done()
        return out
    return check


def _rights(value, where):
    from trainit_data.session_record import CLASSIFICATIONS
    f = s.Fields(value, where)
    out = {'classification': f.req('classification', s.one_of(CLASSIFICATIONS))}
    for name in ('owner_org', 'license', 'notes'):
        s.put(out, name, f.opt(name, s.string))
    f.done()
    return out


def parse(document, vocab) -> dict:
    """The strict, as-written configuration (no defaults)."""
    f = s.Fields(document, 'recording')
    out = {'schema': f.req('schema', s.schema(SCHEMA_NAME, SCHEMA_MAJOR)),
           'source': f.req('source', _source(vocab)),
           'clock': f.opt('clock', _clock(vocab), {}),
           'rights': f.opt('rights', _rights, {'classification': 'INTERNAL'}),
           'sensors': list(f.req('sensors', s.list_of(_sensor(vocab), min_length=1))),
           'triggers': list(f.req('triggers', s.list_of(_trigger(vocab), min_length=1))),
           'sync': f.req('sync', _sync),
           'writer': f.opt('writer', _writer, {}),
           'deployment': f.opt('deployment', _deployment, {}),
           'provenance': f.opt('provenance', _provenance, {})}
    extensions = f.extensions()
    if extensions:
        out['extensions'] = extensions
    f.done()
    return out


def _descriptor(raw, backend, vocab, where, supplied_by):
    if not isinstance(raw, dict):
        raise DataError(SOURCE_DESCRIPTOR_INVALID, f'{where}: no source descriptor for backend {backend!r} '
                        '(declare it in the hardware profile or in source.descriptor)', context={'where': where})
    try:
        f = s.Fields(dict(raw), where)
        declared = f.opt('backend', s.token)
        out = {'intrinsics_kind': f.req('intrinsics_kind', vocab.checker('intrinsic_provenance')),
               'stamp_origin': f.req('stamp_origin', vocab.checker('stamp_origins')),
               'registration_basis': f.req('registration_basis', s.token)}
        s.put(out, 'device', f.opt('device', s.str_map))
        out['supplied_by'] = f.opt('supplied_by', s.string, supplied_by)
        # Facts that belong elsewhere than the session descriptor: the clock's simulator, the intrinsics' note.
        extras = {}
        s.put(extras, 'sim_source', f.opt('sim_source', vocab.checker('sim_sources')))
        s.put(extras, 'intrinsics_notes', f.opt('intrinsics_notes', s.string))
        f.done()
    except DataError as exc:
        raise DataError(SOURCE_DESCRIPTOR_INVALID, f'{where}: {exc.message}', context=exc.context) from None
    if declared is not None and declared != backend:
        raise DataError(SOURCE_DESCRIPTOR_INVALID, f'{where}: the descriptor is for backend {declared!r}, the '
                        f'source is {backend!r}', context={'where': where})
    return out, extras


def resolve(document, vocab, *, load_profile, use_sim_time: bool, replay_descriptor=None,
            replay_extrinsic_provenance=None) -> dict:
    """The EFFECTIVE configuration: every default filled, validated, JSON-plain (persisted in the session).

    ``load_profile(model)`` returns a hardware camera profile dict; ``replay_descriptor`` and
    ``replay_extrinsic_provenance`` (``{sensor_id: kind}``) are what a replay inherits from its original session
    (read by the caller from the store).
    """
    cfg = parse(document, vocab)
    backend = cfg['source']['backend']
    resolved = {'schema': cfg['schema'], 'rights': cfg['rights']}

    sensors, fps, descriptors = [], [], []
    ids = [d['sensor_id'] for d in cfg['sensors']]
    if len(set(ids)) != len(ids):
        s.fail(INVALID_VALUE, 'a sensor is listed twice', 'recording.sensors')
    for i, sensor in enumerate(cfg['sensors']):
        where = f'recording.sensors[{i}]'
        profile = load_profile(sensor['profile'])
        namespace = sensor.get('namespace') or _absolute(profile['topics']['namespace'], f'{where} profile namespace')
        streams = {}
        for name, override in sensor['streams'].items():
            topic, info, modality, frame_key, registered_to = CAMERA_STREAMS[name]
            vocab.require('modalities', modality, f'{where}.streams.{name}')
            stream = {'topic': override.get('topic', join(namespace, topic)),
                      'info': override.get('info', join(namespace, info)),
                      'modality': modality,
                      'expected_frame': override.get('expected_frame', profile['frames'][frame_key]),
                      'fps': float(profile['sensor'][name]['fps'])}
            if registered_to is not None:
                if registered_to not in sensor['streams']:
                    s.fail(INVALID_VALUE, f'{name} is registered to {registered_to}: record both streams', where)
                stream['registered_to'] = registered_to
            fps.append(stream['fps'])
            streams[name] = stream
        for name, stream in streams.items():
            other = streams.get(stream.get('registered_to'))
            if other is not None and stream['expected_frame'] != other['expected_frame']:
                s.fail(INVALID_VALUE, f'{name} is registered to {stream["registered_to"]}: it is expressed in the same '
                       f'optical frame ({other["expected_frame"]!r}), contracts/camera.md "Registered depth"', where)
        inherited = (replay_extrinsic_provenance or {}).get(sensor['sensor_id']) if backend == 'replay' else None
        declared_info = sensor.get('source_info', {})
        entry = {'sensor_id': sensor['sensor_id'], 'profile': sensor['profile'], 'namespace': namespace,
                 'streams': streams,
                 'source_info': {'topic': declared_info.get('topic', join(namespace, SOURCE_INFO)),
                                 'required': declared_info.get('required', False)},
                 'extrinsic_provenance': sensor.get('extrinsic_provenance',
                                                    inherited or DEFAULTS['extrinsic_provenance'])}
        for name in ('extrinsic_parent_frame', 'pose_parent_frame'):
            s.put(entry, name, sensor.get(name))
        sensors.append(entry)
        if 'descriptor' not in cfg['source'] and backend != 'replay':
            raw = ((profile.get('backends') or {}).get(backend) or {}).get('source_descriptor')
            descriptors.append(_descriptor(raw, backend, vocab, f'{where} profile {sensor["profile"]}',
                                           f'trainit_hardware_layer profile {sensor["profile"]} backends.{backend}'))
    resolved['sensors'] = sensors

    if 'descriptor' in cfg['source']:
        descriptor, extras = _descriptor(cfg['source']['descriptor'], backend, vocab, 'recording.source.descriptor',
                                         'recording configuration')
    elif backend == 'replay':
        descriptor, extras = _descriptor(replay_descriptor, backend, vocab, 'recording.source (replay)',
                                         'trainit_data_ros replay')
    else:
        descriptor, extras = descriptors[0]
        if any(d != descriptors[0] for d in descriptors):
            raise DataError(SOURCE_DESCRIPTOR_INVALID, 'the sensors of one session declare different source '
                            'descriptors', context={'descriptors': [d[0] for d in descriptors]})
    declared_sim = extras.get('sim_source')
    if use_sim_time:
        sim_source = cfg['clock'].get('sim_source', declared_sim)
        if sim_source is None:
            s.fail(INVALID_VALUE, 'the node runs on use_sim_time: clock.sim_source (or the backend descriptor\'s '
                   'sim_source) names the simulator', 'recording.clock')
        if declared_sim is not None and sim_source != declared_sim:
            s.fail(INVALID_VALUE, f'clock.sim_source {sim_source!r} contradicts the backend descriptor '
                   f'({declared_sim!r})', 'recording.clock')
        resolved['clock'] = {'session_domain': 'SIM_TIME', 'sim_source': sim_source}
    else:
        if cfg['clock']:
            s.fail(INVALID_VALUE, 'clock.sim_source applies only on use_sim_time', 'recording.clock')
        resolved['clock'] = {'session_domain': 'ROS_TIME'}
    source = {'backend': backend, 'transport': 'LIVE_ROS', 'descriptor': descriptor}
    s.put(source, 'replay_of', cfg['source'].get('replay_of'))
    s.put(source, 'intrinsics_notes', extras.get('intrinsics_notes'))     # copied into each calibration record
    resolved['source'] = source

    trigger_ids = [t['id'] for t in cfg['triggers']]
    if len(set(trigger_ids)) != len(trigger_ids):
        s.fail(INVALID_VALUE, 'a trigger id is used twice', 'recording.triggers')
    if sum(1 for t in cfg['triggers'] if t['kind'] == 'MANUAL') > 1:
        s.fail(INVALID_VALUE, 'at most one MANUAL trigger (the ~/capture service)', 'recording.triggers')
    resolved['triggers'] = cfg['triggers']

    sync = cfg['sync']
    vocab.require_sync_policy(sync['policy'], sync['policy_version'], 'recording.sync.policy')
    if (sync['policy'], sync['policy_version']) not in IMPLEMENTED_POLICIES:
        s.fail(INVALID_VALUE, f'sync policy {sync["policy"]} v{sync["policy_version"]} has no implementation in '
               'this recorder', 'recording.sync')
    keys = [f'{d["sensor_id"]}/{name}' for d in sensors for name in d['streams']]
    reference = sync.get('reference', keys[0])
    if reference not in keys:
        s.fail(INVALID_VALUE, f'reference {reference!r} is not a recorded stream ({keys})', 'recording.sync')
    if 'tolerance_ms' in sync:
        tolerance, tolerance_source = sync['tolerance_ms'], 'configured'
    else:
        slowest = min(fps)
        tolerance, tolerance_source = 1000.0 / slowest / 2.0, f'half the frame period of the slowest stream ' \
                                                              f'({slowest:g} fps, hardware profile)'
    resolved['sync'] = {'policy': sync['policy'], 'policy_version': sync['policy_version'], 'reference': reference,
                        'tolerance_ms': tolerance, 'tolerance_source': tolerance_source,
                        'timeout_ms': sync.get('timeout_ms', DEFAULTS['timeout_ms']),
                        'clock_mismatch_ms': CLOCK_MISMATCH_MS}
    resolved['writer'] = {name: cfg['writer'].get(name, DEFAULTS[name])
                          for name in ('queue_size', 'max_pending', 'buffer_frames')}
    if cfg['deployment']:
        resolved['deployment'] = cfg['deployment']
    if cfg['provenance']:
        if backend == 'replay':
            s.fail(INVALID_VALUE, 'a replay inherits its original session\'s provenance; it declares none',
                   'recording.provenance')
        resolved['provenance'] = cfg['provenance']
    if 'extensions' in cfg:
        resolved['extensions'] = cfg['extensions']
    return copy.deepcopy(s.json_value(resolved, 'recording'))


def load_yaml(path):
    """(document, file bytes) of a recording configuration file."""
    import yaml
    data = open(path, 'rb').read()
    try:
        document = yaml.safe_load(data)
    except yaml.YAMLError as exc:
        raise DataError(INVALID_VALUE, f'{path}: not valid YAML', context={'path': str(path)}, cause=exc) from None
    return document, data
