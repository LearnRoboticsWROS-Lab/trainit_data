"""The D0 recorder (the CaptureCoordinator's ROS node): live camera contract -> canonical TrainIt captures.

Runs in the robot / application namespace slot (ADR-0023): its services and its diagnostics topic are
RELATIVE names, resolved in the slot. The camera topics are the deployed camera contract's (cell-global,
ADR-0023 Decision 4), taken from the recording configuration and the hardware profile -- this node invents
no topic convention. /tf and /tf_static are the standard cell-global transforms. Each camera's OPTIONAL latched
``<ns>/source_info`` document (contract) is kept as received and, at session start, checked against the source
descriptor and recorded with the recorder's own host and transport facts (``acquisition``, D0.4).
With an EVENT or APPLICATION_STATE trigger configured (D0.5) it subscribes to the runner's
``trainit_interfaces/ExecutionEvent`` on the RELATIVE ``trainit/execution_events`` of its slot (RELIABLE, VOLATILE,
keep-last 100) -- never to a process topic -- and fires on the events of the ACTIVE session only
(``execution_triggers.EventGate``), the capture boundary being the event's own stamp.

Parameters: ``config_file`` (a ``trainit-recording/1.0`` YAML, required), ``store_root`` (default:
``$TRAINIT_DATA_ROOT``, else ``~/TrainItData``), ``use_sim_time``.

Services (``std_srvs/srv/Trigger``, the surface ADR-0029 Decision 7 allows TSA to call):
- ``~/start_session``: open an ACTIVE session (INTERVAL triggers start with it);
- ``~/capture``: fire the MANUAL trigger; answers once the capture is written (or refused / dropped);
- ``~/close_session``: stop the triggers, let pending captures finish (bounded), drain the writer, seal.

Threads (none of them blocks a sensor callback):
- the executor: sensor callbacks only buffer the raw message (cheap); services;
- the coordinator thread: the INTERVAL schedule on the SESSION clock (boundaries ``start + k * period``;
  a paused simulation pauses it, by design; rclpy timers are not used because Humble's multi-threaded
  executor can run one twice in a row), the sync policy, conversion, validation, TF lookups at the
  observation stamp (non-blocking ``can_transform`` polled until the host-monotonic deadline);
- the TF thread: a dedicated listener node with its own executor;
- the writer thread (``WriterWorker``): every store write, through a bounded queue.

Diagnostics (``diagnostic_msgs/DiagnosticArray`` on ``diagnostics``, 1 Hz) are observability only; the
session files are the source of truth.
"""
import json
import math
import os
import resource
import threading
import time
from pathlib import Path

import rclpy
from diagnostic_msgs.msg import DiagnosticArray, DiagnosticStatus, KeyValue
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup, ReentrantCallbackGroup
from rclpy.duration import Duration
from rclpy.executors import MultiThreadedExecutor, SingleThreadedExecutor
from rclpy.node import Node
from rclpy.qos import QoSDurabilityPolicy, QoSHistoryPolicy, QoSProfile, QoSReliabilityPolicy
from rclpy.time import Time
from sensor_msgs.msg import CameraInfo, Image
from std_msgs.msg import String
from std_srvs.srv import Trigger

from trainit_data._canonical import sha256_bytes, sha256_json
from trainit_data.errors import INVALID_VALUE, DataError
from trainit_data.store import Store
from trainit_data.vocab import Vocabulary

from . import __version__, acquisition, conversions, recording_config
from . import provenance as provenance_mod
from . import execution_triggers
from .contract import EXECUTION_EVENTS
from .coordinator import Coordinator, Info, Overrun, Sample
from .writer_worker import WriterWorker

SENSOR_QOS = QoSProfile(reliability=QoSReliabilityPolicy.BEST_EFFORT, history=QoSHistoryPolicy.KEEP_LAST, depth=5)
INFO_QOS = QoSProfile(reliability=QoSReliabilityPolicy.RELIABLE, history=QoSHistoryPolicy.KEEP_LAST, depth=5)
# The camera's latched source document (camera.md, "Source information"): a late joiner still receives it.
SOURCE_INFO_QOS = QoSProfile(reliability=QoSReliabilityPolicy.RELIABLE, durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
                             history=QoSHistoryPolicy.KEEP_LAST, depth=1)
# D0.5: events, never state -- a recorder that subscribes later never receives an event of the past
EVENT_QOS = QoSProfile(reliability=QoSReliabilityPolicy.RELIABLE, durability=QoSDurabilityPolicy.VOLATILE,
                       history=QoSHistoryPolicy.KEEP_LAST, depth=100)
WRITE_WAIT_S = 30.0


def _package_version(name: str):
    try:
        from ament_index_python.packages import get_package_share_directory
        text = (Path(get_package_share_directory(name)) / 'package.xml').read_text()
        return text.split('<version>')[1].split('</version>')[0].strip()
    except Exception:  # noqa: BLE001 -- provenance is best effort: absent, never invented
        return None


def _load_profile(model: str) -> dict:
    from trainit_hardware_layer import profiles
    return profiles.load('camera', model)


class Recorder(Node):
    def __init__(self, **node_kwargs):
        super().__init__('trainit_data_recorder', **node_kwargs)
        self.declare_parameter('config_file', '')
        self.declare_parameter('store_root', '')
        self.vocab = Vocabulary.default()
        config_file = self.get_parameter('config_file').value
        if not config_file:
            raise DataError(INVALID_VALUE, 'parameter config_file (a trainit-recording/1.0 file) is required')
        self.store_root = self.get_parameter('store_root').value or None
        document, raw = recording_config.load_yaml(config_file)
        self.config_sha256 = sha256_bytes(raw)
        self.use_sim_time = bool(self.get_parameter('use_sim_time').value)
        replay_descriptor, replay_extrinsics, replay_description, replay_original = \
            self._replay_inheritance(document)
        self.cfg = recording_config.resolve(document, self.vocab, load_profile=_load_profile,
                                            use_sim_time=self.use_sim_time, replay_descriptor=replay_descriptor,
                                            replay_extrinsic_provenance=replay_extrinsics)
        self.profiles = {s['profile']: _load_profile(s['profile']) for s in self.cfg['sensors']}
        self._replay_description = replay_description
        self._replay_original = replay_original
        self._source_info = {}          # sensor id -> the latest source document (raw String data)
        self._session_source_info = None  # sensor id -> the raw document a starting / ACTIVE session recorded
        self._source_info_lock = threading.Lock()
        self.source_info_changes = 0

        self._tf_buffer, self._tf_node, self._tf_executor, self._tf_thread = self._start_tf(node_kwargs)
        self.coordinator = Coordinator(self.cfg, self.vocab, tf_lookup=self._tf_lookup)
        self.state = 'IDLE'
        self.session = None             # the SessionWriter
        self.worker = None
        self._fire_lock = threading.Lock()
        self._clock_lock = threading.Lock()
        self._session_lock = threading.Lock()
        self._intervals = []            # [{'cfg', 'due_ns'}] while ACTIVE
        self._last_clock_ns = None
        self.callback_errors = 0
        self.last_error = None
        self._reset_session_counters()
        self.session_provenance = {}

        sensors = ReentrantCallbackGroup()
        for sensor in self.cfg['sensors']:
            for name, block in sensor['streams'].items():
                key = f'{sensor["sensor_id"]}/{name}'
                self.create_subscription(Image, block['topic'], lambda m, k=key: self._on_image(k, m), SENSOR_QOS,
                                         callback_group=sensors)
                self.create_subscription(CameraInfo, block['info'], lambda m, k=key: self._on_info(k, m), INFO_QOS,
                                         callback_group=sensors)
            sid = sensor['sensor_id']
            self.create_subscription(String, sensor['source_info']['topic'],
                                     lambda m, i=sid: self._on_source_info(i, m), SOURCE_INFO_QOS,
                                     callback_group=sensors)
        services = ReentrantCallbackGroup()
        self.create_service(Trigger, '~/start_session', self._start_session, callback_group=services)
        self.create_service(Trigger, '~/capture', self._capture, callback_group=services)
        self.create_service(Trigger, '~/close_session', self._close_session, callback_group=services)
        self._execution_triggers = [t for t in self.cfg['triggers']
                                    if t['kind'] in execution_triggers.EXECUTION_TRIGGERS]
        self._event_gate = None         # the ACTIVE session's EventGate
        self._closing_gate = None       # the gate of the session being sealed (its counts become warnings)
        self._event_lock = threading.Lock()
        self.events_outside_session = 0
        if self._execution_triggers:
            from trainit_interfaces.msg import ExecutionEvent
            self.create_subscription(ExecutionEvent, EXECUTION_EVENTS, self._on_execution_event, EVENT_QOS,
                                     callback_group=MutuallyExclusiveCallbackGroup())
        self._diag = self.create_publisher(DiagnosticArray, 'diagnostics', 10)
        self.create_timer(1.0, self._publish_diagnostics, callback_group=MutuallyExclusiveCallbackGroup())

        self._running = True
        self._coord_thread = threading.Thread(target=self._coordinate, name='trainit_data_coordinator', daemon=True)
        self._coord_thread.start()
        self.get_logger().info(
            f'recorder ready: {len(self.coordinator.streams)} stream(s) {sorted(self.coordinator.streams)}, '
            f'backend {self.cfg["source"]["backend"]}, clock {self.cfg["clock"]["session_domain"]}, '
            f'policy {self.cfg["sync"]["policy"]} v{self.cfg["sync"]["policy_version"]}')

    def _reset_session_counters(self):
        self.firings = {t['id']: 0 for t in self.cfg['triggers']}
        self.assembled = 0
        self.outcomes = {'complete': 0, 'partial': 0, 'failed': 0}     # COMMITTED captures only
        self.trigger_overruns = self.sync_failures = self.interval_slips = self.clock_jumps = 0
        self.last_skew_ns = None
        self.session_started_mono = None

    # ---- configuration helpers -------------------------------------------------------------------------

    def _store(self, create: bool) -> Store:
        return Store.open(self.store_root, create=create)

    def _replay_inheritance(self, document):
        """A replay inherits its original session's source descriptor (re-stamping declared), per sensor the
        provenance kind of the original extrinsics, and the description provenance -- so identical content keeps
        identical calibration ids."""
        source = (document or {}).get('source') or {}
        if source.get('backend') != 'replay' or 'replay_of' not in source:
            return None, None, None, None
        store = self._store(create=False)
        original = store.open_session(source['replay_of'])
        descriptor = None
        if 'descriptor' not in source:
            descriptor = dict(original.header.source.descriptor)
            descriptor['stamp_origin'] = 'HEADER'
            descriptor['supplied_by'] = ('trainit_data_ros replay of ' + original.session_id +
                                         ' (re-stamped at publish, original intra-capture offsets kept)')
            notes = ((original.header.recording_config or {}).get('resolved', {}).get('source', {})
                     .get('intrinsics_notes'))
            if notes is not None:
                descriptor['intrinsics_notes'] = notes      # the original's fact about its intrinsics
        extrinsics = {}
        for cal_id in original.header.calibration_ids:
            record = store.get_calibration(cal_id, original.vocab)
            if record.extrinsics:
                extrinsics.setdefault(record.sensor_id, record.extrinsics[0].provenance.kind)
        description = ((original.header.provenance or {}).get('robot') or {}).get('description')
        original_acquisition = {'session_id': original.session_id}
        if 'acquisition' in (original.header.provenance or {}):
            original_acquisition['acquisition'] = original.header.provenance['acquisition']
        return descriptor, extrinsics, description, original_acquisition

    def _start_tf(self, node_kwargs):
        from tf2_ros.buffer import Buffer
        from tf2_ros.transform_listener import TransformListener
        from rclpy.parameter import Parameter
        context = node_kwargs.get('context')
        # A launch's node-name remap (`__node:=`) is a GLOBAL argument of the process: it would rename this node
        # too (two nodes of one name). A node-local remap wins over it; every other global argument still applies.
        name = f'{self.get_name()}_tf'
        tf_node = rclpy.create_node(name, context=context, namespace=self.get_namespace(),
                                    cli_args=['--ros-args', '-r', f'__node:={name}'],
                                    start_parameter_services=False,
                                    parameter_overrides=[Parameter('use_sim_time', value=self.use_sim_time)])
        buffer = Buffer(cache_time=Duration(seconds=30.0))
        TransformListener(buffer, tf_node)
        executor = SingleThreadedExecutor(context=context)
        executor.add_node(tf_node)
        thread = threading.Thread(target=self._spin_tf, args=(executor,), name='trainit_data_tf', daemon=True)
        thread.start()
        return buffer, tf_node, executor, thread

    def _spin_tf(self, executor):
        try:
            executor.spin()
        except Exception as exc:  # noqa: BLE001 -- lookups then answer "not available" (TF_UNAVAILABLE), honestly
            if getattr(self, '_running', True):
                self.last_error = f'TF listener stopped: {exc!r}'

    def _tf_lookup(self, parent, child, sec, nanosec):
        """The transform AT the observation stamp, or None (never 'latest'; never blocks)."""
        when = Time(seconds=sec, nanoseconds=nanosec)
        try:
            if not self._tf_buffer.can_transform(parent, child, when, timeout=Duration(seconds=0)):
                return None
            t = self._tf_buffer.lookup_transform(parent, child, when).transform
        except Exception:  # noqa: BLE001 -- extrapolation / connectivity: "not available", decided by the deadline
            return None
        return ((t.translation.x, t.translation.y, t.translation.z),
                (t.rotation.x, t.rotation.y, t.rotation.z, t.rotation.w))

    # ---- callbacks: buffer only ---------------------------------------------------------------------------

    # A malformed message must never raise out of a callback: in a multi-threaded executor that can stop the
    # spin. It is counted (diagnostics) and the stream simply has no usable sample / CameraInfo.

    def _on_image(self, key, msg):
        try:
            self.coordinator.on_sample(key, Sample(
                sec=msg.header.stamp.sec, nanosec=msg.header.stamp.nanosec, clock=self.coordinator.clock_key,
                receipt_mono_ns=time.monotonic_ns(), receipt_wall_ns=time.time_ns(), frame_id=msg.header.frame_id,
                encoding=msg.encoding, msg=msg))
        except Exception as exc:  # noqa: BLE001
            self._callback_error(key, exc)

    def _on_info(self, key, msg):
        try:
            block = conversions.intrinsics(msg)
            if not all(math.isfinite(v) for name in ('K', 'D', 'R', 'P') for v in block[name]):
                raise ValueError('non-finite value in K / D / R / P: CameraInfo rejected (the previous one stays)')
            stamp_ns = msg.header.stamp.sec * 1_000_000_000 + msg.header.stamp.nanosec
            self.coordinator.on_info(key, Info(block, msg.header.frame_id, stamp_ns))
        except Exception as exc:  # noqa: BLE001
            self._callback_error(key, exc)

    def _on_source_info(self, sensor_id, msg):
        # Kept raw: it is parsed and checked at session start, never in a callback. From the moment a session takes
        # its snapshot (before it is even written) a different document is counted -- the session recorded the one
        # valid at its start -- and reported at close.
        with self._source_info_lock:
            self._source_info[sensor_id] = msg.data
            if self._session_source_info is not None and self._session_source_info.get(sensor_id) != msg.data:
                self.source_info_changes += 1

    def _snapshot_source_info(self) -> dict:
        """The documents a starting session records, and the start of change counting. A document whose topic has
        NO publisher any more is stale (a latched message outlives nothing): it is dropped, never recorded as current
        evidence."""
        live = {s['sensor_id']: self.count_publishers(s['source_info']['topic']) > 0 for s in self.cfg['sensors']}
        with self._source_info_lock:
            for sensor_id, alive in live.items():
                if not alive:
                    self._source_info.pop(sensor_id, None)
            snapshot = dict(self._source_info)
            self._session_source_info = snapshot
            self.source_info_changes = 0
        return snapshot

    def _stop_source_info_watch(self):
        with self._source_info_lock:
            self._session_source_info = None

    def _transport(self) -> dict:
        """How the camera contract reaches this recorder: OBSERVED from this process."""
        out = {'origin': 'OBSERVED', 'kind': 'ROS2_DDS',
               'rmw': rclpy.utilities.get_rmw_implementation_identifier()}
        try:
            out['ros_domain_id'] = int(self.context.get_domain_id())
        except Exception:  # noqa: BLE001 -- absent, never invented
            pass
        return out

    def _callback_error(self, key, exc):
        self.callback_errors += 1
        self.last_error = f'{key}: {exc!r}'

    # ---- the coordinator thread -----------------------------------------------------------------------------

    def _coordinate(self):
        while self._running:
            try:
                self._watch_clock(self.get_clock().now().nanoseconds)
                self._run_intervals()
                for draft in self.coordinator.poll():
                    self.assembled += 1
                    worker = self.worker
                    if worker is None:
                        draft.pending.result = 'DROPPED: no active session'
                        draft.pending.done.set()
                    else:
                        worker.submit(draft)
            except Exception as exc:  # noqa: BLE001 -- never let the thread die silently
                self.last_error = f'coordinator: {exc!r}'
                self.get_logger().error(self.last_error)
            time.sleep(0.005)

    def _watch_clock(self, now: int):
        """A session clock going BACKWARDS (a simulation reset) invalidates buffered data, the TF history, the
        captures still waiting and the INTERVAL schedule. Called by the coordinator thread AND by every firing with
        the very reading it uses as the boundary, so no boundary is ever taken against an un-flushed old timeline."""
        with self._clock_lock:
            last = self._last_clock_ns
            if last is not None and now < last:
                stale = self.coordinator.on_clock_jump()
                try:
                    self._tf_buffer.clear()     # static transforms survive; the old dynamic history does not
                except Exception:  # noqa: BLE001
                    pass
                for entry in self._intervals:
                    entry['due_ns'] = None      # re-anchored at the next reading
                with self._event_lock:          # D0.5: events of the old timeline never fire
                    if self._event_gate is not None:
                        self._event_gate.rebase(now)
                if self.state == 'ACTIVE':
                    self.clock_jumps += 1
                self.get_logger().warn(f'the session clock jumped back {(last - now) / 1e9:.3f} s: buffers and TF '
                                       f'history dropped, {stale} pending capture(s) closed CLOCK_MISMATCH')
            if last is None or now != last:
                self._last_clock_ns = now

    def _on_written(self, draft, outcome):
        """The writer committed a capture: count it AS COMMITTED (a write failure may have downgraded it)."""
        self.outcomes[outcome.lower()] += 1
        if draft.skew_ns is not None:
            self.last_skew_ns = draft.skew_ns
        sync_reasons = {'TIMEOUT', 'CLOCK_UNAVAILABLE', 'CLOCK_MISMATCH', 'INCOMPATIBLE_CLOCKS'}
        if any(m['reason'] in sync_reasons for m in draft.missing) or (
                draft.sync and draft.sync['status'] in ('OUT_OF_TOLERANCE', 'INCOMPATIBLE_CLOCKS')):
            self.sync_failures += 1
        self.coordinator.written(draft)

    # ---- triggers --------------------------------------------------------------------------------------------

    def _fire(self, trigger_cfg, origin, reason, boundary_ns=None, context=None, stamp_origin='NODE_CLOCK',
              received=None):
        """``received``: (host monotonic, wall UTC) of an execution event's RECEIPT, recorded as ``issued``'s host
        times beside the EVENT's stamp (``stamp_origin`` HEADER); else this call's own times."""
        with self._fire_lock:
            if self.state != 'ACTIVE':
                return None
            if boundary_ns is None:
                boundary_ns = self.get_clock().now().nanoseconds
                self._watch_clock(boundary_ns)          # a reset is handled BEFORE this boundary is used
            sec, nanosec = divmod(boundary_ns, 1_000_000_000)
            domain = self.cfg['clock']
            stamp = {'sec': sec, 'nanosec': nanosec, 'clock_domain': domain['session_domain'],
                     'origin': stamp_origin}
            if 'sim_source' in domain:
                stamp['sim_source'] = domain['sim_source']
            index = self.firings[trigger_cfg['id']]
            self.firings[trigger_cfg['id']] += 1
            mono, wall = received or (time.monotonic_ns(), time.time_ns())
            trigger = {'trigger_id': trigger_cfg['id'], 'kind': trigger_cfg['kind'], 'firing_index': index,
                       'origin': origin, 'reason': reason,
                       'issued': {'stamp': stamp, 'host_monotonic_ns': mono, 'wall_utc_ns': wall}}
            if context:
                trigger['context'] = context
            try:
                return self.coordinator.fire(trigger, boundary_ns)
            except Overrun:
                self.trigger_overruns += 1
                self.get_logger().warn(f'trigger {trigger_cfg["id"]} dropped: too many captures pending (counted '
                                       'as a writer overrun)')
                return None

    def _on_execution_event(self, msg):
        """An execution event: admitted by the ACTIVE session's gate, then every matching trigger fires at the
        event's stamp. Never raises (a callback that raises stops the executor)."""
        received = (time.monotonic_ns(), time.time_ns())
        try:
            event = execution_triggers.event_from_message(msg)
            with self._event_lock:
                gate = self._event_gate
                if self.state != 'ACTIVE' or gate is None:
                    self.events_outside_session += 1
                    return
                refused = gate.admit(event, self.get_clock().now().nanoseconds)
            if refused is not None:
                self.get_logger().warn(f'execution event refused ({refused}): {event.get("kind")} run '
                                       f'{event.get("run_id")} seq {event.get("seq")}', throttle_duration_sec=5.0)
                return
            for trigger in self._execution_triggers:
                if execution_triggers.matches(trigger, event):
                    reason, context = execution_triggers.firing(trigger, event)
                    self._fire(trigger, 'EXECUTION_EVENT', reason, boundary_ns=event['stamp_ns'], context=context,
                               stamp_origin='HEADER', received=received)
        except Exception as exc:  # noqa: BLE001 -- counted, never raised
            self.callback_errors += 1
            self.last_error = f'execution event: {exc!r}'

    def _start_intervals(self):
        self._intervals = [{'cfg': cfg, 'due_ns': None} for cfg in self.cfg['triggers'] if cfg['kind'] == 'INTERVAL']

    def _run_intervals(self):
        """INTERVAL triggers on the session clock: boundary k = first clock reading + k * period. When late (a
        busy host, a clock jumping forward) only the LATEST due slot fires; the skipped ones are counted."""
        if self.state != 'ACTIVE' or not self._intervals:
            return
        now = self.get_clock().now().nanoseconds
        if now <= 0:
            return                      # the session clock is not running (sim time not yet published)
        for entry in list(self._intervals):
            cfg = entry['cfg']
            period = cfg['period_ms'] * 1_000_000
            if entry['due_ns'] is None:
                entry['due_ns'] = now + period
                continue
            if now < entry['due_ns'] or ('max_captures' in cfg and self.firings[cfg['id']] >= cfg['max_captures']):
                continue
            missed = (now - entry['due_ns']) // period          # whole periods past the due slot
            boundary = entry['due_ns'] + missed * period
            self.interval_slips += missed
            if self.state != 'ACTIVE':
                return
            self._fire(cfg, 'TIMER', {'period_ms': cfg['period_ms']}, boundary_ns=boundary)
            entry['due_ns'] = boundary + period

    # ---- services ---------------------------------------------------------------------------------------------

    def _start_session(self, request, response):
        with self._session_lock:
            if self.state != 'IDLE':
                response.success, response.message = False, f'refused: the recorder is {self.state}'
                return response
            try:
                received = self._snapshot_source_info()
                session_provenance = self._session_provenance({})          # software filled below
                session_provenance['acquisition'] = acquisition.build(
                    self.cfg, received, transport=self._transport(), original=self._replay_original,
                    recorder=acquisition.recorder_host((self.cfg.get('deployment') or {}).get('host_id')))
                store = self._store(create=True)
                sensors = [{'sensor_id': s['sensor_id'],
                            'profile': {'model': s['profile'], 'sha256': sha256_json(self.profiles[s['profile']])},
                            'streams': [{'stream': n, 'modality': b['modality'], 'topic': b['topic']}
                                        for n, b in s['streams'].items()]} for s in self.cfg['sensors']]
                software = {'trainit_data_ros': __version__}
                hw = _package_version('trainit_hardware_layer')
                if hw:
                    software['trainit_hardware_layer'] = hw
                source = {'transport': 'LIVE_ROS', 'backend': self.cfg['source']['backend'],
                          'descriptor': self.cfg['source']['descriptor']}
                if 'replay_of' in self.cfg['source']:
                    source['replay_of'] = self.cfg['source']['replay_of']
                clock = dict(self.cfg['clock'], use_sim_time=self.use_sim_time)
                session_provenance['software'] = software
                writer = store.create_session(
                    source=source, clock=clock, sensors=sensors, rights=self.cfg['rights'],
                    provenance=session_provenance,
                    recording_config={'resolved': self.cfg, 'file_sha256': self.config_sha256}, vocabulary=self.vocab)
            except Exception as exc:  # noqa: BLE001 -- answered, never raised out of the service
                self._stop_source_info_watch()
                text = f'{exc.code}: {exc.message}' if isinstance(exc, DataError) else repr(exc)
                response.success, response.message = False, f'refused: {text}'
                return response
            self.session_provenance = session_provenance
            for pending in self.coordinator.drain_pending():       # leftovers of a previous session
                pending.result = 'DROPPED: no active session'
                pending.done.set()
            self.coordinator.reset_counters()
            self.coordinator.description_sha256 = (self.session_provenance.get('robot', {})
                                                   .get('description', {}).get('sha256'))
            self.coordinator.bind_to_receipt(self._receipt_bound(session_provenance['acquisition']))
            self._reset_session_counters()
            self.session = writer
            self.worker = WriterWorker(store, writer, self.cfg['writer']['queue_size'], on_written=self._on_written)
            gate = execution_triggers.EventGate(          # D0.5: only events stamped from NOW on may fire
                self.get_clock().now().nanoseconds, int(self.cfg['sync']['clock_mismatch_ms'] * 1e6))
            self.session_started_mono = time.monotonic()
            self._cpu_start = resource.getrusage(resource.RUSAGE_SELF)
            with self._event_lock:          # the gate and ACTIVE together: no event falls in between
                self._event_gate = gate
                self.state = 'ACTIVE'
            self._start_intervals()
            response.success, response.message = True, writer.session_id
            self.get_logger().info(f'session {writer.session_id} ACTIVE in {store.root}')
            return response

    def _receipt_bound(self, acquisition_block):
        """Sensors whose stamps can never follow their receipt: proven on THIS host (the boot-id evidence) while
        the session clock is this host's ROS time. Remote, unknown (a replay, a source without a document) or
        simulated sources are never bound. It assumes a driver that stamps at or before publishing (a RealSense: the
        exposure); a driver stamping forward on purpose would need its own source descriptor first."""
        if self.cfg['clock']['session_domain'] != 'ROS_TIME' or self.use_sim_time:
            return set()
        return {sid for sid, entry in acquisition_block['sensors'].items()
                if entry.get('relation') == 'LOCAL_SENSOR_LOCAL_RECORDER'}

    def _session_provenance(self, software):
        """Hashed NOW (session start), from the installed artifacts the configuration names; a replay inherits
        its original session's description instead (it never hashes today's files)."""
        provenance = {'software': software}
        declared = self.cfg.get('provenance', {})
        description = self._replay_description
        if description is None and 'description' in declared:
            description = provenance_mod.description_provenance(declared['description'])
        if description is not None:
            provenance['robot'] = {'description': description}
        if 'scene' in declared:
            provenance['scene'] = {'sim_stage': provenance_mod.scene_provenance(declared['scene'])}
        return provenance

    def _capture(self, request, response):
        """MANUAL trigger. ``success`` = the capture was COMMITTED to the index (its outcome -- COMPLETE, PARTIAL
        or FAILED -- is the second word of ``message``); a dropped, refused or lost capture is ``success=False``."""
        manual = [t for t in self.cfg['triggers'] if t['kind'] == 'MANUAL']
        if self.state != 'ACTIVE' or not manual:
            response.success = False
            response.message = 'refused: no ACTIVE session' if self.state != 'ACTIVE' else \
                'refused: the configuration has no MANUAL trigger'
            return response
        pending = self._fire(manual[0], 'OPERATOR', {'caller': '~/capture'})
        if pending is None:
            response.success, response.message = False, 'DROPPED: too many captures pending (writer overrun)'
            return response
        if not pending.done.wait(self.coordinator.timeout_ns / 1e9 + WRITE_WAIT_S):
            response.success, response.message = False, 'not confirmed: the writer did not answer in time'
            return response
        result = pending.result
        if isinstance(result, DataError) or not hasattr(result, 'capture_id'):
            text = f'{result.code}: {result.message}' if isinstance(result, DataError) else str(result)
            response.success, response.message = False, text
            return response
        missing = [f'{m.stream}:{m.reason}' for m in result.missing]
        response.success = True
        response.message = f'{result.capture_id} {result.outcome}' + (f' missing={missing}' if missing else '')
        return response

    def _close_session(self, request, response):
        with self._session_lock:
            if self.state != 'ACTIVE':
                response.success, response.message = False, f'refused: the recorder is {self.state}'
                return response
            with self._fire_lock:
                self.state = 'CLOSING'          # no trigger fires after this point
            self._intervals = []
            with self._event_lock:
                self._closing_gate, self._event_gate = self._event_gate, None
            try:
                return self._seal(response)
            except Exception as exc:  # noqa: BLE001 -- never raised out of the service; the session stays ACTIVE
                self.session.release()
                response.success = False
                response.message = f'close failed: {exc!r} (the session stays ACTIVE: trainit-data recover)'
                return response
            finally:
                self.state, self.session, self.worker = 'IDLE', None, None
                self._stop_source_info_watch()

    def _seal(self, response):
        self.coordinator.expire_all()           # every pending capture is assembled now, never abandoned
        deadline = time.monotonic() + 5.0
        while self.coordinator.pending and time.monotonic() < deadline:
            time.sleep(0.01)
        worker = self.worker
        drained = worker.drain(WRITE_WAIT_S)
        stopped = worker.stop()
        performance = self.performance()
        if not (drained and stopped):
            self.session.release()
            response.success = False
            response.message = 'the writer did not finish in time; the session stays ACTIVE: trainit-data recover'
            return response
        warnings = self._session_warnings(worker)
        try:
            header = self.session.close(received=dict(self.coordinator.received),
                                        writer_overruns=worker.overruns + self.trigger_overruns,
                                        calibration_changes=self.coordinator.calibration_changes, warnings=warnings)
        except DataError as exc:
            self.session.release()
            response.success = False
            response.message = f'{exc.code}: {exc.message} (the session stays ACTIVE: trainit-data recover)'
            return response
        self.get_logger().info(f'session {header.session_id} CLOSED: {json.dumps(performance)}')
        response.success = True
        response.message = json.dumps({'session_id': header.session_id, 'counters': header.counters,
                                       'warnings': warnings, 'performance': performance}, sort_keys=True)
        return response

    def _session_warnings(self, worker) -> list:
        """What the files alone would not tell: persisted in the sealed header (not only in diagnostics)."""
        warnings = []
        if worker.lost:
            warnings.append({'code': 'CAPTURES_LOST', 'detail': f'{worker.lost} assembled capture(s) were refused or '
                             'failed before their index line', 'context': {
                                 'lost': worker.lost, 'last': repr(worker.last_loss)[:300]}})
        if self.trigger_overruns:
            warnings.append({'code': 'TRIGGER_OVERRUN', 'detail': f'{self.trigger_overruns} firing(s) dropped: too '
                             'many captures pending', 'context': {'count': self.trigger_overruns}})
        if self.interval_slips:
            warnings.append({'code': 'INTERVAL_SLIPS', 'detail': f'{self.interval_slips} INTERVAL slot(s) skipped '
                             '(the recorder or the clock ran late)', 'context': {'count': self.interval_slips}})
        if self.clock_jumps:
            warnings.append({'code': 'CLOCK_JUMP_BACKWARDS', 'detail': f'the session clock jumped back '
                             f'{self.clock_jumps} time(s); buffers were dropped',
                             'context': {'count': self.clock_jumps}})
        gate = self._closing_gate
        silent = sorted(t['id'] for t in self._execution_triggers if self.firings[t['id']] == 0)
        if silent:
            # a task / step / channel the application never reached -- or that does not exist: TSA cannot check
            # a recording configuration's ids before D0.8, so the session says it (and whether ANY event came)
            admitted = gate.admitted if gate is not None else 0
            why = ('no execution event of this session was admitted (is the runner TMR >= 2.7.0, in this slot, '
                   'started after the session?)' if not admitted else
                   'no matching event (check the task / step ids and the process channels)')
            warnings.append({'code': 'EXECUTION_TRIGGERS_NEVER_FIRED',
                             'detail': f'{len(silent)} execution trigger(s) never fired: {why}',
                             'context': {'triggers': silent, 'events_admitted': admitted}})
        if gate is not None and gate.superseded:
            warnings.append({'code': 'EXECUTION_RUN_SUPERSEDED', 'detail': f'{gate.superseded} run(s) never ended '
                             'and were superseded by a new runner start', 'context': {'count': gate.superseded}})
        refused = {k: v for k, v in (gate.refused.items() if gate else ()) if v}
        if refused:
            warnings.append({'code': 'EXECUTION_EVENTS_REFUSED', 'detail': f'{sum(refused.values())} execution '
                             'event(s) did not fire: not of this session (stale, another run, a repeat) or '
                             'malformed', 'context': dict(sorted(refused.items()))})
        if self.coordinator.stamps_after_receipt:
            warnings.append({'code': 'STAMPS_AFTER_RECEIPT', 'detail': f'{self.coordinator.stamps_after_receipt} '
                             'stream decision(s) refused CLOCK_MISMATCH: a same-host source stamped frames after '
                             'this host received them (its clock mapping ran ahead, or this host\'s clock stepped '
                             'back)',
                             'context': {'count': self.coordinator.stamps_after_receipt}})
        if self.coordinator.assembly_errors:
            warnings.append({'code': 'ASSEMBLY_ERRORS', 'detail': f'{self.coordinator.assembly_errors} capture(s) '
                             'recorded FAILED after an assembly error', 'context': {
                                 'count': self.coordinator.assembly_errors}})
        if self.source_info_changes:
            warnings.append({'code': 'SOURCE_INFO_CHANGED', 'detail': f'a camera published a different source '
                             f'document {self.source_info_changes} time(s) during the session; the session records '
                             'the one valid at its start', 'context': {'count': self.source_info_changes}})
        if self.callback_errors:
            warnings.append({'code': 'CALLBACK_ERRORS', 'detail': f'{self.callback_errors} malformed message(s) '
                             'ignored', 'context': {'count': self.callback_errors,
                                                    'last': (self.last_error or '')[:300]}})
        return warnings

    # ---- observability -----------------------------------------------------------------------------------------

    def performance(self) -> dict:
        """Lightweight measurements of this session (observability, not a benchmark)."""
        started = self.session_started_mono
        elapsed = (time.monotonic() - started) if started else None
        received = dict(self.coordinator.received)
        out = {'elapsed_s': elapsed, 'received': received, 'outcomes': dict(self.outcomes),
               'assembled': self.assembled}
        if elapsed:
            out['received_hz'] = {k: v / elapsed for k, v in received.items()}
            out['captures_hz'] = sum(self.outcomes.values()) / elapsed
        worker = self.worker
        if worker is not None:
            out['writer'] = worker.stats()
        usage = resource.getrusage(resource.RUSAGE_SELF)
        start = getattr(self, '_cpu_start', None)
        if start is not None and elapsed:
            cpu = (usage.ru_utime + usage.ru_stime) - (start.ru_utime + start.ru_stime)
            out['cpu_percent'] = 100.0 * cpu / elapsed
        out['max_rss_mb'] = usage.ru_maxrss / 1024.0
        return out

    def _publish_diagnostics(self):
        try:
            self._diag.publish(self._diagnostics())
        except Exception as exc:  # noqa: BLE001 -- observability never takes the recorder down
            self.last_error = f'diagnostics: {exc!r}'

    def _diagnostics(self) -> DiagnosticArray:
        session, worker, state = self.session, self.worker, self.state      # one read each (other threads)
        gate = self._event_gate
        stats = worker.stats() if worker is not None else {}
        status = DiagnosticStatus(name=f'trainit_data_recorder: {self.get_fully_qualified_name()}',
                                  hardware_id=self.cfg['source']['backend'])
        level = DiagnosticStatus.OK
        if stats.get('faulted') or self.last_error:
            level = DiagnosticStatus.ERROR
        elif (stats.get('overruns') or stats.get('lost') or stats.get('write_failures') or self.trigger_overruns
              or self.sync_failures or self.clock_jumps or self.coordinator.stamps_after_receipt):
            level = DiagnosticStatus.WARN
        status.level = level
        session_id = session.session_id if session is not None else ''
        status.message = f'{state} {session_id}'.strip()
        values = {'session_state': state, 'session_id': session_id,
                  'captures_assembled': self.assembled, 'captures_attempted': sum(self.outcomes.values()),
                  'captures_complete': self.outcomes['complete'], 'captures_partial': self.outcomes['partial'],
                  'captures_failed': self.outcomes['failed'],
                  'queue_overruns': stats.get('overruns', 0) + self.trigger_overruns,
                  'captures_lost': stats.get('lost', 0),
                  'write_failures': stats.get('write_failures', 0),
                  'calibration_changes': self.coordinator.calibration_changes, 'sync_failures': self.sync_failures,
                  'stamps_after_receipt': self.coordinator.stamps_after_receipt,
                  'execution_events_admitted': gate.admitted if gate else '',
                  'execution_events_refused': json.dumps({k: v for k, v in gate.refused.items() if v})
                  if gate else '',
                  'execution_run': (gate.open_run or '') if gate else '',
                  'execution_events_outside_session': self.events_outside_session,
                  'last_skew_ns': '' if self.last_skew_ns is None else self.last_skew_ns,
                  'queue_depth': stats.get('queue_depth', 0), 'queue_high_water': stats.get('queue_high_water', 0),
                  'write_latency_ms_last': (stats.get('write_latency_ms') or {}).get('last') or '',
                  'interval_slips': self.interval_slips, 'clock_jumps': self.clock_jumps,
                  'callback_errors': self.callback_errors, 'last_error': self.last_error or ''}
        values.update({f'received/{k}': v for k, v in dict(self.coordinator.received).items()})
        status.values = [KeyValue(key=k, value=str(v)) for k, v in values.items()]
        msg = DiagnosticArray(status=[status])
        msg.header.stamp = self.get_clock().now().to_msg()
        return msg

    def shutdown(self):
        """Stop the threads. An ACTIVE session is sealed when possible (else left for recover)."""
        if self.state == 'ACTIVE':
            self._close_session(Trigger.Request(), Trigger.Response())
        self._running = False
        self._coord_thread.join(timeout=5.0)
        self._tf_executor.shutdown(timeout_sec=2.0)
        self._tf_node.destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = Recorder()
    streams = len(node.coordinator.streams)
    # Every ~/capture may block a thread while it waits for its write: keep threads for the callbacks too.
    executor = MultiThreadedExecutor(num_threads=node.cfg['writer']['max_pending'] + 2 * streams + 4)
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.shutdown()
        executor.shutdown()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    os.environ.setdefault('PYTHONUNBUFFERED', '1')
    main()
