"""The CaptureCoordinator's logic: buffering, trigger boundaries, the sync policy, assembly (no ROS imports).

The ROS node (``recorder_node``) feeds it samples and CameraInfos and fires triggers; this module decides.
Keeping it free of rclpy makes every rule unit-testable with an injected clock.

**Policy ``FIRST_AT_OR_AFTER_BOUNDARY`` v1** (D0_DESIGN.md §7.3, §9; the M3 ``DetectObject`` rule):
- a trigger fires at boundary ``B``, read on the SESSION clock (ROS time, or sim time);
- ``B <= 0`` -> every stream ``CLOCK_UNAVAILABLE`` (the session clock is not running);
- per stream, the sample with the smallest source stamp ``>= B`` is selected (buffered or arriving);
- a sample whose stamp is 0 has no source stamp: never selected, never replaced by its receipt time
  (``CLOCK_UNAVAILABLE`` if nothing else arrives);
- the first sample at or after ``B`` lying more than ``clock_mismatch_ms`` after it, or (at the deadline) samples
  received after the trigger all stamped more than ``clock_mismatch_ms`` BEFORE it -> ``CLOCK_MISMATCH``: the
  publisher and the session clock disagree (a ROS header carries no clock domain, so magnitude is the evidence);
- a sample on another clock domain is never compared -> ``INCOMPATIBLE_CLOCKS``;
- for a sensor proven to run on the recorder's own host while the session clock is that host's ROS time
  (``bind_to_receipt``), a selected sample stamped more than ``RECEIPT_MARGIN_NS`` AFTER this host received it ->
  ``CLOCK_MISMATCH``: a driver that stamps at or before publishing cannot do that on this host's clock (2026-10-30,
  D0-C: librealsense's global-time fit ran up to 0.53 s ahead and the first sample "at or after" a MANUAL boundary
  was a frame taken before it). Stricter, never a wider tolerance. LIMITS: a stamp running ahead by less than the
  source's own latency (about one frame for a RealSense: ~33 ms) is not caught; a stamp running BEHIND is not caught
  (it only makes a capture later than its trigger); a backwards STEP of this host's wall clock larger than the
  margin can refuse healthy frames too (as it already disturbs every ROS_TIME session);
- the wait is bounded on the HOST MONOTONIC clock (``timeout_ms``), so a frozen /clock cannot hang a
  capture: an undecided stream becomes ``TIMEOUT`` and the capture is always assembled (an error while
  assembling one stream makes that stream missing; an error assembling the capture makes it FAILED).

**Policy ``FIRST_ALIGNED_AT_OR_AFTER_BOUNDARY`` v1** (selectable, owner decision 2026-11-05; D0_DESIGN.md §9: a
registry row + this implementation, no schema change). For sources that lose frames of one stream (Isaac's RGB and
depth come from ONE render tick, but a lost depth frame made v1 pair colour with the depth of a later tick):
- the capture is the EARLIEST reference-stream sample at or after ``B`` for which every other stream has a sample
  within ``tolerance_ms`` of it (the nearest one; an identical stamp wins), all of them eligible under the v1 rules
  above (session clock, stamped, at or after ``B``, not ``CLOCK_MISMATCH``, not stamped after a bound receipt);
- a reference sample is given up only when every stream that lacks its partner has already delivered a sample
  stamped beyond it plus the tolerance (streams arrive in stamp order), so a late partner is waited for and the
  choice does not depend on arrival timing;
- a stream v1 already decides with a reason (``CLOCK_UNAVAILABLE``, ``INCOMPATIBLE_CLOCKS``, ...) is decided the
  same way, and the group is formed over the remaining streams;
- latency: the decision waits while a partner may still arrive -- in a steady stream about one frame period of the
  slower stream (until it delivers a sample past the reference); a stalled stream costs up to ``timeout_ms``. A
  partner that arrives OUT of stamp order after the decision is not reconsidered (as in v1);
- where v1 is already COMPLETE the aligned policy keeps v1's reference sample and a skew no larger than v1's (the
  NEAREST partner, so the kept depth and the sync status may differ: SYNCED where v1 said WITHIN_TOLERANCE);
- at the deadline, a group that was only waiting for a possibly nearer partner is used with the nearest partner
  it has; without any aligned group, each stream falls back to v1 (its first sample at or after ``B``, or
  its reason): the frames are kept and the sync verdict says how far apart they are (``OUT_OF_TOLERANCE`` ->
  PARTIAL). Nothing is invented: a capture is either aligned or says it is not.
The wait is bounded by the same ``timeout_ms``.

Each image is paired with ITS CameraInfo (same stamp, else the newest stamped before it), never with whatever
arrived last. A backwards jump of the session clock (a simulation reset) drops the buffers (``on_clock_jump``).

Then each selected sample is validated against the contract (encoding, optical frame, CameraInfo,
registered depth, required TF) and the sync status is computed from the stamps actually kept:
``SYNCED`` / ``WITHIN_TOLERANCE`` / ``OUT_OF_TOLERANCE`` / ``SINGLE_STREAM`` / ``INCOMPATIBLE_CLOCKS``.
COMPLETE = every stream kept and in tolerance; PARTIAL = some kept; FAILED = none. Nothing is dropped
silently: every requested stream is either an observation or a ``missing`` entry with its reason.
"""
import collections
import threading
from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np

from trainit_data.calibration import CalibrationRecord
from trainit_data.observation import PoseAtStamp, Receipt
from trainit_data.timestamps import Timestamp
from trainit_data.writer import ImageSample

from . import conversions
from .contract import DEPTH_UNIT_SCALE_M

POLICY = ('FIRST_AT_OR_AFTER_BOUNDARY', 1)
ALIGNED_POLICY = ('FIRST_ALIGNED_AT_OR_AFTER_BOUNDARY', 1)
POLICIES = (POLICY, ALIGNED_POLICY)
RECEIPT_MARGIN_NS = 2_000_000       # a same-host stamp may follow its receipt by at most this (clock reads, slew)
_NS = 1_000_000_000


@dataclass
class Sample:
    """One received image, already converted (or refused) by the node's callback."""
    sec: int
    nanosec: int
    clock: tuple                        # the clock key it is stamped on (the session clock for a ROS header)
    receipt_mono_ns: int
    receipt_wall_ns: int
    frame_id: str
    encoding: str
    array: Optional[np.ndarray] = None
    refused: Optional[str] = None       # why the image could not be converted
    msg: object = None                  # the raw Image, converted only if this sample is selected

    @property
    def stamp_ns(self) -> int:
        return self.sec * _NS + self.nanosec

    def load(self):
        """Convert the raw message once (in the coordinator thread, never in a ROS callback)."""
        if self.array is None and self.refused is None and self.msg is not None:
            try:
                self.array = conversions.image_to_array(self.msg)
            except conversions.ImageRefused as exc:
                self.refused = str(exc)
            self.msg = None


@dataclass
class Info:
    block: dict                         # canonical intrinsics (conversions.intrinsics)
    frame_id: str
    stamp_ns: int = 0                   # the CameraInfo header stamp (pairs it with its image)


@dataclass
class Pending:
    trigger: dict                       # the TriggerRecord dict (issued included)
    boundary_ns: int
    deadline_mono_ns: int
    fired_mono_ns: int = 0
    selected: dict = field(default_factory=dict)        # stream key -> Sample
    decided: dict = field(default_factory=dict)         # stream key -> (reason, detail)
    done: threading.Event = field(default_factory=threading.Event)
    result: object = None               # set by the writer: the committed Capture, or the DataError / 'DROPPED'


@dataclass
class Draft:
    """An assembled capture, ready for the writer (store I/O happens in the writer thread only)."""
    pending: Pending
    outcome: str
    samples: list
    missing: list
    sync: Optional[dict]
    calibrations: list                  # CalibrationRecord objects to put before the capture
    skew_ns: Optional[int] = None
    calibration_ids: dict = field(default_factory=dict)  # stream key -> id (counted once WRITTEN)


class Overrun(Exception):
    """More triggers pending than the configuration allows: the firing is dropped and counted."""


# Extrinsic changes below these bounds are TF float noise, not a calibration change: the previous values are kept
# (one calibration id). Above them a new record is made and counted (D0_DESIGN.md §8 uses the same bounds).
EXTRINSIC_TOLERANCE_M = 0.001
EXTRINSIC_TOLERANCE_RAD = 0.1 * 3.141592653589793 / 180.0


class Coordinator:
    def __init__(self, resolved: dict, vocab, *, tf_lookup: Callable = None, now_mono: Callable = None,
                 description_sha256: str = None):
        import time
        self.cfg = resolved
        self.vocab = vocab
        self.clock_key = (resolved['clock']['session_domain'], resolved['clock'].get('sim_source'))
        self.stamp_origin = resolved['source']['descriptor']['stamp_origin']
        self.tf_lookup = tf_lookup or (lambda parent, child, sec, nanosec: None)
        self.description_sha256 = description_sha256
        self.now_mono = now_mono or time.monotonic_ns
        sync = resolved['sync']
        self.timeout_ns = int(sync['timeout_ms'] * 1e6)
        self.mismatch_ns = int(sync['clock_mismatch_ms'] * 1e6)
        self.tolerance_ns = int(round(sync['tolerance_ms'] * 1e6))
        self.reference = sync['reference']
        self.policy = (sync['policy'], sync['policy_version'])
        if self.policy not in POLICIES:
            raise ValueError(f'sync policy {self.policy} has no implementation in this coordinator')
        self.max_pending = resolved['writer']['max_pending']
        self.streams = {}               # key -> (sensor block, stream name, stream block)
        for sensor in resolved['sensors']:
            for name, block in sensor['streams'].items():
                self.streams[f'{sensor["sensor_id"]}/{name}'] = (sensor, name, block)
        depth = resolved['writer']['buffer_frames']
        self.buffers = {k: collections.deque(maxlen=depth) for k in self.streams}
        self.infos = {k: collections.deque(maxlen=2 * depth) for k in self.streams}
        self.received = {k: 0 for k in self.streams}
        self.pending = []
        self.last_calibration = {}      # stream key -> id of the last WRITTEN capture
        self.last_extrinsic = {}        # stream key -> (parent, translation, rotation, kind) last used
        self.calibration_changes = 0
        self.assembly_errors = 0
        self.receipt_bound = frozenset()    # sensor ids whose stamps can never follow their receipt (same host)
        self.stamps_after_receipt = 0
        self.lock = threading.Lock()

    # ---- inputs (called from ROS callbacks: cheap, never blocking) ---------------------------------

    def on_sample(self, key: str, sample: Sample):
        with self.lock:
            self.buffers[key].append(sample)
            self.received[key] += 1

    def on_info(self, key: str, info: Info):
        with self.lock:
            self.infos[key].append(info)

    def on_clock_jump(self) -> int:
        """The session clock went BACKWARDS (a simulation reset): buffered samples and CameraInfos belong to the
        old timeline and are dropped, and every stream a pending capture has not decided yet is closed
        CLOCK_MISMATCH -- a capture never mixes the two timelines (a deterministic simulation replays the SAME
        stamps). Returns the number of pending captures affected."""
        with self.lock:
            for key in self.streams:
                self.buffers[key].clear()
                self.infos[key].clear()
            for pending in self.pending:
                for key in self.streams:
                    if key not in pending.selected and key not in pending.decided:
                        pending.decided[key] = ('CLOCK_MISMATCH', 'the session clock jumped back (simulation reset) '
                                                'before this stream was decided')
            return len(self.pending)

    def bind_to_receipt(self, sensor_ids):
        """From now on (a session start), these sensors' selected samples must not be stamped after their receipt:
        the recorder proved them on its own host and its session clock is that host's ROS time."""
        with self.lock:
            self.receipt_bound = frozenset(sensor_ids)

    def fire(self, trigger: dict, boundary_ns: int) -> Pending:
        """Open a capture at boundary ``B`` (the trigger record carries ``issued``)."""
        with self.lock:
            if len(self.pending) >= self.max_pending:
                raise Overrun(f'{len(self.pending)} captures already pending')
            now = self.now_mono()
            pending = Pending(trigger, boundary_ns, now + self.timeout_ns, now)
            if boundary_ns <= 0:
                for key in self.streams:
                    pending.decided[key] = ('CLOCK_UNAVAILABLE', 'the session clock reads 0 (not running)')
            self.pending.append(pending)
            return pending

    def expire_all(self):
        """Session close: every pending capture reaches its deadline now (it is assembled, never abandoned)."""
        with self.lock:
            now = self.now_mono()
            for pending in self.pending:
                pending.deadline_mono_ns = min(pending.deadline_mono_ns, now)

    # ---- the policy -----------------------------------------------------------------------------------

    def _select(self, pending: Pending, key: str):
        """FIRST_AT_OR_AFTER_BOUNDARY v1 for one stream: a Sample, a (reason, detail), or None (keep waiting).

        The choice is the smallest source stamp >= B among the samples buffered when the stream is decided (a
        sample arriving later with a smaller stamp is not reconsidered); one sample may serve two concurrent
        captures whose boundaries it follows -- each record stays true.
        """
        b = pending.boundary_ns
        best, unstamped, foreign, newest_after_fire = None, False, None, None
        for sample in self.buffers[key]:
            fresh = sample.receipt_mono_ns >= pending.fired_mono_ns
            if sample.clock != self.clock_key:
                if fresh:
                    foreign = sample
                continue
            if sample.stamp_ns == 0:
                unstamped = unstamped or fresh
                continue
            if fresh and (newest_after_fire is None or sample.stamp_ns > newest_after_fire):
                newest_after_fire = sample.stamp_ns
            if sample.stamp_ns >= b and (best is None or sample.stamp_ns < best.stamp_ns):
                best = sample
        if best is not None:
            ahead = best.stamp_ns - best.receipt_wall_ns
            if self.streams[key][0]['sensor_id'] in self.receipt_bound and ahead > RECEIPT_MARGIN_NS:
                self.stamps_after_receipt += 1
                return ('CLOCK_MISMATCH', f'the selected sample is stamped {ahead / 1e6:.1f} ms AFTER this host '
                        'received it: a same-host source\'s stamp is not this host\'s clock (a driver\'s '
                        'device-clock mapping running ahead, or this host\'s wall clock stepped back)')
            if best.stamp_ns - b > self.mismatch_ns:
                return ('CLOCK_MISMATCH', f'first sample {(best.stamp_ns - b) / 1e9:.3f} s after the boundary '
                        f'(> {self.mismatch_ns / 1e9:g} s): the publisher and the session clock disagree')
            return best
        if foreign is not None:
            return ('INCOMPATIBLE_CLOCKS', f'samples are stamped on {foreign.clock[0]}, the session clock is '
                    f'{self.clock_key[0]}: never compared')
        if self.now_mono() >= pending.deadline_mono_ns:
            if newest_after_fire is not None and b - newest_after_fire > self.mismatch_ns:
                behind = (b - newest_after_fire) / 1e9
                return ('CLOCK_MISMATCH', f'samples arrived after the trigger stamped {behind:.3f} s BEFORE the '
                        'boundary: the publisher is behind the session clock')
            if unstamped:
                return ('CLOCK_UNAVAILABLE', 'samples arrived without a source stamp (0); receipt time is never '
                        'substituted')
            return ('TIMEOUT', f'no sample at or after the boundary within {self.timeout_ns / 1e6:g} ms')
        return None

    def _eligible(self, pending: Pending, key: str, sample: Sample) -> bool:
        """May ``sample`` join an aligned group? The v1 rules, applied to one sample."""
        if sample.clock != self.clock_key or sample.stamp_ns == 0 or sample.stamp_ns < pending.boundary_ns:
            return False
        if sample.stamp_ns - pending.boundary_ns > self.mismatch_ns:
            return False
        bound = self.streams[key][0]['sensor_id'] in self.receipt_bound
        return not (bound and sample.stamp_ns - sample.receipt_wall_ns > RECEIPT_MARGIN_NS)

    def _aligned_group(self, pending: Pending, keys: list, final: bool = False):
        """FIRST_ALIGNED_AT_OR_AFTER_BOUNDARY v1 over ``keys``: {key: Sample}, or None (keep waiting / no group).

        Before the deadline the earliest live reference sample decides: its group, or wait. At the deadline
        (``final``) the earliest reference sample whose every partner is buffered wins with the nearest partners
        it has -- a group that was only waiting for a possibly nearer partner is used, never dropped for v1."""
        reference = self.reference if self.reference in keys else keys[0]
        others = [k for k in keys if k != reference]
        candidates = {k: sorted((x for x in self.buffers[k] if self._eligible(pending, k, x)),
                                key=lambda x: x.stamp_ns) for k in keys}
        newest = {k: max((x.stamp_ns for x in self.buffers[k] if x.clock == self.clock_key), default=None)
                  for k in others}
        for ref in candidates[reference]:
            group, nearer_possible, missing, dead = {reference: ref}, False, False, False
            for key in others:
                near = [x for x in candidates[key] if abs(x.stamp_ns - ref.stamp_ns) <= self.tolerance_ns]
                if near:
                    group[key] = min(near, key=lambda x: (abs(x.stamp_ns - ref.stamp_ns), x.stamp_ns))
                    if newest[key] < ref.stamp_ns and group[key].stamp_ns != ref.stamp_ns:
                        nearer_possible = True  # a nearer partner (the same stamp) may still arrive
                elif newest[key] is not None and newest[key] > ref.stamp_ns + self.tolerance_ns:
                    dead = True                 # this stream moved past: the partner was lost
                else:
                    missing = True              # no partner yet; it may still arrive
            if dead or (final and missing):
                continue
            if final:
                return group
            return None if (missing or nearer_possible) else group
        return None

    def _poll_aligned(self, pending: Pending, final: bool):
        open_keys = []
        for key in self.streams:
            if key in pending.selected or key in pending.decided:
                continue
            choice = self._select(pending, key)
            if isinstance(choice, Sample) or choice is None:
                open_keys.append(key)
            else:
                pending.decided[key] = choice          # a v1 reason (clock, timeout at the deadline)
        if not open_keys:
            return
        group = self._aligned_group(pending, open_keys, final)
        if group is not None:
            pending.selected.update(group)
        elif final:                                     # no aligned group in time: v1, honestly labelled
            for key in open_keys:
                choice = self._select(pending, key)
                if isinstance(choice, Sample):
                    pending.selected[key] = choice
                elif choice is not None:
                    pending.decided[key] = choice

    def poll(self) -> list:
        """Advance every pending capture; return the Drafts that are ready. A capture ALWAYS terminates: at its
        deadline it is assembled with whatever it has; an error while assembling it becomes FAILED, never a
        capture stuck in the list."""
        ready = []
        with self.lock:
            now = self.now_mono()
            for pending in list(self.pending):
                final = now >= pending.deadline_mono_ns
                try:
                    if self.policy == ALIGNED_POLICY:
                        self._poll_aligned(pending, final)
                    else:
                        for key in self.streams:
                            if key in pending.selected or key in pending.decided:
                                continue
                            choice = self._select(pending, key)
                            if isinstance(choice, Sample):
                                pending.selected[key] = choice
                            elif choice is not None:
                                pending.decided[key] = choice
                    if len(pending.selected) + len(pending.decided) < len(self.streams):
                        continue
                    draft = self._assemble(pending, final=final)
                except Exception as exc:  # noqa: BLE001 -- a defect here must not wedge the coordinator
                    self.assembly_errors += 1
                    draft = self._failed(pending, f'assembly error: {exc!r}')
                if draft is not None:
                    self.pending.remove(pending)
                    ready.append(draft)
        return ready

    def _failed(self, pending, detail) -> Draft:
        missing = [{'stream': key, 'reason': 'CALIBRATION_MISMATCH', 'detail': detail[:500]} for key in self.streams]
        return Draft(pending, 'FAILED', [], missing, None, [])

    # ---- assembly -------------------------------------------------------------------------------------

    def _assemble(self, pending: Pending, *, final: bool) -> Optional[Draft]:
        kept, missing, calibrations, waiting = {}, [], {}, False
        for key, (sensor, name, block) in self.streams.items():
            if key in pending.decided:
                reason, detail = pending.decided[key]
                missing.append({'stream': key, 'reason': reason, 'detail': detail})
                continue
            sample = pending.selected[key]
            try:
                verdict = self._validate(key, sample, final)
                if verdict is None:
                    waiting = True          # its CameraInfo may still arrive; the deadline decides
                    continue
                if isinstance(verdict, tuple):
                    missing.append({'stream': key, 'reason': verdict[0], 'detail': verdict[1]})
                    continue
                info = verdict
                extrinsic, pose = self._transforms(sensor, block, sample)
                if extrinsic is False or pose is False:
                    if not final:
                        waiting = True      # TF may still arrive; the deadline decides
                        continue
                    missing.append({'stream': key, 'reason': 'TF_UNAVAILABLE', 'detail': 'no transform at the '
                                    'observation stamp within the timeout (latest is never substituted)'})
                    continue
                record = self._calibration(key, sensor, name, block, sample, info, extrinsic)
            except Exception as exc:  # noqa: BLE001 -- e.g. a non-finite CameraInfo: that stream only
                missing.append({'stream': key, 'reason': 'CALIBRATION_MISMATCH',
                                'detail': f'no canonical calibration could be formed: {exc}'[:500]})
                continue
            calibrations[key] = record
            kept[key] = (sample, record, pose)
        if waiting:
            return None
        return self._draft(pending, kept, missing, calibrations)

    def _info_for(self, key, stamp_ns):
        """The CameraInfo of the image at ``stamp_ns``: the one with the same stamp (the contract publishes them
        together), else the newest one stamped at or before it; ``None`` if none (yet)."""
        exact = before = None
        for info in self.infos[key]:
            if info.stamp_ns == stamp_ns:
                exact = info
            elif info.stamp_ns < stamp_ns and (before is None or info.stamp_ns >= before.stamp_ns):
                before = info
        return exact or before

    def _validate(self, key, sample: Sample, final: bool):
        """The CameraInfo to use (validated), a (reason, detail) refusal, or None (wait for its CameraInfo)."""
        sensor, name, block = self.streams[key]
        row = self.vocab.modalities[block['modality']]
        sample.load()
        if sample.refused is not None or sample.encoding not in row['encodings']:
            return ('ENCODING_REFUSED', sample.refused or f'{sample.encoding} is not an encoding of modality '
                    f'{block["modality"]} ({list(row["encodings"])})')
        if sample.frame_id != block['expected_frame']:
            return ('FRAME_MISMATCH', f'image frame_id {sample.frame_id!r}, the deployed contract expects '
                    f'{block["expected_frame"]!r} (never rewritten)')
        info = self._info_for(key, sample.stamp_ns)
        if info is None:
            if not final:
                return None
            return ('NO_CAMERA_INFO', f'no CameraInfo at or before the image stamp on {block["info"]}')
        if info.frame_id != block['expected_frame']:
            return ('FRAME_MISMATCH', f'CameraInfo frame_id {info.frame_id!r}, expected {block["expected_frame"]!r}')
        height, width = sample.array.shape[:2]
        if (info.block['width'], info.block['height']) != (width, height):
            return ('CALIBRATION_MISMATCH', f'image {width}x{height}, CameraInfo '
                    f'{info.block["width"]}x{info.block["height"]}')
        if 'registered_to' in block:
            other = f'{sensor["sensor_id"]}/{block["registered_to"]}'
            reference = self._info_for(other, sample.stamp_ns)
            if reference is None and not final:
                return None
            if reference is None or not conversions.same_geometry(info.block, reference.block):
                return ('REGISTRATION_MISMATCH', f'registered depth needs a CameraInfo identical to {other}\'s at '
                        'the same moment (contracts/camera.md, "Registered depth")')
        return info

    def _transforms(self, sensor, block, sample: Sample):
        """(extrinsic, pose): None = not configured; False = configured but not (yet) available."""
        extrinsic = pose = None
        if 'extrinsic_parent_frame' in sensor:
            found = self.tf_lookup(sensor['extrinsic_parent_frame'], block['expected_frame'], sample.sec,
                                   sample.nanosec)
            extrinsic = False if found is None else (sensor['extrinsic_parent_frame'], *found,
                                                     sensor['extrinsic_provenance'])
        if 'pose_parent_frame' in sensor:
            found = self.tf_lookup(sensor['pose_parent_frame'], block['expected_frame'], sample.sec, sample.nanosec)
            pose = False if found is None else PoseAtStamp(sensor['pose_parent_frame'], tuple(found[0]),
                                                           tuple(found[1]))
        return extrinsic, pose

    def _stable_extrinsic(self, key, extrinsic):
        """Keep the previous extrinsic when the new one differs only by TF noise (one calibration id)."""
        previous = self.last_extrinsic.get(key)
        if extrinsic is None or previous is None or previous[0] != extrinsic[0] or previous[3] != extrinsic[3]:
            self.last_extrinsic[key] = extrinsic
            return extrinsic
        dt = max(abs(a - b) for a, b in zip(previous[1], extrinsic[1]))
        dot = abs(sum(a * b for a, b in zip(previous[2], extrinsic[2])))
        angle = 2.0 * np.arccos(min(1.0, dot))
        if dt <= EXTRINSIC_TOLERANCE_M and angle <= EXTRINSIC_TOLERANCE_RAD:
            return previous
        self.last_extrinsic[key] = extrinsic
        return extrinsic

    def _calibration(self, key, sensor, name, block, sample, info, extrinsic) -> CalibrationRecord:
        descriptor = self.cfg['source']['descriptor']
        content = conversions.calibration_content(
            sensor_id=sensor['sensor_id'], stream=name, optical_frame=block['expected_frame'], info_block=info.block,
            intrinsics_kind=descriptor['intrinsics_kind'], intrinsics_notes=self.cfg['source'].get('intrinsics_notes'),
            encoding=sample.encoding,
            registered_to=block.get('registered_to'), registration_basis=descriptor['registration_basis'],
            extrinsic=self._stable_extrinsic(key, extrinsic), moving='pose_parent_frame' in sensor,
            description_sha256=self.description_sha256)
        return CalibrationRecord.build(content, self.vocab)

    def _draft(self, pending, kept, missing, calibrations) -> Draft:
        session_domain, sim_source = self.clock_key
        samples = []
        for key, (sample, record, pose) in kept.items():
            sensor, name, block = self.streams[key]
            samples.append(ImageSample(
                sensor_id=sensor['sensor_id'], stream=name,
                stamp=Timestamp(sample.sec, sample.nanosec, session_domain, self.stamp_origin, sim_source),
                array=sample.array, encoding=sample.encoding, frame_id=sample.frame_id,
                unit_scale_m=DEPTH_UNIT_SCALE_M.get(sample.encoding) if block['modality'] == 'depth' else None,
                calibration_id=record.calibration_id, receipt=Receipt(sample.receipt_mono_ns, sample.receipt_wall_ns),
                pose_at_stamp=pose))
        sync, skew = self._sync(kept, missing)
        if not kept:
            outcome = 'FAILED'
        elif missing or (sync and sync['status'] == 'OUT_OF_TOLERANCE'):
            outcome = 'PARTIAL'
        else:
            outcome = 'COMPLETE'
        return Draft(pending, outcome, samples, missing, sync, [calibrations[k] for k in kept], skew,
                     {k: calibrations[k].calibration_id for k in kept})

    def written(self, draft: Draft):
        """The writer committed ``draft``: count calibration changes (per stream) against WRITTEN captures only."""
        with self.lock:
            for key, cal_id in draft.calibration_ids.items():
                previous = self.last_calibration.get(key)
                if previous is not None and previous != cal_id:
                    self.calibration_changes += 1          # a new id, never merged into the old one
                self.last_calibration[key] = cal_id

    def _sync(self, kept, missing):
        if not kept:
            return None, None
        policy = {'name': self.policy[0], 'version': self.policy[1],
                  'params': {'timeout_ms': self.timeout_ns // 1_000_000,
                             'clock_mismatch_ms': self.mismatch_ns // 1_000_000}}
        reference = self.reference if self.reference in kept else next(iter(kept))
        sync = {'policy': policy, 'reference': reference}
        if any(m['reason'] == 'INCOMPATIBLE_CLOCKS' for m in missing):
            sync['status'] = 'INCOMPATIBLE_CLOCKS'
            return sync, None
        if len(kept) == 1:
            sync['status'] = 'SINGLE_STREAM'
            return sync, None
        ref = kept[reference][0].stamp_ns
        skew = max(abs(sample.stamp_ns - ref) for sample, _, _ in kept.values())
        sync.update(max_skew_ns=skew, tolerance_ns=self.tolerance_ns)
        sync['status'] = 'SYNCED' if skew == 0 else (
            'WITHIN_TOLERANCE' if skew <= self.tolerance_ns else 'OUT_OF_TOLERANCE')
        return sync, skew

    def reset_counters(self):
        with self.lock:
            self.received = {k: 0 for k in self.streams}
            self.last_calibration = {}
            self.last_extrinsic = {}
            self.calibration_changes = 0
            self.assembly_errors = 0
            self.stamps_after_receipt = 0

    def drain_pending(self) -> list:
        """Remove and return every pending capture (a new session must not inherit an old one's)."""
        with self.lock:
            pendings, self.pending = self.pending, []
        return pendings
