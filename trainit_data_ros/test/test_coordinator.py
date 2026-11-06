"""FIRST_AT_OR_AFTER_BOUNDARY v1 and capture assembly, driven by an injected host-monotonic clock (no ROS).

Every draft is also written through a real trainit_data SessionWriter: the coordinator can only produce
captures the canonical core accepts.
"""
import numpy as np
import pytest

from _ros_helpers import FRAME, config, depth_mm, info_msg, load_profile, rgb
from trainit_data.store import Store
from trainit_data.vocab import Vocabulary
from trainit_data_ros import conversions
from trainit_data_ros.coordinator import Coordinator, Info, Overrun, Sample
from trainit_data_ros.recording_config import resolve
from trainit_data_ros.writer_worker import WriterWorker

VOCAB = Vocabulary.default()
SEC = 1_000_000_000
B = 100 * SEC
COLOR, DEPTH = 'overhead_camera/color', 'overhead_camera/depth'


class Clock:
    def __init__(self):
        self.ns = 5 * SEC

    def __call__(self):
        return self.ns

    def advance(self, ms):
        self.ns += int(ms * 1e6)


def make(doc=None, tf=None, use_sim_time=False):
    clock = Clock()
    cfg = resolve(doc or config(), VOCAB, load_profile=load_profile, use_sim_time=use_sim_time)
    coordinator = Coordinator(cfg, VOCAB, tf_lookup=tf, now_mono=clock)
    return coordinator, clock, cfg


def sample(coordinator, clock, stamp_ns, array, encoding, frame=FRAME, clock_key=None):
    sec, nsec = divmod(stamp_ns, SEC)
    return Sample(sec, nsec, clock_key or coordinator.clock_key, clock.ns, 1, frame, encoding, array=array)


def trigger(boundary_ns=B, index=0):
    sec, nsec = divmod(boundary_ns, SEC)
    return {'trigger_id': 'manual', 'kind': 'MANUAL', 'firing_index': index, 'origin': 'OPERATOR',
            'issued': {'stamp': {'sec': sec, 'nanosec': nsec, 'clock_domain': 'ROS_TIME', 'origin': 'NODE_CLOCK'}}}


def feed_infos(coordinator, depth_fx=4.0, frame=FRAME):
    coordinator.on_info(COLOR, Info(conversions.intrinsics(info_msg()), FRAME))
    coordinator.on_info(DEPTH, Info(conversions.intrinsics(info_msg(fx=depth_fx)), frame))


def run(coordinator, clock, wait_ms=1100):
    drafts = coordinator.poll()
    steps = 0
    while not drafts and steps < 200:
        clock.advance(wait_ms / 100)
        drafts = coordinator.poll()
        steps += 1
    assert len(drafts) == 1, 'a capture must always terminate'
    return drafts[0]


def reasons(draft):
    return {m['stream']: m['reason'] for m in draft.missing}


@pytest.fixture
def written(tmp_path):
    """Write drafts through the real core; returns the committed Capture."""
    store = Store.open(tmp_path / 'TrainItData', create=True)
    holder = {}

    def write(draft, cfg):
        if 'writer' not in holder:
            sensors = [{'sensor_id': s['sensor_id'],
                        'streams': [{'stream': n, 'modality': b['modality']} for n, b in s['streams'].items()]}
                       for s in cfg['sensors']]
            writer = store.create_session(source={'transport': 'LIVE_ROS', 'backend': 'mock',
                                                  'descriptor': cfg['source']['descriptor']},
                                          clock=cfg['clock'], sensors=sensors)
            holder['writer'] = writer
            holder['worker'] = WriterWorker(store, writer, 4)
        assert holder['worker'].submit(draft)
        assert draft.pending.done.wait(10)
        return draft.pending.result
    yield write
    if 'worker' in holder:
        holder['worker'].stop()
        holder['writer'].close()


def test_identical_stamps_are_synced_and_complete(written):
    c, clock, cfg = make()
    feed_infos(c)
    c.fire(trigger(), B)
    c.on_sample(COLOR, sample(c, clock, B + 5_000_000, rgb(), 'rgb8'))
    c.on_sample(DEPTH, sample(c, clock, B + 5_000_000, depth_mm(), '16UC1'))
    draft = run(c, clock)
    assert draft.outcome == 'COMPLETE' and draft.sync['status'] == 'SYNCED' and draft.sync['max_skew_ns'] == 0
    capture = written(draft, cfg)
    assert capture.outcome == 'COMPLETE' and len(capture.observations) == 2
    depth = capture.observations[1]
    assert depth.data.unit_scale_m == 0.001 and depth.stamp.origin == 'HEADER' and depth.data.frame_id == FRAME


@pytest.mark.parametrize('offset_ms, status, outcome', [(3, 'WITHIN_TOLERANCE', 'COMPLETE'),
                                                        (40, 'OUT_OF_TOLERANCE', 'PARTIAL')])
def test_skew_against_the_reference(written, offset_ms, status, outcome):
    c, clock, cfg = make()
    feed_infos(c)
    c.fire(trigger(), B)
    c.on_sample(COLOR, sample(c, clock, B + 1_000_000, rgb(), 'rgb8'))
    c.on_sample(DEPTH, sample(c, clock, B + 1_000_000 + offset_ms * 1_000_000, depth_mm(), '16UC1'))
    draft = run(c, clock)
    assert (draft.sync['status'], draft.outcome, draft.sync['max_skew_ns']) == (status, outcome, offset_ms * 1_000_000)
    assert written(draft, cfg).outcome == outcome


def test_the_first_sample_at_or_after_the_boundary_is_chosen(written):
    c, clock, cfg = make()
    feed_infos(c)
    for stamp, seed in ((B - 10_000_000, 1), (B + 20_000_000, 2), (B + 3_000_000, 3)):
        c.on_sample(COLOR, sample(c, clock, stamp, rgb(seed), 'rgb8'))
        c.on_sample(DEPTH, sample(c, clock, stamp, depth_mm(seed), '16UC1'))
    c.fire(trigger(), B)
    draft = run(c, clock)
    assert [s.stamp.nanosec for s in draft.samples] == [3_000_000, 3_000_000]
    assert np.array_equal(draft.samples[0].array, rgb(3))


def test_a_missing_stream_times_out_on_the_host_clock(written):
    c, clock, cfg = make()
    feed_infos(c)
    c.fire(trigger(), B)
    c.on_sample(COLOR, sample(c, clock, B + 1, rgb(), 'rgb8'))
    assert c.poll() == []                                   # still waiting for depth
    clock.advance(999)
    assert c.poll() == []
    clock.advance(2)                                        # the monotonic deadline passes
    draft = c.poll()[0]
    assert draft.outcome == 'PARTIAL' and reasons(draft) == {DEPTH: 'TIMEOUT'}
    assert draft.sync['status'] == 'SINGLE_STREAM'
    assert written(draft, cfg).outcome == 'PARTIAL'


def test_nothing_at_all_is_failed_not_dropped(written):
    c, clock, cfg = make()
    c.fire(trigger(), B)
    draft = run(c, clock)
    assert draft.outcome == 'FAILED' and reasons(draft) == {COLOR: 'TIMEOUT', DEPTH: 'TIMEOUT'}
    assert draft.sync is None and written(draft, cfg).outcome == 'FAILED'


def test_a_stopped_session_clock_is_clock_unavailable(written):
    c, clock, cfg = make()
    pending = c.fire(trigger(0), 0)
    draft = c.poll()[0]                                     # decided at once
    assert draft.outcome == 'FAILED' and set(reasons(draft).values()) == {'CLOCK_UNAVAILABLE'}
    assert pending is draft.pending and written(draft, cfg).outcome == 'FAILED'


def test_unstamped_samples_are_never_given_their_receipt_time(written):
    c, clock, cfg = make()
    feed_infos(c)
    c.fire(trigger(), B)
    c.on_sample(COLOR, sample(c, clock, B + 1, rgb(), 'rgb8'))
    c.on_sample(DEPTH, sample(c, clock, 0, depth_mm(), '16UC1'))
    draft = run(c, clock)
    assert reasons(draft) == {DEPTH: 'CLOCK_UNAVAILABLE'} and draft.outcome == 'PARTIAL'
    written(draft, cfg)


def test_a_sample_far_after_the_boundary_is_clock_mismatch(written):
    c, clock, cfg = make()
    feed_infos(c)
    c.fire(trigger(), B)
    c.on_sample(COLOR, sample(c, clock, B + 1, rgb(), 'rgb8'))
    c.on_sample(DEPTH, sample(c, clock, B + 1_500_000_000, depth_mm(), '16UC1'))
    draft = c.poll()[0]                                     # decided without waiting for the timeout
    assert reasons(draft) == {DEPTH: 'CLOCK_MISMATCH'}
    written(draft, cfg)


def test_a_sample_on_another_clock_is_never_compared(written):
    c, clock, cfg = make()
    feed_infos(c)
    c.fire(trigger(), B)
    c.on_sample(COLOR, sample(c, clock, B + 1, rgb(), 'rgb8'))
    c.on_sample(DEPTH, sample(c, clock, B + 1, depth_mm(), '16UC1', clock_key=('SIM_TIME', 'isaac')))
    draft = c.poll()[0]
    assert reasons(draft) == {DEPTH: 'INCOMPATIBLE_CLOCKS'} and draft.sync['status'] == 'INCOMPATIBLE_CLOCKS'
    assert 'max_skew_ns' not in draft.sync and draft.outcome == 'PARTIAL'
    written(draft, cfg)


@pytest.mark.parametrize('setup, reason', [
    (lambda c: c.on_info(COLOR, Info(conversions.intrinsics(info_msg()), FRAME)), 'NO_CAMERA_INFO'),
    (lambda c: feed_infos(c, frame='overhead_camera_depth_optical_frame'), 'FRAME_MISMATCH'),
    (lambda c: feed_infos(c, depth_fx=4.5), 'REGISTRATION_MISMATCH'),
    (lambda c: (feed_infos(c), c.on_info(DEPTH, Info(conversions.intrinsics(info_msg(width=9)), FRAME))),
     'CALIBRATION_MISMATCH'),
])
def test_depth_contract_violations_are_named(written, setup, reason):
    c, clock, cfg = make()
    setup(c)
    c.fire(trigger(), B)
    c.on_sample(COLOR, sample(c, clock, B + 1, rgb(), 'rgb8'))
    c.on_sample(DEPTH, sample(c, clock, B + 1, depth_mm(), '16UC1'))
    draft = run(c, clock)
    assert reasons(draft) == {DEPTH: reason} and draft.outcome == 'PARTIAL'
    written(draft, cfg)


@pytest.mark.parametrize('frame, array, encoding, reason', [
    ('overhead_camera_depth_optical_frame', depth_mm(), '16UC1', 'FRAME_MISMATCH'),   # never rewritten
    (FRAME, rgb()[:, :, 0], 'mono8', 'ENCODING_REFUSED'),
])
def test_image_violations_are_named(written, frame, array, encoding, reason):
    c, clock, cfg = make()
    feed_infos(c)
    c.fire(trigger(), B)
    c.on_sample(COLOR, sample(c, clock, B + 1, rgb(), 'rgb8'))
    c.on_sample(DEPTH, sample(c, clock, B + 1, array, encoding, frame=frame))
    draft = run(c, clock)
    assert reasons(draft) == {DEPTH: reason}
    written(draft, cfg)


def test_tf_is_looked_up_at_the_observation_stamp_and_waited_for(written):
    calls, available = [], {'yes': False}

    def tf(parent, child, sec, nanosec):
        calls.append((parent, child, sec, nanosec))
        return ((0.5, 0.0, 0.5), (0.0, 0.0, 0.0, 1.0)) if available['yes'] else None
    doc = config()
    doc['sensors'][0].update(extrinsic_parent_frame='base_link', pose_parent_frame='world')
    c, clock, cfg = make(doc, tf)
    feed_infos(c)
    c.fire(trigger(), B)
    c.on_sample(COLOR, sample(c, clock, B + 7, rgb(), 'rgb8'))
    c.on_sample(DEPTH, sample(c, clock, B + 7, depth_mm(), '16UC1'))
    assert c.poll() == []                                   # TF not there yet: wait, never "latest"
    available['yes'] = True
    draft = c.poll()[0]
    assert {call[2:] for call in calls} == {(100, 7)}
    assert draft.calibrations[0].extrinsics[0].parent_frame == 'base_link'
    assert draft.samples[0].pose_at_stamp.parent_frame == 'world'
    capture = written(draft, cfg)
    assert capture.observations[0].data.pose_at_stamp is not None


def test_tf_that_never_arrives_is_tf_unavailable(written):
    doc = config()
    doc['sensors'][0]['extrinsic_parent_frame'] = 'base_link'
    c, clock, cfg = make(doc, lambda *a: None)
    feed_infos(c)
    c.fire(trigger(), B)
    c.on_sample(COLOR, sample(c, clock, B + 1, rgb(), 'rgb8'))
    c.on_sample(DEPTH, sample(c, clock, B + 1, depth_mm(), '16UC1'))
    draft = run(c, clock)
    assert set(reasons(draft).values()) == {'TF_UNAVAILABLE'} and draft.outcome == 'FAILED'
    written(draft, cfg)


def test_a_camera_info_change_is_a_new_calibration_never_merged(written):
    c, clock, cfg = make()
    ids = []
    for i, fx in enumerate((4.0, 4.0, 4.2)):
        c.on_info(COLOR, Info(conversions.intrinsics(info_msg(fx=fx)), FRAME))
        c.on_info(DEPTH, Info(conversions.intrinsics(info_msg(fx=fx)), FRAME))
        boundary = B + i * SEC
        c.fire(trigger(boundary, i), boundary)
        c.on_sample(COLOR, sample(c, clock, boundary + 1, rgb(i), 'rgb8'))
        c.on_sample(DEPTH, sample(c, clock, boundary + 1, depth_mm(i), '16UC1'))
        draft = run(c, clock)
        ids.append(draft.samples[0].calibration_id)
        written(draft, cfg)
        assert c.calibration_changes == 0                   # nothing counted before the write
        c.written(draft)                                    # counted only once the writer committed it
    assert ids[0] == ids[1] != ids[2]
    assert c.calibration_changes == 2                       # colour and depth each changed once (per stream)


def test_too_many_pending_triggers_overrun():
    c, clock, cfg = make()
    for i in range(cfg['writer']['max_pending']):
        c.fire(trigger(B, i), B)
    with pytest.raises(Overrun):
        c.fire(trigger(B, 9), B)


def test_sim_time_samples_carry_the_sim_source(written):
    c, clock, cfg = make(config(clock={'sim_source': 'isaac'}), use_sim_time=True)
    feed_infos(c)
    t = trigger()
    t['issued']['stamp'].update(clock_domain='SIM_TIME', sim_source='isaac')
    c.fire(t, B)
    c.on_sample(COLOR, sample(c, clock, B + 1, rgb(), 'rgb8'))
    c.on_sample(DEPTH, sample(c, clock, B + 1, depth_mm(), '16UC1'))
    draft = run(c, clock)
    assert draft.samples[0].stamp.clock == ('SIM_TIME', 'isaac')
    assert written(draft, cfg).observations[0].stamp.sim_source == 'isaac'


# ---- review fixes (D0.2 independent review) ---------------------------------------------------------------

def test_a_sample_exactly_at_the_boundary_is_selected(written):
    c, clock, cfg = make()
    feed_infos(c)
    c.fire(trigger(), B)
    c.on_sample(COLOR, sample(c, clock, B, rgb(), 'rgb8'))
    c.on_sample(DEPTH, sample(c, clock, B, depth_mm(), '16UC1'))
    assert c.poll()[0].outcome == 'COMPLETE'


def test_float_depth_is_metres(written):
    c, clock, cfg = make()
    feed_infos(c)
    c.fire(trigger(), B)
    c.on_sample(COLOR, sample(c, clock, B + 1, rgb(), 'rgb8'))
    c.on_sample(DEPTH, sample(c, clock, B + 1, np.full((6, 8), 1.5, np.float32), '32FC1'))
    draft = run(c, clock)
    assert draft.samples[1].unit_scale_m == 1.0
    assert written(draft, cfg).observations[1].data.unit_scale_m == 1.0


def test_an_assembly_error_never_wedges_the_coordinator(written):
    c, clock, cfg = make()
    bad = conversions.intrinsics(info_msg())
    bad['K'][0] = float('nan')                              # a non-finite CameraInfo that slipped through
    c.on_info(COLOR, Info(bad, FRAME))
    c.on_info(DEPTH, Info(conversions.intrinsics(info_msg()), FRAME))
    first = c.fire(trigger(B, 0), B)
    second = c.fire(trigger(B + 1, 1), B + 1)
    c.on_sample(COLOR, sample(c, clock, B + 2, rgb(), 'rgb8'))
    c.on_sample(DEPTH, sample(c, clock, B + 2, depth_mm(), '16UC1'))
    drafts = c.poll()
    assert [d.pending for d in drafts] == [first, second] and c.pending == []
    # colour: no calibration can be formed; depth: its info is not identical to colour's (NaN != NaN)
    assert reasons(drafts[0]) == {COLOR: 'CALIBRATION_MISMATCH', DEPTH: 'REGISTRATION_MISMATCH'}
    assert drafts[0].outcome == 'FAILED'
    written(drafts[0], cfg)


def test_a_clock_jumping_back_drops_the_stale_buffer(written):
    c, clock, cfg = make()
    feed_infos(c)
    for stamp in (B + 100_000_000, B + 800_000_000):        # the old timeline
        c.on_sample(COLOR, sample(c, clock, stamp, rgb(), 'rgb8'))
        c.on_sample(DEPTH, sample(c, clock, stamp, depth_mm(), '16UC1'))
    c.on_clock_jump()                                       # simulation reset
    feed_infos(c)
    c.fire(trigger(B + 50_000_000), B + 50_000_000)
    draft = run(c, clock)
    assert set(reasons(draft).values()) == {'TIMEOUT'}      # never the stale frame, never a false COMPLETE


def test_a_publisher_behind_the_session_clock_is_clock_mismatch(written):
    c, clock, cfg = make()
    feed_infos(c)
    c.fire(trigger(), B)
    clock.advance(1)
    for _ in range(3):
        c.on_sample(COLOR, sample(c, clock, B - 50 * SEC, rgb(), 'rgb8'))
        c.on_sample(DEPTH, sample(c, clock, B - 50 * SEC, depth_mm(), '16UC1'))
    draft = run(c, clock)
    assert set(reasons(draft).values()) == {'CLOCK_MISMATCH'} and 'behind' in draft.missing[0]['detail']


def test_an_old_unstamped_sample_does_not_turn_a_timeout_into_clock_unavailable(written):
    c, clock, cfg = make()
    feed_infos(c)
    c.on_sample(DEPTH, sample(c, clock, 0, depth_mm(), '16UC1'))    # received BEFORE the trigger
    clock.advance(1)
    c.fire(trigger(), B)
    c.on_sample(COLOR, sample(c, clock, B + 1, rgb(), 'rgb8'))
    assert reasons(run(c, clock)) == {DEPTH: 'TIMEOUT'}


def test_each_image_gets_the_camera_info_of_its_own_stamp(written):
    c, clock, cfg = make()
    for fx, stamp in ((4.0, B + 1), (4.2, B + 2 * SEC)):
        c.on_info(COLOR, Info(conversions.intrinsics(info_msg(fx=fx)), FRAME, stamp))
        c.on_info(DEPTH, Info(conversions.intrinsics(info_msg(fx=fx)), FRAME, stamp))
    c.fire(trigger(), B)
    c.on_sample(COLOR, sample(c, clock, B + 1, rgb(), 'rgb8'))
    c.on_sample(DEPTH, sample(c, clock, B + 1, depth_mm(), '16UC1'))
    draft = c.poll()[0]
    assert draft.outcome == 'COMPLETE' and draft.calibrations[0].intrinsics.K[0] == 4.0   # not the newer 4.2


def test_a_camera_info_not_yet_arrived_is_waited_for(written):
    c, clock, cfg = make()
    c.fire(trigger(), B)
    c.on_sample(COLOR, sample(c, clock, B + 1, rgb(), 'rgb8'))
    c.on_sample(DEPTH, sample(c, clock, B + 1, depth_mm(), '16UC1'))
    assert c.poll() == []                                   # wait: the infos may still come
    c.on_info(COLOR, Info(conversions.intrinsics(info_msg()), FRAME, B + 1))
    c.on_info(DEPTH, Info(conversions.intrinsics(info_msg()), FRAME, B + 1))
    assert c.poll()[0].outcome == 'COMPLETE'


def test_extrinsic_tf_noise_keeps_one_calibration_a_real_move_does_not(written):
    offsets = iter([0.0, 1e-9, 0.01])

    def tf(parent, child, sec, nanosec):
        return ((0.5 + tf.dx, 0.0, 0.5), (0.0, 0.0, 0.0, 1.0))
    doc = config()
    doc['sensors'][0]['extrinsic_parent_frame'] = 'base_link'
    c, clock, cfg = make(doc, tf)
    feed_infos(c)
    ids = []
    for i in range(3):
        tf.dx = next(offsets)
        c.fire(trigger(B + i * SEC, i), B + i * SEC)
        c.on_sample(COLOR, sample(c, clock, B + i * SEC + 1, rgb(), 'rgb8'))
        c.on_sample(DEPTH, sample(c, clock, B + i * SEC + 1, depth_mm(), '16UC1'))
        draft = run(c, clock)
        ids.append(draft.samples[0].calibration_id)
        c.written(draft)
    assert ids[0] == ids[1] != ids[2] and c.calibration_changes == 2


def test_closing_assembles_every_pending_capture_at_once(written):
    c, clock, cfg = make()
    c.fire(trigger(), B)
    assert c.poll() == []
    c.expire_all()
    assert c.poll()[0].outcome == 'FAILED' and c.pending == []


def test_a_capture_pending_at_a_reset_never_mixes_the_two_timelines(written):
    c, clock, cfg = make()
    feed_infos(c)
    c.fire(trigger(), B)
    c.on_sample(COLOR, sample(c, clock, B + 1, rgb(), 'rgb8'))          # colour taken from the old timeline
    assert c.poll() == []
    assert c.on_clock_jump() == 1                                       # simulation reset
    feed_infos(c)
    c.on_sample(DEPTH, sample(c, clock, B + 1, depth_mm(), '16UC1'))    # a deterministic sim replays the stamp
    draft = c.poll()[0]
    assert reasons(draft) == {DEPTH: 'CLOCK_MISMATCH'} and draft.outcome == 'PARTIAL'
    written(draft, cfg)


# ---- a same-host source's stamp can never follow its receipt (D0-C finding, 2026-10-30) ------------------------

def received(coordinator, clock, stamp_ns, array, encoding, receipt_wall_ns):
    sec, nsec = divmod(stamp_ns, SEC)
    return Sample(sec, nsec, coordinator.clock_key, clock.ns, receipt_wall_ns, FRAME, encoding, array=array)


def test_a_bound_sensors_sample_stamped_after_its_receipt_is_clock_mismatch(written):
    c, clock, cfg = make()
    c.bind_to_receipt({'overhead_camera'})
    feed_infos(c)
    c.fire(trigger(), B)
    ahead = B + 300_000_000                                 # the driver's clock fit ran 300 ms ahead
    c.on_sample(COLOR, received(c, clock, ahead, rgb(), 'rgb8', receipt_wall_ns=B + 33_000_000))
    c.on_sample(DEPTH, received(c, clock, ahead, depth_mm(), '16UC1', receipt_wall_ns=B + 33_000_000))
    draft = c.poll()[0]                                     # decided at once: never COMPLETE
    assert reasons(draft) == {COLOR: 'CLOCK_MISMATCH', DEPTH: 'CLOCK_MISMATCH'}
    assert 'AFTER this host received it' in draft.missing[0]['detail']
    assert c.stamps_after_receipt == 2
    assert written(draft, cfg).outcome == 'FAILED'
    c.reset_counters()
    assert c.stamps_after_receipt == 0


def test_a_bound_sensors_ordinary_latency_is_complete_and_one_bad_stream_is_partial(written):
    c, clock, cfg = make()
    c.bind_to_receipt({'overhead_camera'})
    feed_infos(c)
    c.fire(trigger(), B)
    c.on_sample(COLOR, received(c, clock, B + 1, rgb(), 'rgb8', receipt_wall_ns=B + 33_000_000))   # healthy
    c.on_sample(DEPTH, received(c, clock, B + 1, depth_mm(), '16UC1', receipt_wall_ns=B + 33_000_000))
    assert written(run(c, clock), cfg).outcome == 'COMPLETE'
    c.fire(trigger(B + SEC, 1), B + SEC)
    c.on_sample(COLOR, received(c, clock, B + SEC + 1, rgb(), 'rgb8', receipt_wall_ns=B + SEC + 33_000_000))
    c.on_sample(DEPTH, received(c, clock, B + SEC + 1, depth_mm(), '16UC1', receipt_wall_ns=B + SEC - 300_000_000))
    draft = run(c, clock)
    assert reasons(draft) == {DEPTH: 'CLOCK_MISMATCH'} and written(draft, cfg).outcome == 'PARTIAL'


def test_an_unbound_sensor_is_never_judged_against_its_receipt(written):
    c, clock, cfg = make()                                  # remote, unknown or simulated: another clock
    feed_infos(c)
    c.fire(trigger(), B)
    c.on_sample(COLOR, received(c, clock, B + 300_000_000, rgb(), 'rgb8', receipt_wall_ns=B))
    c.on_sample(DEPTH, received(c, clock, B + 300_000_000, depth_mm(), '16UC1', receipt_wall_ns=B))
    assert written(run(c, clock), cfg).outcome == 'COMPLETE' and c.stamps_after_receipt == 0


def test_the_receipt_margin_is_the_boundary_of_the_rule(written):
    from trainit_data_ros.coordinator import RECEIPT_MARGIN_NS
    c, clock, cfg = make()
    c.bind_to_receipt({'overhead_camera'})
    feed_infos(c)
    c.fire(trigger(), B)
    c.on_sample(COLOR, received(c, clock, B + RECEIPT_MARGIN_NS, rgb(), 'rgb8', receipt_wall_ns=B))      # at margin
    c.on_sample(DEPTH, received(c, clock, B + RECEIPT_MARGIN_NS + 1, depth_mm(), '16UC1', receipt_wall_ns=B))
    assert reasons(run(c, clock)) == {DEPTH: 'CLOCK_MISMATCH'}


@pytest.mark.parametrize('domain, sim, relation, bound', [
    ('ROS_TIME', False, 'LOCAL_SENSOR_LOCAL_RECORDER', {'cam'}),
    ('ROS_TIME', True, 'LOCAL_SENSOR_LOCAL_RECORDER', set()),          # sim time: not this host's clock
    ('SIM_TIME', False, 'LOCAL_SENSOR_LOCAL_RECORDER', set()),
    ('ROS_TIME', False, 'REMOTE_SENSOR_SOURCE', set()),                # another host's clock
    ('ROS_TIME', False, 'UNKNOWN', set()),                             # no evidence (a replay, no document)
])
def test_only_a_proven_same_host_source_on_ros_time_is_bound(domain, sim, relation, bound):
    from types import SimpleNamespace
    from trainit_data_ros.recorder_node import Recorder
    me = SimpleNamespace(cfg={'clock': {'session_domain': domain}}, use_sim_time=sim)
    assert Recorder._receipt_bound(me, {'sensors': {'cam': {'relation': relation}}}) == bound


# ---- FIRST_ALIGNED_AT_OR_AFTER_BOUNDARY v1 (owner decision 2026-11-05) -----------------------------------------
# Owner flow 2 (ses-20261104T163012Z-fe31fad1): Isaac's colour and depth share a render tick, but the depth frame of
# the tick after the trigger was lost in transport; v1 paired colour with the depth of a tick 50 ms later.

ALIGNED = {'policy': 'FIRST_ALIGNED_AT_OR_AFTER_BOUNDARY', 'policy_version': 1}
TICK = 16_666_667


def aligned(**sync):
    return make(config(sync=dict(ALIGNED, **sync)))


def both(c, clock, stamp, seed=1, streams=(COLOR, DEPTH)):
    if COLOR in streams:
        c.on_sample(COLOR, sample(c, clock, stamp, rgb(seed), 'rgb8'))
    if DEPTH in streams:
        c.on_sample(DEPTH, sample(c, clock, stamp, depth_mm(seed), '16UC1'))


@pytest.mark.parametrize('policy, tolerance_ms, kept, status, outcome', [
    ('FIRST_AT_OR_AFTER_BOUNDARY', None, (1, 3), 'OUT_OF_TOLERANCE', 'PARTIAL'),
    # the default tolerance (half a 30 fps frame) accepts the neighbouring 60 Hz render tick, and says so
    ('FIRST_ALIGNED_AT_OR_AFTER_BOUNDARY', None, (2, 3), 'WITHIN_TOLERANCE', 'COMPLETE'),
    # a tolerance below one tick demands the same tick: the simulator's identical stamps
    ('FIRST_ALIGNED_AT_OR_AFTER_BOUNDARY', 1.0, (3, 3), 'SYNCED', 'COMPLETE')])
def test_a_lost_depth_frame_is_paired_by_v1_and_skipped_by_the_aligned_policy(written, policy, tolerance_ms, kept,
                                                                              status, outcome):
    sync = {'policy': policy, 'policy_version': 1}
    if tolerance_ms is not None:
        sync['tolerance_ms'] = tolerance_ms
    c, clock, cfg = make(config(sync=sync))
    assert cfg['sync']['tolerance_ms'] == (tolerance_ms or 1000.0 / 30 / 2)
    feed_infos(c)
    c.fire(trigger(), B)
    both(c, clock, B + 1 * TICK, 1, (COLOR,))                    # its depth twin is lost
    both(c, clock, B + 2 * TICK, 2, (COLOR,))                    # this one too
    both(c, clock, B + 3 * TICK, 3)
    draft = run(c, clock)
    assert tuple(round(s.stamp.nanosec / TICK) for s in draft.samples) == kept
    assert (draft.sync['status'], draft.outcome) == (status, outcome)
    assert draft.sync['policy']['name'] == policy
    capture = written(draft, cfg)
    assert capture.outcome == outcome and capture.sync.policy.name == policy


def test_a_late_partner_is_waited_for_not_skipped(written):
    c, clock, cfg = aligned()
    feed_infos(c)
    c.fire(trigger(), B)
    both(c, clock, B + 1 * TICK, 1, (COLOR,))
    both(c, clock, B + 2 * TICK, 2, (COLOR,))                    # colour runs ahead of depth
    assert c.poll() == []                                        # depth has not passed tick 1: it may still come
    both(c, clock, B + 1 * TICK, 1, (DEPTH,))
    draft = run(c, clock)
    assert [s.stamp.nanosec for s in draft.samples] == [TICK, TICK] and draft.sync['status'] == 'SYNCED'


def test_the_nearest_partner_within_tolerance_is_taken_and_an_identical_stamp_wins(written):
    c, clock, cfg = aligned(tolerance_ms=10.0)
    feed_infos(c)
    c.fire(trigger(), B)
    c.on_sample(COLOR, sample(c, clock, B + 5_000_000, rgb(), 'rgb8'))
    c.on_sample(DEPTH, sample(c, clock, B + 1_000_000, depth_mm(1), '16UC1'))     # 4 ms before
    c.on_sample(DEPTH, sample(c, clock, B + 8_000_000, depth_mm(2), '16UC1'))     # 3 ms after: nearer
    c.on_sample(DEPTH, sample(c, clock, B + 30_000_000, depth_mm(3), '16UC1'))
    draft = run(c, clock)
    assert [s.stamp.nanosec for s in draft.samples] == [5_000_000, 8_000_000]
    assert (draft.sync['status'], draft.outcome, draft.sync['max_skew_ns']) == ('WITHIN_TOLERANCE', 'COMPLETE',
                                                                                3_000_000)
    assert written(draft, cfg).outcome == 'COMPLETE'
    c.fire(trigger(index=1), B + 50_000_000)
    c.on_sample(COLOR, sample(c, clock, B + 60_000_000, rgb(), 'rgb8'))
    c.on_sample(DEPTH, sample(c, clock, B + 57_000_000, depth_mm(), '16UC1'))
    c.on_sample(DEPTH, sample(c, clock, B + 60_000_000, depth_mm(), '16UC1'))
    c.on_sample(DEPTH, sample(c, clock, B + 70_000_000, depth_mm(), '16UC1'))
    assert run(c, clock).sync['status'] == 'SYNCED'


def test_without_an_aligned_pair_in_time_the_frames_are_kept_and_said_out_of_tolerance(written):
    c, clock, cfg = aligned()
    feed_infos(c)
    c.fire(trigger(), B)
    both(c, clock, B + 1 * TICK, 1, (COLOR,))
    c.on_sample(DEPTH, sample(c, clock, B + 1 * TICK + 40_000_000, depth_mm(), '16UC1'))   # never closer
    draft = run(c, clock)
    assert draft.outcome == 'PARTIAL' and draft.sync['status'] == 'OUT_OF_TOLERANCE' and not draft.missing
    assert draft.sync['policy']['name'] == 'FIRST_ALIGNED_AT_OR_AFTER_BOUNDARY'
    assert written(draft, cfg).outcome == 'PARTIAL'


def test_a_silent_stream_under_the_aligned_policy_times_out_and_the_other_is_kept(written):
    c, clock, cfg = aligned()
    feed_infos(c)
    c.fire(trigger(), B)
    both(c, clock, B + TICK, 1, (COLOR,))
    draft = run(c, clock)
    assert draft.outcome == 'PARTIAL' and reasons(draft) == {DEPTH: 'TIMEOUT'}
    assert [s.stream for s in draft.samples] == ['color']
    assert written(draft, cfg).outcome == 'PARTIAL'


def test_a_stream_on_another_clock_is_decided_at_once_under_the_aligned_policy(written):
    c, clock, cfg = aligned()
    feed_infos(c)
    c.fire(trigger(), B)
    both(c, clock, B + TICK, 1, (COLOR,))
    c.on_sample(DEPTH, sample(c, clock, B + TICK, depth_mm(), '16UC1', clock_key=('SIM_TIME', 'isaac')))
    draft = c.poll()                                             # no waiting for the deadline
    assert len(draft) == 1 and reasons(draft[0]) == {DEPTH: 'INCOMPATIBLE_CLOCKS'}
    assert draft[0].sync['status'] == 'INCOMPATIBLE_CLOCKS'


def test_at_the_deadline_a_group_waiting_for_a_nearer_partner_keeps_its_nearest_one(written):
    """Independent review of 1797fea: colour at +5 ms has depth at +1 and +4 ms within 5 ms; the next depth (maybe
    +5 ms) never comes. The deadline keeps depth +4 (1 ms), never v1's +1 ms."""
    c, clock, cfg = aligned(tolerance_ms=5.0)
    feed_infos(c)
    c.fire(trigger(), B)
    c.on_sample(DEPTH, sample(c, clock, B + 1_000_000, depth_mm(1), '16UC1'))
    c.on_sample(DEPTH, sample(c, clock, B + 4_000_000, depth_mm(2), '16UC1'))
    c.on_sample(COLOR, sample(c, clock, B + 5_000_000, rgb(), 'rgb8'))
    assert c.poll() == []                                        # depth +5 ms may still come: waited for
    draft = run(c, clock)
    assert [s.stamp.nanosec for s in draft.samples] == [5_000_000, 4_000_000]
    assert draft.sync['status'] == 'WITHIN_TOLERANCE' and draft.sync['max_skew_ns'] == 1_000_000
    assert draft.outcome == 'COMPLETE'
    assert written(draft, cfg).outcome == 'COMPLETE'


@pytest.mark.parametrize('depths', [(0.5,), (0.5, 1.0), (0.2, 0.9, 3.0), (1.0,), (1.0, 4.0), (2.5,)])
def test_where_v1_is_complete_the_aligned_policy_keeps_its_reference_and_never_a_larger_skew(written, depths):
    kept = {}
    for policy in ('FIRST_AT_OR_AFTER_BOUNDARY', 'FIRST_ALIGNED_AT_OR_AFTER_BOUNDARY'):
        c, clock, cfg = make(config(sync={'policy': policy, 'policy_version': 1, 'tolerance_ms': 5.0}))
        feed_infos(c)
        c.fire(trigger(), B)
        c.on_sample(COLOR, sample(c, clock, B + 1_000_000, rgb(), 'rgb8'))
        for i, ms in enumerate(depths):
            c.on_sample(DEPTH, sample(c, clock, B + int(ms * 1_000_000), depth_mm(i + 1), '16UC1'))
        kept[policy] = run(c, clock)
    v1, aligned_ = kept['FIRST_AT_OR_AFTER_BOUNDARY'], kept['FIRST_ALIGNED_AT_OR_AFTER_BOUNDARY']
    assert v1.outcome == aligned_.outcome == 'COMPLETE'
    assert [s.stamp.nanosec for s in v1.samples][0] == [s.stamp.nanosec for s in aligned_.samples][0]
    assert aligned_.sync['max_skew_ns'] <= v1.sync['max_skew_ns']
