"""D0.5 automated acceptance on MOCK: the runner's ExecutionEvent -> EVENT / APPLICATION_STATE / pick-attempt
triggers -> CaptureCoordinator -> canonical captures -> verify (private domain 87).

The camera is the hardware layer's mock (the camera contract); the events are real
``trainit_interfaces/ExecutionEvent`` messages on the RELATIVE ``trainit/execution_events`` of the recorder's slot,
with TMR's QoS (RELIABLE, VOLATILE, keep-last 100), published exactly as TMR's observer emits them (its sequence is
pinned by TMR's test_d0_5_execution_events). The letters are the owner's D0.5 acceptance list:

  A  a matching EVENT fires exactly once          F  SUCCEEDED / FAILED transitions
  B  a non-matching EVENT does not fire           G  the pick-attempt context (attempt, candidate) persisted
  C  a stale event does not fire                  H  event time and recorder receipt kept apart
  D  APPLICATION_STATE task ENTER                 I  the application context persisted
  E  APPLICATION_STATE step ENTER                 J  no subscription to a process topic, ever

plus the session-safety cases: an event before the session, a run already going (never joined mid-way), another
run while one is open, a repeat, a restarted runner (a new run), a runner that never ended (superseded), an event of
an ended run, a recorder that starts after the application did.
"""
import time
from pathlib import Path

import pytest

rclpy = pytest.importorskip('rclpy', reason='the acquisition tests need a sourced ROS 2 Humble')
pytest.importorskip('trainit_interfaces.msg', reason='D0.5 needs trainit_interfaces >= 0.3.0 (ExecutionEvent)')

from ros_graph import Graph, static_tf, wait_for  # noqa: E402
from test_ros_acquisition import SERVICE, SLOT, cli_verify, mock_camera, recording, start_recorder  # noqa: E402
from trainit_data.integrity import verify_session  # noqa: E402
from trainit_data.store import Store  # noqa: E402

SEC = 1_000_000_000
RUN_A, RUN_B = 'run-20261030T120000Z-0000000a', 'run-20261030T120500Z-0000000b'
RUN_C, RUN_D = 'run-20261030T121000Z-0000000c', 'run-20261030T121500Z-0000000d'
TRIGGERS = [
    {'id': 'manual', 'kind': 'MANUAL'},
    {'id': 'glue_on', 'kind': 'EVENT', 'process_channel': 'glue', 'edge': 'rising'},
    {'id': 'cycle_started', 'kind': 'APPLICATION_STATE', 'task': 'cycle', 'transition': 'enter'},
    {'id': 'approach', 'kind': 'APPLICATION_STATE', 'task': 'cycle', 'step': 'approach', 'transition': 'enter'},
    {'id': 'pick_failed', 'kind': 'APPLICATION_STATE', 'task': 'cycle', 'step': 'pick', 'transition': 'failed'},
    {'id': 'picked', 'kind': 'APPLICATION_STATE', 'task': 'cycle', 'step': 'pick', 'transition': 'succeeded',
     'context': {'pick_attempt': {'phase': 'post_grasp'}}},
]


@pytest.fixture
def graph():
    g = Graph()
    yield g
    g.close()


@pytest.fixture
def store_root(tmp_path):
    root = tmp_path / 'TrainItData'
    Store.open(root, create=True)
    return root


class FakeRunner:
    """Publishes what TMR's observer publishes, on the same relative topic and QoS, in the recorder's slot."""

    def __init__(self, graph, run_id=RUN_A):
        from rclpy.node import Node
        from rclpy.qos import QoSDurabilityPolicy, QoSHistoryPolicy, QoSProfile, QoSReliabilityPolicy
        from trainit_interfaces.msg import ExecutionEvent
        self.Msg = ExecutionEvent
        self.node = graph.add(Node('trainit_run_bt', namespace=SLOT, context=graph.ctx,
                                   start_parameter_services=False))
        qos = QoSProfile(reliability=QoSReliabilityPolicy.RELIABLE, durability=QoSDurabilityPolicy.VOLATILE,
                         history=QoSHistoryPolicy.KEEP_LAST, depth=100)
        self.pub = self.node.create_publisher(ExecutionEvent, 'trainit/execution_events', qos)
        self.run_id, self.seq, self.sent = run_id, 0, []

    def matched(self):
        return wait_for(lambda: self.pub.get_subscription_count() > 0, 10.0)

    def send(self, kind, *, offset_s=0.0, stamp_ns=None, seq=None, run_id=None, status=0, **fields):
        m = self.Msg()
        m.kind = getattr(self.Msg, kind)
        m.status = getattr(self.Msg, status) if isinstance(status, str) else status
        ns = stamp_ns if stamp_ns is not None else self.node.get_clock().now().nanoseconds + int(offset_s * SEC)
        m.header.stamp.sec, m.header.stamp.nanosec = divmod(ns, SEC)
        m.run_id = run_id or self.run_id
        if seq is None:
            self.seq += 1
            seq = self.seq
        m.seq = seq
        m.process_value = 0.0
        for key, value in fields.items():
            setattr(m, key, value)
        self.pub.publish(m)
        self.sent.append(m)
        return ns


def open_session(graph, rec):
    client = graph.helper('test_client')
    started = graph.call(client, f'{SERVICE}/start_session')
    assert started.success, started.message
    assert wait_for(lambda: all(v > 3 for v in rec.coordinator.received.values()), 10.0)
    return client, started.message


def close_session(graph, client, rec, fired):
    """Wait until ``fired`` captures were assembled, then seal (pending captures are never abandoned)."""
    assert wait_for(lambda: sum(rec.firings.values()) >= fired and not rec.coordinator.pending, 10.0), \
        dict(rec.firings)
    time.sleep(0.3)                                  # no further firing is expected: give a stray one the chance
    import json
    closed = graph.call(client, f'{SERVICE}/close_session')
    assert closed.success, closed.message
    return json.loads(closed.message)


def by_trigger(reader):
    out = {}
    for capture in reader.captures():
        out.setdefault(capture.trigger.trigger_id, []).append(capture)
    return out


def test_d0_5_mock_execution_events_fire_their_triggers_with_application_context(graph, tmp_path, store_root):
    mock_camera(graph)
    static_tf(graph)
    rec = start_recorder(graph, tmp_path, store_root, recording(triggers=TRIGGERS))
    runner = FakeRunner(graph)
    assert runner.matched()
    client, session_id = open_session(graph, rec)
    runner.send('RUN_STARTED')
    time.sleep(0.3)                                   # H: an event stamped 0.3 s before it is received, yet in-session
    cycle = runner.send('TASK_ENTERED', stamp_ns=rec._event_gate.session_start_ns + 1, task_id='cycle')
    runner.send('STEP_ENTERED', task_id='cycle', step_id='approach', step_kind='move')        # E
    glue = runner.send('PROCESS_EVENT', task_id='cycle', step_id='approach', step_kind='move',
                       process_channel='glue', process_kind='process_on', process_value=1.0)   # A
    runner.send('PROCESS_EVENT', process_channel='glue', process_kind='process_off', task_id='cycle')    # B: edge
    runner.send('PROCESS_EVENT', process_channel='light', process_kind='process_on', process_value=1.0,
                task_id='cycle')                                                                # B: channel
    runner.send('STEP_EXITED', status='SUCCEEDED', task_id='cycle', step_id='approach', step_kind='move')
    runner.send('STEP_ENTERED', task_id='cycle', step_id='pick', step_kind='grasp', attempt_index=1,
                candidate_id='c3', candidate_provider='synthetic')
    runner.send('STEP_EXITED', status='FAILED', task_id='cycle', step_id='pick', step_kind='grasp', attempt_index=1,
                candidate_id='c3', candidate_provider='synthetic')                              # F: failed
    runner.send('STEP_ENTERED', task_id='cycle', step_id='pick', step_kind='grasp', attempt_index=2,
                candidate_id='c4', candidate_provider='synthetic')
    runner.send('STEP_EXITED', status='SUCCEEDED', task_id='cycle', step_id='pick', step_kind='grasp',
                attempt_index=2, candidate_id='c4', candidate_provider='synthetic')            # F + G
    runner.send('TASK_EXITED', status='SUCCEEDED', task_id='cycle')
    runner.send('RUN_ENDED', status='SUCCEEDED')
    closed = close_session(graph, client, rec, fired=5)
    assert closed['warnings'] == []

    reader = Store.open(store_root).open_session(session_id)
    fired = by_trigger(reader)
    assert {k: len(v) for k, v in fired.items()} == {'cycle_started': 1, 'approach': 1, 'glue_on': 1,
                                                     'pick_failed': 1, 'picked': 1}               # A, B, D, E, F
    assert reader.header.counters['captures'] == {'attempted': 5, 'complete': 5, 'partial': 0, 'failed': 0}
    for captures in fired.values():
        capture = captures[0]
        assert capture.trigger.kind in ('EVENT', 'APPLICATION_STATE')
        assert capture.trigger.origin == 'EXECUTION_EVENT' and capture.trigger.issued.stamp.origin == 'HEADER'
        for obs in capture.observations:
            assert obs.stamp.to_ns() >= capture.trigger.issued.stamp.to_ns()       # the EVENT's time is the boundary
    started = fired['cycle_started'][0]                                                         # H
    assert started.trigger.issued.stamp.to_ns() == cycle
    assert started.trigger.issued.wall_utc_ns - cycle > 0.05 * SEC     # the recorder's receipt, after the event
    assert started.trigger.reason == {'event': {'run_id': RUN_A, 'seq': 2, 'kind': 'TASK_ENTERED',
                                                'task_id': 'cycle'}, 'transition': 'enter'}
    assert started.trigger.context == {'trainit.task_step': {'run_id': RUN_A, 'task_id': 'cycle'}}       # I
    approach = fired['approach'][0]
    assert approach.trigger.context == {'trainit.task_step': {'run_id': RUN_A, 'task_id': 'cycle',
                                                              'step_id': 'approach'}}
    glue_on = fired['glue_on'][0]
    assert glue_on.trigger.issued.stamp.to_ns() == glue and glue_on.trigger.reason['edge'] == 'rising'
    assert glue_on.trigger.reason['event']['process_channel'] == 'glue'
    assert glue_on.trigger.reason['event']['process_kind'] == 'process_on'
    assert fired['pick_failed'][0].trigger.reason['event']['status'] == 'FAILED'
    picked = fired['picked'][0].trigger.context                                                 # G
    assert picked['trainit.pick_attempt'] == {'phase': 'post_grasp', 'attempt_index': 2,
                                              'attempt_id': f'{RUN_A}/cycle/pick/2',
                                              'candidate': {'id': 'c4', 'provider': 'synthetic'}}
    assert picked['trainit.task_step'] == {'run_id': RUN_A, 'task_id': 'cycle', 'step_id': 'pick'}
    assert verify_session(Store.open(store_root), session_id).ok
    code, _ = cli_verify(store_root, session_id)
    assert code == 0


def test_events_of_another_time_or_another_run_never_fire(graph, tmp_path, store_root):
    mock_camera(graph)
    static_tf(graph)
    rec = start_recorder(graph, tmp_path, store_root, recording(triggers=TRIGGERS))
    runner = FakeRunner(graph)
    assert runner.matched()
    runner.send('RUN_STARTED')                                       # the application ran BEFORE the session
    runner.send('TASK_ENTERED', task_id='cycle')
    assert wait_for(lambda: rec.events_outside_session >= 2, 5.0)
    client, session_id = open_session(graph, rec)
    start_ns = rec._event_gate.session_start_ns
    runner.send('TASK_ENTERED', stamp_ns=start_ns - 1, task_id='cycle')               # C: happened before the start
    runner.send('TASK_ENTERED', task_id='cycle')                                       # run A, joined mid-way: never
    runner.send('RUN_ENDED', status='SUCCEEDED')
    runner.run_id, runner.seq = RUN_B, 0                                               # the runner restarted: run B
    runner.send('RUN_STARTED')
    runner.send('TASK_ENTERED', task_id='cycle')                                       # run B: fires
    runner.send('TASK_ENTERED', task_id='cycle', seq=runner.seq)                       # a repeat of B's last event
    runner.send('TASK_ENTERED', task_id='cycle', run_id=RUN_C, seq=5)                  # another run while B is open
    runner.run_id, runner.seq = RUN_D, 0                                               # B never ended: a crash
    runner.send('RUN_STARTED')                                                         # D supersedes B
    runner.send('TASK_ENTERED', task_id='cycle')                                       # run D: fires
    runner.send('TASK_EXITED', task_id='cycle', run_id=RUN_B, seq=9, status='SUCCEEDED')   # B after its end
    closed = close_session(graph, client, rec, fired=2)
    codes = {w['code']: w for w in closed['warnings']}
    assert codes['EXECUTION_EVENTS_REFUSED']['context'] == {
        'AMBIGUOUS_RUN': 1, 'DUPLICATE_EVENT': 1, 'ENDED_RUN': 1, 'STALE_EVENT': 1, 'UNSTARTED_RUN': 2}
    assert codes['EXECUTION_RUN_SUPERSEDED']['context'] == {'count': 1}
    assert codes['EXECUTION_TRIGGERS_NEVER_FIRED']['context'] == {
        'triggers': ['approach', 'glue_on', 'pick_failed', 'picked'], 'events_admitted': 4}
    captures = by_trigger(Store.open(store_root).open_session(session_id))['cycle_started']
    assert [c.trigger.reason['event']['run_id'] for c in captures] == [RUN_B, RUN_D]
    assert [c.trigger.firing_index for c in captures] == [0, 1]
    assert verify_session(Store.open(store_root), session_id).ok


def test_a_recorder_that_starts_after_the_application_receives_no_past_event(graph, tmp_path, store_root):
    mock_camera(graph)
    static_tf(graph)
    runner = FakeRunner(graph)
    runner.send('RUN_STARTED')                                       # nobody listens yet: VOLATILE keeps nothing
    runner.send('TASK_ENTERED', task_id='cycle')
    rec = start_recorder(graph, tmp_path, store_root, recording(triggers=TRIGGERS))
    assert runner.matched()
    client, session_id = open_session(graph, rec)
    time.sleep(1.0)
    closed = close_session(graph, client, rec, fired=0)
    assert closed['counters']['captures']['attempted'] == 0
    never = closed['warnings'][0]
    assert never['code'] == 'EXECUTION_TRIGGERS_NEVER_FIRED' and never['context']['events_admitted'] == 0
    assert 'no execution event of this session was admitted' in never['detail'] and len(closed['warnings']) == 1
    assert rec.events_outside_session == 0                          # not even delivered: the topic keeps no history


def test_the_recorder_listens_to_execution_events_and_never_to_a_process_topic(graph, tmp_path, store_root):
    rec = start_recorder(graph, tmp_path, store_root, recording(triggers=TRIGGERS))
    topics = {s.topic_name for s in rec.subscriptions}
    assert f'/{SLOT}/trainit/execution_events' in topics                                       # J
    assert not any('process' in t for t in topics), topics
    source = Path(__file__).resolve().parents[1] / 'trainit_data_ros'
    for path in source.glob('*.py'):                 # no process TOPIC name anywhere in the recorder's code
        text = path.read_text()
        assert 'process_enabled' not in text and '/process_value' not in text, path


def test_without_an_execution_trigger_the_recorder_does_not_subscribe(graph, tmp_path, store_root):
    rec = start_recorder(graph, tmp_path, store_root, recording())
    assert not any('execution_events' in s.topic_name for s in rec.subscriptions)
