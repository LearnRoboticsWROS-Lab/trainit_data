"""D0.5: EVENT and APPLICATION_STATE triggers from the runner's ExecutionEvent -- the configuration, the gate that
keeps events of another time or another run out of the ACTIVE session, the matching, and what a firing records
(no ROS: the events are the dicts ``event_from_message`` makes)."""
from types import SimpleNamespace

import pytest

from _ros_helpers import config, load_profile
from trainit_data.errors import DataError
from trainit_data.vocab import Vocabulary
from trainit_data_ros import execution_triggers as X
from trainit_data_ros.recording_config import resolve

VOCAB = Vocabulary.default()
SEC = 1_000_000_000
START = 100 * SEC                       # the session's start on the session clock
MISMATCH = SEC


def triggers(*specs):
    return resolve(config(triggers=[{'id': 'manual', 'kind': 'MANUAL'}, *specs]), VOCAB,
                   load_profile=load_profile, use_sim_time=False)['triggers'][1:]


def event(kind, *, seq=1, run='run-a', at=START + SEC, task='', step='', step_kind='', status=None, channel='',
          process_kind='', value=0.0, attempt=0, candidate=('', '')):
    return {'kind': kind, 'status': status, 'run_id': run, 'seq': seq, 'stamp_ns': at, 'task_id': task,
            'step_id': step, 'step_kind': step_kind, 'process_channel': channel, 'process_kind': process_kind,
            'process_value': value, 'attempt_index': attempt, 'candidate_id': candidate[0],
            'candidate_provider': candidate[1]}


# ---- configuration --------------------------------------------------------------------------------------------

def test_the_execution_triggers_resolve_with_their_own_fields():
    on_glue, approach, inspected, before_grasp = triggers(
        {'id': 'on_glue', 'kind': 'EVENT', 'process_channel': 'glue', 'edge': 'rising'},
        {'id': 'approach', 'kind': 'APPLICATION_STATE', 'task': 'pick_one', 'step': 'approach', 'transition': 'enter'},
        {'id': 'inspected', 'kind': 'APPLICATION_STATE', 'task': 'inspect', 'transition': 'succeeded'},
        {'id': 'before_grasp', 'kind': 'APPLICATION_STATE', 'task': 'pick_one', 'step': 'pick', 'transition': 'enter',
         'context': {'pick_attempt': {'phase': 'pre_grasp'}}})
    assert on_glue == {'id': 'on_glue', 'kind': 'EVENT', 'process_channel': 'glue', 'edge': 'rising'}
    assert approach['step'] == 'approach' and 'step' not in inspected
    assert before_grasp['context'] == {'pick_attempt': {'phase': 'pre_grasp'}}


@pytest.mark.parametrize('spec, needle', [
    ({'id': 't', 'kind': 'EVENT', 'process_channel': '/glue/process_enabled', 'edge': 'rising'}, 'process_channel'),
    ({'id': 't', 'kind': 'EVENT', 'process_channel': 'glue', 'edge': 'up'}, 'edge'),
    ({'id': 't', 'kind': 'EVENT', 'process_channel': 'glue', 'edge': 'any', 'task': 'x'}, 'task'),
    ({'id': 't', 'kind': 'APPLICATION_STATE', 'task': 'a', 'transition': 'started'}, 'transition'),
    ({'id': 't', 'kind': 'APPLICATION_STATE', 'task': 'a', 'transition': 'enter', 'process_channel': 'glue'},
     'process_channel'),
    ({'id': 't', 'kind': 'APPLICATION_STATE', 'task': 'a', 'transition': 'enter',
      'context': {'pick_attempt': {'phase': 'pre_grasp'}}}, 'grasp STEP'),
    ({'id': 't', 'kind': 'APPLICATION_STATE', 'task': 'a', 'step': 'g', 'transition': 'succeeded',
      'context': {'pick_attempt': {'phase': 'pre_grasp'}}}, 'phase pre_grasp'),
    ({'id': 't', 'kind': 'APPLICATION_STATE', 'task': 'a', 'step': 'g', 'transition': 'enter',
      'context': {'pick_attempt': {'phase': 'post_place'}}}, 'phase'),            # no canonical place attempt
    ({'id': 't', 'kind': 'APPLICATION_STATE', 'task': 'a', 'step': 'g', 'transition': 'enter',
      'context': {'outcome': {}}}, 'pick_attempt'),
    ({'id': 't', 'kind': 'MANUAL', 'process_channel': 'glue'}, 'process_channel'),
])
def test_a_malformed_execution_trigger_is_refused(spec, needle):
    with pytest.raises(DataError) as err:
        triggers(spec)
    assert needle in str(err.value)


# ---- the gate -------------------------------------------------------------------------------------------------

def started(gate, run='run-a', now=START + 2 * SEC):
    assert gate.admit(event('RUN_STARTED', seq=1, run=run), now) is None


def test_a_run_started_in_the_session_is_admitted_and_counted():
    gate = X.EventGate(START, MISMATCH)
    started(gate)
    assert gate.admit(event('TASK_ENTERED', seq=2, task='t'), START + 2 * SEC) is None
    assert gate.admitted == 2 and sum(gate.refused.values()) == 0 and gate.open_run == 'run-a'


def test_an_event_from_before_the_session_never_fires_however_late_it_arrives():
    gate = X.EventGate(START, MISMATCH)
    assert gate.admit(event('RUN_STARTED', at=START - 1), START + 50 * SEC) == 'STALE_EVENT'
    assert gate.admit(event('RUN_STARTED', at=START), START) is None                   # AT the start: this session


def test_an_event_far_ahead_of_the_session_clock_is_a_clock_mismatch():
    gate = X.EventGate(START, MISMATCH)
    assert gate.admit(event('RUN_STARTED', at=START + 3 * SEC), START + SEC) == 'CLOCK_MISMATCH'


def test_a_run_already_going_when_the_session_began_is_never_joined_mid_way():
    gate = X.EventGate(START, MISMATCH)
    now = START + 2 * SEC
    assert gate.admit(event('TASK_ENTERED', seq=7, task='t'), now) == 'UNSTARTED_RUN'
    assert gate.admit(event('RUN_STARTED', seq=3), now) == 'UNSTARTED_RUN'            # a start that is not seq 1
    assert gate.admit(event('RUN_ENDED', seq=8, status='SUCCEEDED'), now) == 'UNSTARTED_RUN'
    started(gate, run='run-b')                                                         # the next run: adopted


def test_another_run_while_one_is_open_is_ambiguous():
    gate = X.EventGate(START, MISMATCH)
    started(gate)
    assert gate.admit(event('TASK_ENTERED', seq=4, run='run-b', task='t'), START + 2 * SEC) == 'AMBIGUOUS_RUN'


def test_a_new_runner_start_supersedes_a_run_that_never_ended_and_closes_it():
    gate = X.EventGate(START, MISMATCH)
    now = START + 2 * SEC
    started(gate)                                                                      # run-a ... then a crash
    started(gate, run='run-b')                                                         # the restarted runner
    assert gate.superseded == 1 and gate.open_run == 'run-b'
    assert gate.admit(event('TASK_ENTERED', seq=2, run='run-a', task='t'), now) == 'ENDED_RUN'
    assert gate.admit(event('TASK_ENTERED', seq=2, run='run-b', task='t'), now) is None


def test_e1_sequential_job_runs_of_one_task_server_are_each_adopted():
    """P0.a case E1 (ADR-0030 Amendment P0): a robot task server makes ONE run per job in ONE process -- run ids
    "<server session>-<job counter>", seq from 1 in each, RUN_ENDED before the next RUN_STARTED. Every run is adopted
    from its own RUN_STARTED, none is a superseded crash, nothing is refused."""
    gate = X.EventGate(START, MISMATCH)
    at = START + SEC
    for n in (1, 2, 3):
        run = f'0a1b2c3d-{n}'
        assert gate.admit(event('RUN_STARTED', seq=1, run=run, at=at), at) is None
        assert gate.admit(event('TASK_ENTERED', seq=2, run=run, task='home', at=at + 1), at + 1) is None
        assert gate.admit(event('TASK_EXITED', seq=3, run=run, task='home', status='SUCCEEDED', at=at + 2),
                          at + 2) is None
        assert gate.admit(event('RUN_ENDED', seq=4, run=run, status='SUCCEEDED', at=at + 3), at + 3) is None
        assert gate.open_run is None
        at += SEC
    assert gate.admitted == 12 and gate.superseded == 0 and sum(gate.refused.values()) == 0


def test_e1_real_execution_events_carrying_the_p0a_job_fields_convert_and_are_adopted():
    """P0.a case E1 with the REAL message type (trainit_interfaces >= 0.4.0): the appended ``job_id`` /
    ``result_code`` of a task server's runs do not disturb event_from_message nor the gate -- three sequential runs,
    each adopted from its own RUN_STARTED (the data layer does not need the job fields: ADR-0030 Amendment P0)."""
    msgs = pytest.importorskip('trainit_interfaces.msg', reason='needs a built trainit_interfaces >= 0.4.0')
    E = msgs.ExecutionEvent
    assert hasattr(E(), 'job_id') and hasattr(E(), 'result_code'), 'trainit_interfaces < 0.4.0'
    gate = X.EventGate(START, MISMATCH)
    sec = START // SEC + 1
    for n in (1, 2, 3):
        for seq, (kind, status, task, code) in enumerate(((E.RUN_STARTED, E.STATUS_NONE, '', 0),
                                                          (E.TASK_ENTERED, E.STATUS_NONE, 'home', 0),
                                                          (E.TASK_EXITED, E.SUCCEEDED, 'home', 100),
                                                          (E.RUN_ENDED, E.SUCCEEDED, '', 100)), 1):
            m = E()
            m.header.stamp.sec, m.header.stamp.nanosec = sec, seq
            m.run_id, m.seq, m.kind, m.status, m.task_id = f'0a1b2c3d-{n}', seq, kind, status, task
            m.job_id, m.result_code = n, code
            e = X.event_from_message(m)
            assert e['run_id'] == f'0a1b2c3d-{n}' and e['seq'] == seq
            assert gate.admit(e, sec * SEC + seq) is None, (n, seq)
        sec += 1
    assert gate.admitted == 12 and gate.superseded == 0 and sum(gate.refused.values()) == 0


def test_an_event_of_an_ended_run_or_a_repeat_is_refused():
    gate = X.EventGate(START, MISMATCH)
    now = START + 2 * SEC
    started(gate)
    assert gate.admit(event('RUN_STARTED', seq=1), now) == 'DUPLICATE_EVENT'
    assert gate.admit(event('RUN_ENDED', seq=2, status='SUCCEEDED'), now) is None
    assert gate.admit(event('TASK_ENTERED', seq=3, task='t'), now) == 'ENDED_RUN'      # a late event of an ended run
    assert gate.open_run is None


def test_while_the_session_clock_reads_zero_nothing_fires_and_the_start_is_its_first_reading():
    gate = X.EventGate(0, MISMATCH)                                                    # sim time not yet running
    assert gate.admit(event('RUN_STARTED', at=5 * SEC), 0) == 'CLOCK_UNAVAILABLE'
    assert gate.admit(event('RUN_STARTED', at=5 * SEC), 10 * SEC) == 'STALE_EVENT'      # before the first reading
    assert gate.session_start_ns == 10 * SEC
    assert gate.admit(event('RUN_STARTED', at=11 * SEC), 11 * SEC) is None


def test_a_simulation_reset_rebases_the_start_and_old_timeline_events_never_fire():
    gate = X.EventGate(START, MISMATCH)
    started(gate)
    gate.rebase(3 * SEC)                                                               # the sim clock went back
    assert gate.rebases == 1 and gate.session_start_ns == 3 * SEC
    assert gate.admit(event('TASK_ENTERED', seq=2, at=START + SEC, task='t'), 4 * SEC) == 'CLOCK_MISMATCH'
    assert gate.admit(event('TASK_ENTERED', seq=3, at=2 * SEC, task='t'), 4 * SEC) == 'STALE_EVENT'
    assert gate.admit(event('TASK_ENTERED', seq=4, at=4 * SEC, task='t'), 4 * SEC) is None   # the run goes on


@pytest.mark.parametrize('broken', [dict(kind=None), dict(run_id=''), dict(seq=0), dict(stamp_ns=0)])
def test_a_malformed_event_is_refused(broken):
    gate = X.EventGate(START, MISMATCH)
    assert gate.admit(dict(event('RUN_STARTED'), **broken), START + SEC) == 'MALFORMED_EVENT'


def test_an_unknown_kind_from_a_newer_runner_is_malformed_never_guessed():
    msg = SimpleNamespace(kind=42, status=0, run_id='run-a', seq=1, header=SimpleNamespace(
        stamp=SimpleNamespace(sec=101, nanosec=0)), task_id='', step_id='', step_kind='', process_channel='',
        process_kind='', process_value=0.0, attempt_index=0, candidate_id='', candidate_provider='')
    e = X.event_from_message(msg)
    assert e['kind'] is None and e['stamp_ns'] == 101 * SEC
    assert X.EventGate(START, MISMATCH).admit(e, START + SEC) == 'MALFORMED_EVENT'


# ---- matching -------------------------------------------------------------------------------------------------

@pytest.mark.parametrize('edge, kinds', [('rising', {'process_on'}), ('falling', {'process_off'}),
                                         ('any', {'process_on', 'process_off', 'set_value', 'external'})])
def test_an_event_trigger_matches_its_channel_and_edge(edge, kinds):
    (trigger,) = triggers({'id': 't', 'kind': 'EVENT', 'process_channel': 'glue', 'edge': edge})
    for kind in ('process_on', 'process_off', 'set_value', 'external'):
        assert X.matches(trigger, event('PROCESS_EVENT', channel='glue', process_kind=kind)) == (kind in kinds)
    assert not X.matches(trigger, event('PROCESS_EVENT', channel='other', process_kind='process_on'))
    assert not X.matches(trigger, event('STEP_ENTERED', task='t', step='glue', step_kind='process'))


@pytest.mark.parametrize('transition, fires', [
    ('enter', ['ENTERED']), ('exit', ['SUCCEEDED', 'FAILED', 'HALTED']), ('succeeded', ['SUCCEEDED']),
    ('failed', ['FAILED'])])
def test_an_application_state_trigger_matches_its_task_or_step_and_transition(transition, fires):
    task_trigger, step_trigger = triggers(
        {'id': 'a', 'kind': 'APPLICATION_STATE', 'task': 'inspect', 'transition': transition},
        {'id': 'b', 'kind': 'APPLICATION_STATE', 'task': 'pick_one', 'step': 'approach', 'transition': transition})
    for what in ('ENTERED', 'SUCCEEDED', 'FAILED', 'HALTED'):
        status = None if what == 'ENTERED' else what
        task_e = event('TASK_ENTERED' if what == 'ENTERED' else 'TASK_EXITED', task='inspect', status=status)
        step_e = event('STEP_ENTERED' if what == 'ENTERED' else 'STEP_EXITED', task='pick_one', step='approach',
                       step_kind='move', status=status)
        assert X.matches(task_trigger, task_e) == (what in fires)
        assert X.matches(step_trigger, step_e) == (what in fires)
        assert not X.matches(task_trigger, step_e) and not X.matches(step_trigger, task_e)
    assert not X.matches(step_trigger, event('STEP_ENTERED', task='other', step='approach', step_kind='move'))
    assert not X.matches(step_trigger, event('STEP_ENTERED', task='pick_one', step='retreat', step_kind='move'))


def test_a_pick_attempt_context_fires_only_for_a_grasp_step():
    (trigger,) = triggers({'id': 'g', 'kind': 'APPLICATION_STATE', 'task': 'pick_one', 'step': 'pick',
                           'transition': 'enter', 'context': {'pick_attempt': {'phase': 'pre_grasp'}}})
    assert X.matches(trigger, event('STEP_ENTERED', task='pick_one', step='pick', step_kind='grasp'))
    assert not X.matches(trigger, event('STEP_ENTERED', task='pick_one', step='pick', step_kind='move'))


# ---- what a firing records ------------------------------------------------------------------------------------

def test_a_step_firing_records_the_event_and_the_task_step_context():
    (trigger,) = triggers({'id': 'a', 'kind': 'APPLICATION_STATE', 'task': 'pick_one', 'step': 'approach',
                           'transition': 'exit'})
    reason, context = X.firing(trigger, event('STEP_EXITED', seq=4, task='pick_one', step='approach',
                                              step_kind='move', status='HALTED'))
    assert reason == {'event': {'run_id': 'run-a', 'seq': 4, 'kind': 'STEP_EXITED', 'task_id': 'pick_one',
                                'step_id': 'approach', 'step_kind': 'move', 'status': 'HALTED'},
                      'transition': 'exit'}
    assert context == {'trainit.task_step': {'run_id': 'run-a', 'task_id': 'pick_one', 'step_id': 'approach'}}


def test_a_process_firing_records_channel_kind_value_and_edge():
    (trigger,) = triggers({'id': 'e', 'kind': 'EVENT', 'process_channel': 'speed', 'edge': 'any'})
    reason, context = X.firing(trigger, event('PROCESS_EVENT', task='cycle', channel='speed',
                                              process_kind='set_value', value=2.5))
    assert reason['event']['process_channel'] == 'speed' and reason['event']['process_value'] == 2.5
    assert reason['edge'] == 'any' and context == {'trainit.task_step': {'run_id': 'run-a', 'task_id': 'cycle'}}


@pytest.mark.parametrize('candidate, expected', [
    (('c7', 'synthetic'), {'id': 'c7', 'provider': 'synthetic'}), (('', ''), None)])
def test_a_pick_attempt_records_its_attempt_and_its_candidate_when_known(candidate, expected):
    (trigger,) = triggers({'id': 'g', 'kind': 'APPLICATION_STATE', 'task': 'pick_one', 'step': 'pick',
                           'transition': 'succeeded', 'context': {'pick_attempt': {'phase': 'post_grasp'}}})
    _, context = X.firing(trigger, event('STEP_EXITED', task='pick_one', step='pick', step_kind='grasp',
                                         status='SUCCEEDED', attempt=2, candidate=candidate))
    attempt = context['trainit.pick_attempt']
    assert attempt['phase'] == 'post_grasp' and attempt['attempt_index'] == 2
    assert attempt['attempt_id'] == 'run-a/pick_one/pick/2'
    assert attempt.get('candidate') == expected
    assert 'success' not in str(attempt) and 'outcome' not in attempt          # never a D1 / M4.5 verdict
