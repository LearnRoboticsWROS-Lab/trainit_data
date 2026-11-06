"""EVENT and APPLICATION_STATE triggers from the runner's ExecutionEvent (D0.5, ADR-0030; no ROS imports).

The recorder's ONE source of application facts is ``trainit_interfaces/ExecutionEvent`` on the relative topic
``trainit/execution_events`` of its robot slot (the runner, TMR, is the only producer). It never reads a process
topic: those are latched process STATE, carry no event time and no run identity (D0_DESIGN.md §7.2).

Trigger configuration (validated here, written in the recording configuration)::

    - {id: on_glue, kind: EVENT, process_channel: glue, edge: rising}         # rising | falling | any
    - {id: approach, kind: APPLICATION_STATE, task: pick_one, step: approach, transition: enter}
    - {id: inspected, kind: APPLICATION_STATE, task: inspect, transition: succeeded}
    - {id: before_grasp, kind: APPLICATION_STATE, task: pick_one, step: pick, transition: enter,
       context: {pick_attempt: {phase: pre_grasp}}}                          # pre_grasp | post_grasp

- EVENT: a PROCESS_EVENT on the LOGICAL channel (never a topic). ``rising`` = ``process_on``, ``falling`` =
  ``process_off``, ``any`` = every process event of the channel (``set_value`` and ``external`` included).
- APPLICATION_STATE: the canonical task (its name) and, optionally, the canonical step id (D18). Without a step
  it is the TASK's transition, with one the STEP's. ``enter`` = entered; ``exit`` = exited, however (succeeded,
  failed or halted); ``succeeded`` / ``failed`` = exited so.
- PICK_ATTEMPT is a CONTEXT, not a kind: an APPLICATION_STATE trigger on a GRASP step may ask for it, ``pre_grasp``
  with ``enter``, ``post_grasp`` with an exit; it fires only for an event whose step kind is ``grasp``, and the
  capture then carries ``trainit.pick_attempt`` {attempt_id, attempt_index, phase, candidate?}. Whether the
  attempt succeeded is never stated (D1 / M4.5).

**Which events may fire** (:class:`EventGate`, fail closed): only events of the ACTIVE session -- stamped at or
after its start on the session clock (an event of the past never fires, however late it arrives), at most
``clock_mismatch_ns`` ahead of the session clock, of the ONE run whose RUN_STARTED this session received (a run
already going is never joined mid-way; another run while one is open is ambiguous; a new run's start supersedes
one that never ended), in increasing ``seq``, well formed. Every refusal is counted by reason.

**Time:** the capture boundary is the EVENT's stamp (the runner's clock; the same session clock: ROS time, or
sim time under use_sim_time). The recorder's receipt is recorded beside it, never substituted.
"""
from trainit_data import _strict as s          # same repository, same version line (ADR-0029 Decision 1)
from trainit_data.errors import INVALID_VALUE

EVENT_KINDS = {1: 'RUN_STARTED', 2: 'RUN_ENDED', 3: 'TASK_ENTERED', 4: 'TASK_EXITED', 5: 'STEP_ENTERED',
               6: 'STEP_EXITED', 7: 'PROCESS_EVENT'}
STATUSES = {0: None, 1: 'SUCCEEDED', 2: 'FAILED', 3: 'HALTED'}
EDGES = {'rising': ('process_on',), 'falling': ('process_off',),
         'any': ('process_on', 'process_off', 'set_value', 'external')}
TRANSITIONS = ('enter', 'exit', 'succeeded', 'failed')
PHASES = {'pre_grasp': ('enter',), 'post_grasp': ('exit', 'succeeded', 'failed')}
REFUSALS = ('STALE_EVENT', 'CLOCK_MISMATCH', 'CLOCK_UNAVAILABLE', 'UNSTARTED_RUN', 'AMBIGUOUS_RUN', 'ENDED_RUN',
            'DUPLICATE_EVENT', 'MALFORMED_EVENT')
EXECUTION_TRIGGERS = ('EVENT', 'APPLICATION_STATE')


def trigger_fields(out, f, where):
    """Read and check an EVENT / APPLICATION_STATE trigger's own fields into ``out`` (``f``: its strict Fields)."""
    kind = out['kind']
    if kind == 'EVENT':
        out['process_channel'] = f.req('process_channel', s.ident)
        out['edge'] = f.req('edge', s.one_of(EDGES))
    elif kind == 'APPLICATION_STATE':
        out['task'] = f.req('task', s.ident)
        s.put(out, 'step', f.opt('step', s.ident))
        out['transition'] = f.req('transition', s.one_of(TRANSITIONS))
        context = f.opt('context', s.opaque, None)
        if context is not None:
            out['context'] = _context(context, out, f'{where}.context')


def _context(value, trigger, where):
    c = s.Fields(value, where)
    attempt = c.req('pick_attempt', s.opaque)
    c.done()
    a = s.Fields(attempt, f'{where}.pick_attempt')
    phase = a.req('phase', s.one_of(PHASES))
    a.done()
    if 'step' not in trigger:
        s.fail(INVALID_VALUE, 'a pick_attempt context names the grasp STEP (a task has no attempt)', where)
    if trigger['transition'] not in PHASES[phase]:
        s.fail(INVALID_VALUE, f'phase {phase} is the transition {" / ".join(PHASES[phase])} of the grasp step, '
               f'not {trigger["transition"]}', where)
    return {'pick_attempt': {'phase': phase}}


def event_from_message(msg) -> dict:
    """The JSON-plain facts of a ``trainit_interfaces/ExecutionEvent`` (no ROS import: attribute access)."""
    return {'kind': EVENT_KINDS.get(msg.kind), 'status': STATUSES.get(msg.status, 'UNKNOWN'),
            'run_id': msg.run_id, 'seq': int(msg.seq),
            'stamp_ns': msg.header.stamp.sec * 1_000_000_000 + msg.header.stamp.nanosec,
            'task_id': msg.task_id, 'step_id': msg.step_id, 'step_kind': msg.step_kind,
            'process_channel': msg.process_channel, 'process_kind': msg.process_kind,
            'process_value': float(msg.process_value), 'attempt_index': int(msg.attempt_index),
            'candidate_id': msg.candidate_id, 'candidate_provider': msg.candidate_provider}


class EventGate:
    """Which execution events belong to the ACTIVE session (one instance per session). Fail closed:

    - a run is ADOPTED only from its RUN_STARTED (seq 1) received in this session -- a run already going when the
      session began is never joined mid-way (its identity is incomplete: ``UNSTARTED_RUN``). Start the session
      before the application run;
    - one run at a time: another run's events while one is open are ``AMBIGUOUS_RUN``; a NEW run's RUN_STARTED
      (seq 1) supersedes an open run that never ended (a runner restarted after a crash: one runner per robot slot),
      counted in ``superseded``; the superseded run's later events are refused;
    - an event of a run that ENDED (or was superseded) is ``ENDED_RUN``; a repeated / older seq ``DUPLICATE_EVENT``;
    - an event stamped before the session start is ``STALE_EVENT``; more than ``clock_mismatch_ns`` ahead of the
      session clock ``CLOCK_MISMATCH``; while the session clock reads 0 (sim time not yet running) every event is
      ``CLOCK_UNAVAILABLE`` and the start is taken at its first non-zero reading; a session clock that jumps BACK (a
      simulation reset) re-bases the start (``rebase``): events of the old timeline are then stale or ahead.
    """

    def __init__(self, session_start_ns: int, clock_mismatch_ns: int):
        self.session_start_ns = session_start_ns if session_start_ns > 0 else None
        self.clock_mismatch_ns = clock_mismatch_ns
        self.open_run = None                # the adopted run whose events are admitted
        self.closed_runs = set()            # ended or superseded: never admitted again
        self.last_seq = {}                  # run_id -> last admitted seq
        self.refused = {r: 0 for r in REFUSALS}
        self.admitted = 0
        self.superseded = 0
        self.rebases = 0

    def rebase(self, now_ns: int):
        """The session clock jumped back (a simulation reset): the session's start is its new reading."""
        self.session_start_ns = now_ns if now_ns > 0 else None
        self.rebases += 1

    def admit(self, event: dict, now_ns: int):
        """``None`` when the event may fire triggers, else the refusal reason (counted)."""
        if self.session_start_ns is None and now_ns > 0:
            self.session_start_ns = now_ns          # sim time began after the session did
        reason = self._judge(event, now_ns)
        if reason is not None:
            self.refused[reason] += 1
            return reason
        run = event['run_id']
        if event['kind'] == 'RUN_STARTED' and self.open_run not in (None, run):
            self.closed_runs.add(self.open_run)
            self.superseded += 1
        if event['kind'] == 'RUN_STARTED':
            self.open_run = run
        self.admitted += 1
        self.last_seq[run] = event['seq']
        if event['kind'] == 'RUN_ENDED':
            self.closed_runs.add(run)
            self.open_run = None
        return None

    def _judge(self, event, now_ns):
        if event.get('kind') is None or not event.get('run_id') or event.get('seq', 0) < 1 \
                or event.get('stamp_ns', 0) <= 0:
            return 'MALFORMED_EVENT'
        if self.session_start_ns is None or now_ns <= 0:
            return 'CLOCK_UNAVAILABLE'
        if event['stamp_ns'] < self.session_start_ns:
            return 'STALE_EVENT'                    # it happened before this session began
        if event['stamp_ns'] - now_ns > self.clock_mismatch_ns:
            return 'CLOCK_MISMATCH'                 # the runner and the session clock disagree
        run = event['run_id']
        if run in self.closed_runs:
            return 'ENDED_RUN'
        if event['seq'] <= self.last_seq.get(run, 0):
            return 'DUPLICATE_EVENT'
        if event['kind'] == 'RUN_STARTED' and event['seq'] == 1:
            return None                             # a runner's start: adopted (superseding an open run)
        if self.open_run is None:
            return 'UNSTARTED_RUN'                  # a run this session did not see start: never joined mid-way
        if run != self.open_run:
            return 'AMBIGUOUS_RUN'                  # another run while one is open: fail closed
        return None


def matches(trigger: dict, event: dict) -> bool:
    """Whether an ADMITTED event fires ``trigger`` (an EVENT or APPLICATION_STATE trigger)."""
    if trigger['kind'] == 'EVENT':
        return event['kind'] == 'PROCESS_EVENT' and event['process_channel'] == trigger['process_channel'] \
            and event['process_kind'] in EDGES[trigger['edge']]
    if trigger['kind'] != 'APPLICATION_STATE' or event['task_id'] != trigger['task']:
        return False
    if 'step' in trigger:
        if event['step_id'] != trigger['step'] or event['kind'] not in ('STEP_ENTERED', 'STEP_EXITED'):
            return False
        if 'context' in trigger and event['step_kind'] != 'grasp':
            return False                            # a pick attempt is a grasp step's, never assumed
    elif event['kind'] not in ('TASK_ENTERED', 'TASK_EXITED'):
        return False
    entered = event['kind'] in ('STEP_ENTERED', 'TASK_ENTERED')
    return {'enter': entered, 'exit': not entered,
            'succeeded': not entered and event['status'] == 'SUCCEEDED',
            'failed': not entered and event['status'] == 'FAILED'}[trigger['transition']]


def firing(trigger: dict, event: dict):
    """(reason, context) of a capture fired by ``event``: the event's facts and the namespaced context ids."""
    facts = {k: event[k] for k in ('run_id', 'seq', 'kind') if event.get(k) is not None}
    for k in ('task_id', 'step_id', 'step_kind', 'process_channel', 'process_kind'):
        if event.get(k):
            facts[k] = event[k]
    if event.get('status'):
        facts['status'] = event['status']
    if event['kind'] == 'PROCESS_EVENT':
        facts['process_value'] = event['process_value']
    reason = {'event': facts}
    if trigger['kind'] == 'EVENT':
        reason['edge'] = trigger['edge']
    else:
        reason['transition'] = trigger['transition']
    context = {}
    if event.get('task_id'):
        step = {'run_id': event['run_id'], 'task_id': event['task_id']}
        if event.get('step_id'):
            step['step_id'] = event['step_id']
        context['trainit.task_step'] = step
    if 'context' in trigger:
        attempt = {'phase': trigger['context']['pick_attempt']['phase'],
                   'attempt_index': event['attempt_index'],
                   'attempt_id': f'{event["run_id"]}/{event["task_id"]}/{event["step_id"]}/{event["attempt_index"]}'}
        if event.get('candidate_id'):
            attempt['candidate'] = {'id': event['candidate_id'], 'provider': event['candidate_provider']}
        context['trainit.pick_attempt'] = attempt
    return reason, context


__all__ = ['EDGES', 'EVENT_KINDS', 'EXECUTION_TRIGGERS', 'EventGate', 'PHASES', 'REFUSALS', 'TRANSITIONS',
           'event_from_message', 'firing', 'matches', 'trigger_fields']
