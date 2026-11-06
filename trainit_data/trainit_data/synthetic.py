"""Synthetic generation provenance: randomization runs, the generation block, the scene manifest (PUBLIC, D0.7).

A synthetic session is an ordinary canonical session (transport ``INGEST``) whose captures were generated, not
acquired. Nothing here is simulator-specific: an Isaac Replicator adapter fills these records, a Gazebo one would
fill the same.

**A capture** of a generator is fired by a ``SYNTHETIC_FRAME`` trigger (origin ``GENERATOR``):

- ``trigger.context["trainit.randomization"] = {run, run_index, view}``: ids only (the join seam);
- ``trigger.reason = {recipe, run_seed, draws, view_draws}``: the frame's RECORDED randomization decisions -- the
  run's draws (shared by every view of the run) and this view's own. ``recipe`` names what the draws mean.

**A randomization run** is one randomized scene and every view rendered of it: the unit of correlation, so it is
a split group (``group_by: randomization_run``). Its id is CONTENT-ADDRESSED: ``rr-`` + 16 hex of the sha256 of
``{recipe, draws}``. The same scene drawn twice (the same config and seed, or a regenerated duplicate) is the same
run wherever it was generated, so it can never sit in two splits.

**The generation block** ``provenance.randomization`` (``trainit-randomization/1.0``) states the recipe: the
generator, the recipe name, the master seed, the hash of the generation config (= the session's
``recording_config.resolved_sha256``), the run / view counts and the reproducibility claim with the environment it
holds in. The claim is ``SAME_DRAWS``: the same supported environment + the same config + the same seed give the
same recorded runs, ordering, draws and semantic assignments; rendered pixels are NOT promised bit-identical.

**The scene manifest** ``provenance.scene`` (``trainit-synthetic-scene/1.0``) identifies the scene the data was
generated from, without copying it: the base stage's hash BEFORE randomization, the layers and assets it depends
on (by logical / relative id + sha256; an absolute machine path is refused), how complete that is
(``RESOLVED_LAYERS`` only when every dependency was hashed; ``PARTIAL`` names what was not), the semantic class
configuration and the stated limitations. It never claims more than it hashed.

:func:`compare_generations` is acceptance D0-G: it compares two synthetic sessions on exactly the claim.
"""
import re

from . import _strict as s
from ._canonical import sha256_json
from .errors import INVALID_VALUE, DataError

NAMESPACE = 'trainit.randomization'
TRIGGER_KIND = 'SYNTHETIC_FRAME'
GENERATION_SCHEMA_NAME = 'trainit-randomization'
GENERATION_SCHEMA = 'trainit-randomization/1.0'
SCENE_SCHEMA_NAME = 'trainit-synthetic-scene'
SCENE_SCHEMA = 'trainit-synthetic-scene/1.0'
CLAIMS = frozenset({'SAME_DRAWS'})
SCENE_KINDS = frozenset({'PROCEDURAL', 'STAGE_FILE'})
COMPLETENESS = frozenset({'RESOLVED_LAYERS', 'ROOT_LAYER_ONLY', 'PARTIAL'})

run_id_check = s.matching(re.compile(r'^rr-[0-9a-f]{16}$'), 'randomization run id (rr-<16 hex>)')
_ABSOLUTE_RE = re.compile(r'^(?:/|~|[A-Za-z]:[\\/]|\\\\|file:)')


def run_id(recipe: str, draws: dict) -> str:
    """The content-addressed id of a randomization run."""
    return 'rr-' + sha256_json({'recipe': s.token(recipe, 'recipe'), 'draws': s.opaque(draws, 'draws')})[7:23]


def run_seed(master_seed: int, run_index: int) -> int:
    """The seed of run ``run_index``: a 63-bit integer derived from the master seed alone (independent of how many
    runs are generated, so run k is the same scene in a 10-run and a 1000-run generation)."""
    s.integer(0)(master_seed, 'master_seed')
    s.integer(0)(run_index, 'run_index')
    return int(sha256_json({'master_seed': master_seed, 'run_index': run_index})[7:23], 16) >> 1


# ---- capture-level records ---------------------------------------------------------------------------------

def _context(value, where):
    f = s.Fields(value, where)
    out = {'run': f.req('run', run_id_check), 'run_index': f.req('run_index', s.integer(0)),
           'view': f.req('view', s.integer(0))}
    f.done()
    return out


def _reason(value, where):
    f = s.Fields(value, where)
    out = {'recipe': f.req('recipe', s.token), 'run_seed': f.req('run_seed', s.integer(0)),
           'draws': f.req('draws', s.opaque), 'view_draws': f.req('view_draws', s.opaque)}
    f.done()
    return out


def frame_of(capture, where: str = 'capture'):
    """``(context, reason)`` of a generated capture, validated; ``None`` for a capture no generator fired."""
    if capture.trigger.kind != TRIGGER_KIND:
        if NAMESPACE in capture.trigger.context:
            s.fail(INVALID_VALUE, f'only a {TRIGGER_KIND} capture carries {NAMESPACE}', f'{where}.trigger.context')
        return None
    if NAMESPACE not in capture.trigger.context:
        s.fail(INVALID_VALUE, f'a {TRIGGER_KIND} capture carries its {NAMESPACE} context', f'{where}.trigger.context')
    context = _context(capture.trigger.context[NAMESPACE], f'{where}.trigger.context.{NAMESPACE}')
    reason = _reason(capture.trigger.reason, f'{where}.trigger.reason')
    if run_id(reason['recipe'], reason['draws']) != context['run']:
        s.fail(INVALID_VALUE, f'run {context["run"]} is not the id of the recorded draws '
               f'({run_id(reason["recipe"], reason["draws"])})', f'{where}.trigger')
    return context, reason


def group_of(capture, group_by: str) -> str:
    """The split group of a capture: its session, or (``randomization_run``) its run -- a capture no generator
    fired (a live camera's) falls back to its session, so real-session grouping never changes."""
    if group_by == 'session':
        return capture.session_id
    frame = frame_of(capture)
    return capture.session_id if frame is None else frame[0]['run']


# ---- session-level records ---------------------------------------------------------------------------------

def _environment(value, where):
    return s.str_map(value, where)


def _reproducibility(value, where):
    f = s.Fields(value, where)
    out = {'claim': f.req('claim', s.one_of(CLAIMS)),
           'render_bit_identical': f.req('render_bit_identical', s.boolean),
           'environment': f.req('environment', _environment)}
    f.done()
    if out['render_bit_identical']:
        s.fail(INVALID_VALUE, 'rendered pixels are never promised bit-identical (the claim is SAME_DRAWS)', where)
    return out


def generation_block(value, where='provenance.randomization') -> dict:
    """``provenance.randomization`` of a synthetic session, validated."""
    f = s.Fields(value, where)
    out = {'schema': f.req('schema', s.schema(GENERATION_SCHEMA_NAME, 1)),
           'generator': f.req('generator', s.token), 'recipe': f.req('recipe', s.token),
           'master_seed': f.req('master_seed', s.integer(0)), 'config_sha256': f.req('config_sha256', s.sha256),
           'runs': f.req('runs', s.integer(1)), 'views_per_run': f.req('views_per_run', s.integer(1)),
           'reproducibility': f.req('reproducibility', _reproducibility)}
    f.done()
    return out


def logical_id(value, where):
    """A layer / asset identity: a logical or relative reference, never an absolute machine path."""
    value = s.string(value, where)
    if _ABSOLUTE_RE.match(value) or '\\' in value or '..' in value.split('/'):
        s.fail(INVALID_VALUE, f'{value!r} is an absolute or escaping path: identify layers and assets by a logical or '
               'relative id', where)
    return value


def _hashed(value, where):
    f = s.Fields(value, where)
    out = {'id': f.req('id', logical_id), 'sha256': f.req('sha256', s.sha256)}
    f.done()
    return out


def _unique_ids(items, where):
    ids = [x['id'] for x in items]
    if ids != sorted(ids) or len(set(ids)) != len(ids):
        s.fail(INVALID_VALUE, 'ids are unique and sorted', where)


def _base_stage(value, where):
    f = s.Fields(value, where)
    out = {'sha256': f.req('sha256', s.sha256), 'completeness': f.req('completeness', s.one_of(COMPLETENESS)),
           'layers': [dict(x) for x in f.req('layers', s.list_of(_hashed))],
           'unresolved': list(f.req('unresolved', s.list_of(logical_id, unique=True)))}
    f.done()
    _unique_ids(out['layers'], f'{where}.layers')
    if (out['completeness'] == 'PARTIAL') != bool(out['unresolved']):
        s.fail(INVALID_VALUE, 'PARTIAL exactly when a dependency is unresolved (and named)', where)
    if out['completeness'] == 'ROOT_LAYER_ONLY' and out['layers']:
        s.fail(INVALID_VALUE, 'ROOT_LAYER_ONLY hashes no dependency layer', where)
    return out


def _semantics(value, where):
    f = s.Fields(value, where)
    classes = list(f.req('classes', s.list_of(s.ident, min_length=1, unique=True)))
    out = {'classes': classes, 'sha256': f.req('sha256', s.sha256), 'rule': f.req('rule', s.string)}
    f.done()
    if out['sha256'] != sha256_json(classes):
        s.fail(INVALID_VALUE, 'semantics.sha256 is not the sha256 of its class list', where)
    return out


def scene_manifest(value, where='provenance.scene') -> dict:
    """``provenance.scene`` of a synthetic session (the SYNTHETIC_SCENE lineage), validated."""
    f = s.Fields(value, where)
    gf = s.Fields(f.req('generator', s.opaque), f'{where}.generator')
    generator = {'name': gf.req('name', s.token), 'sha256': gf.req('sha256', s.sha256)}
    gf.done()
    out = {'schema': f.req('schema', s.schema(SCENE_SCHEMA_NAME, 1)), 'kind': f.req('kind', s.one_of(SCENE_KINDS)),
           'generator': generator, 'base_stage': f.req('base_stage', _base_stage),
           'assets': [dict(x) for x in f.req('assets', s.list_of(_hashed))],
           'semantics': f.req('semantics', _semantics),
           'limitations': list(f.req('limitations', s.list_of(s.string)))}
    f.done()
    _unique_ids(out['assets'], f'{where}.assets')
    return out


def semantics_of(classes, rule: str) -> dict:
    return {'classes': list(classes), 'sha256': sha256_json(list(classes)), 'rule': rule}


def session_problems(header, captures) -> list:
    """What a synthetic session breaks of the generation rules (``verify``): its generation block and scene
    manifest parse, the config hash is the recording config's, every capture is a generated frame of the recipe,
    one run has one run index and one set of draws, a view appears once per run, and the counts hold."""
    block = header.provenance.get('randomization')
    if header.source.transport != 'INGEST' or block is None:
        frames = [c.capture_id for c in captures if c.trigger.kind == TRIGGER_KIND]
        return [f'{frames[:3]} are generated frames in a session with no generation block'] if frames else []
    problems = []
    try:
        generation = generation_block(block)
        scene_manifest(header.provenance.get('scene'))
    except Exception as exc:                                         # noqa: BLE001 -- a finding, never a crash
        return [f'the generation provenance is invalid: {getattr(exc, "message", exc)}']
    config = header.recording_config or {}
    if config.get('resolved_sha256') != generation['config_sha256']:
        problems.append('provenance.randomization.config_sha256 is not the recording config\'s resolved_sha256')
    runs, indices = {}, {}
    for capture in captures:
        try:
            frame = frame_of(capture, capture.capture_id)
        except Exception as exc:                                     # noqa: BLE001
            problems.append(f'{capture.capture_id}: {getattr(exc, "message", exc)}')
            continue
        if frame is None:
            problems.append(f'{capture.capture_id} is not a generated frame')
            continue
        context, reason = frame
        if reason['recipe'] != generation['recipe']:
            problems.append(f'{capture.capture_id}: recipe {reason["recipe"]} is not the session\'s')
        if context['run_index'] >= generation['runs'] or context['view'] >= generation['views_per_run']:
            problems.append(f'{capture.capture_id}: run {context["run_index"]} / view {context["view"]} is outside the '
                            f'declared {generation["runs"]} runs x {generation["views_per_run"]} views')
        if reason['run_seed'] != run_seed(generation['master_seed'], context['run_index']):
            problems.append(f'{capture.capture_id}: run_seed is not the seed of run {context["run_index"]} of master '
                            f'seed {generation["master_seed"]}')
        if indices.setdefault(context['run_index'], context['run']) != context['run']:
            problems.append(f'{capture.capture_id}: run index {context["run_index"]} holds two runs (two sets of '
                            'draws)')
        run = runs.setdefault(context['run'], {'index': context['run_index'], 'seed': reason['run_seed'], 'views': []})
        if (run['index'], run['seed']) != (context['run_index'], reason['run_seed']):
            problems.append(f'{capture.capture_id}: run {context["run"]} has two run indices / seeds (two runs of one '
                            'generation drew the same scene)')
        if context['view'] in run['views']:
            problems.append(f'{capture.capture_id}: view {context["view"]} of run {context["run"]} appears twice')
        run['views'].append(context['view'])
    return problems


# ---- D0-G ---------------------------------------------------------------------------------------------------

def _generation_view(session) -> dict:
    """What the reproducibility claim covers, of one sealed synthetic session."""
    header = session.header
    if 'randomization' not in header.provenance or 'scene' not in header.provenance:
        raise DataError(INVALID_VALUE, f'session {session.session_id} is not a synthetic generation (no generation '
                        'block / scene manifest)', context={'session_id': session.session_id})
    if not session.captures:
        raise DataError(INVALID_VALUE, f'session {session.session_id} holds no frame: there is nothing to compare',
                        context={'session_id': session.session_id})
    frames, semantics, calibrations, payloads = [], [], {}, {}
    for capture in session.captures:
        frame = frame_of(capture, capture.capture_id)
        if frame is None:
            raise DataError(INVALID_VALUE, f'{capture.capture_id} is not a generated frame',
                            context={'capture_id': capture.capture_id})
        context, reason = frame
        frames.append({'run': context['run'], 'run_index': context['run_index'], 'view': context['view'],
                       'run_seed': reason['run_seed'], 'draws': reason['draws'], 'view_draws': reason['view_draws'],
                       'outcome': capture.outcome})
        for obs in capture.observations:
            key = (context['run_index'], context['view'], obs.stream_key)
            if obs.data.block == 'instance_image':
                semantics.append({'run_index': context['run_index'], 'view': context['view'],
                                  'instances': [{'id': i['id'], 'object': i['object'], 'class': i['class']}
                                                for i in obs.data.instances]})
            calibrations[key] = obs.data.calibration_id
            payloads[key] = obs.payload.sha256 if obs.payload is not None else None
    generation = generation_block(header.provenance['randomization'])
    scene = scene_manifest(header.provenance['scene'])
    recipe = {k: v for k, v in generation.items() if k != 'reproducibility'}
    return {'frames': frames, 'semantics': semantics, 'calibrations': calibrations, 'payloads': payloads,
            'generation': recipe, 'scene': scene, 'config_sha256': generation['config_sha256'],
            'environment': generation['reproducibility']['environment']}


def compare_generations(store, session_a: str, session_b: str) -> dict:
    """D0-G: do two sealed synthetic sessions hold the same RECORDED generation (the ``SAME_DRAWS`` claim)?

    Compared and required equal: the generation block but its environment (generator, recipe, master seed, config
    hash, run / view counts), the whole scene manifest (generator hash, base stage, layers, assets, semantics), the
    run structure and ordering (runs, run indices, views, run seeds, outcomes), every draw, the semantic assignment of
    every frame (instance id -> object -> class) and every observation's calibration id. Reported, never required:
    whether the rendered payload bytes are identical (``render``) and whether the environments match. Two DIFFERENT
    sessions are compared, each with at least one frame (else ``INVALID_VALUE``)."""
    from ._sealed import SealedSessions
    if session_a == session_b:
        raise DataError(INVALID_VALUE, 'D0-G compares two different generations, not a session with itself',
                        context={'session_id': session_a})
    sessions = SealedSessions(store)
    a, b = (_generation_view(sessions.get(x)) for x in (session_a, session_b))
    structure = [[(f['run'], f['run_index'], f['view'], f['run_seed'], f['outcome']) for f in x['frames']]
                 for x in (a, b)]
    draws = [[(f['draws'], f['view_draws']) for f in x['frames']] for x in (a, b)]
    checks = {
        'config_sha256': a['config_sha256'] == b['config_sha256'],
        'generation': a['generation'] == b['generation'],
        'scene': a['scene'] == b['scene'],
        'run_structure': structure[0] == structure[1],
        'draws': draws[0] == draws[1],
        'semantic_assignment': a['semantics'] == b['semantics'],
        'calibration_ids': a['calibrations'] == b['calibrations'],
    }
    keys = sorted(set(a['payloads']) & set(b['payloads']))
    same = [k for k in keys if a['payloads'][k] == b['payloads'][k]]
    by_stream = {}
    for key in keys:
        stream = key[2]
        entry = by_stream.setdefault(stream, {'identical': 0, 'different': 0})
        entry['identical' if key in same else 'different'] += 1
    return {'reproducible': all(checks.values()), 'checks': checks,
            'frames': {'a': len(a['frames']), 'b': len(b['frames'])},
            'runs': len({f['run'] for f in a['frames']}),
            'render': {'bit_identical': len(same) == len(keys) and bool(keys), 'by_stream': by_stream,
                       'promised': False},
            'environment': {'a': a['environment'], 'b': b['environment'],
                            'same': a['environment'] == b['environment']}}


__all__ = ['run_id', 'run_seed', 'frame_of', 'group_of', 'generation_block', 'scene_manifest', 'semantics_of',
           'session_problems', 'compare_generations', 'logical_id', 'NAMESPACE', 'TRIGGER_KIND', 'GENERATION_SCHEMA',
           'SCENE_SCHEMA']
