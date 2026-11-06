"""``trainit-synthetic-provider/1.0``: what a synthetic data PROVIDER declares about itself (D0.8; pure).

A provider is an external generation runtime (the reference one: ``examples/isaac_replicator_bin``, Isaac
Replicator). It ships a ``provider.json`` descriptor beside its entry script and base configuration. A configuration
tool (TSA) renders its form FROM the descriptor, validates the user's values against it, writes the resolved
configuration (the base configuration with the exposed parameters set) and LAUNCHES the provider as an external
process; the provider validates the same configuration against the same descriptor (:func:`config_problems`) before
its runtime starts. Neither side imports the other: the descriptor file is the contract. Canonical data is written
by the provider through ``trainit_data`` (one ordinary session), never by the tool::

    {"schema": "trainit-synthetic-provider/1.0",
     "id": "isaac_replicator_bin", "name": "...", "version": "1.0.0", "description": "...",
     "runtime": {"kind": "isaac_sim_python", "min_version": "5.1"},
     "backends": ["isaac"],                     # the deployment backends whose cells it serves
     "entry": "generate.py", "base_config": "bin_config.json",
     "general": {"seed": {...}, "runs": {...}, "views": {...}},   # each an int parameter without a key
     "parameters": [{"key": "lights.dome_intensity", "type": "float_range", "min": 0, "max": 5000,
                     "unit": "nits", "label": "Ambient light", "help": "...", "group": "lighting",
                     "level": "simple"}, ...],
     "outputs": {"modalities": [...], "annotations": [...], "classes": [...]},
     "progress": "trainit-provider-progress/1"}

``key`` is a REAL dotted path of the base configuration (a parameter the provider does not read is never declared:
unsupported controls are absent, not decorative). The DEFAULT of a parameter is its value in the base configuration
(not repeated here). Types: ``int``, ``float`` (``min`` <= v <= ``max``); ``int_range`` / ``float_range`` (``[lo, hi]``,
``min`` <= lo <= hi <= ``max``); ``float_pair`` (``[a, b]``, each within bounds); ``choice`` (one of ``values``).

The invocation and the progress protocol ``trainit-provider-progress/1`` are fixed for every provider::

    <runtime python> <entry> --config <resolved.json> --seed N --runs N --views N [--store DIR]
    stdout: TRAINIT_PROGRESS {"phase": ...}   (starting | session | run | sealing; JSON object, one line)
            TRAINIT_SESSION <session_id>      (success: the last protocol line)
            TRAINIT_FAILED                    (failure; a session already opened is left ACTIVE for recover)
"""
import json
from pathlib import Path

from . import _strict as s
from .errors import INVALID_VALUE, DataError

SCHEMA_NAME = 'trainit-synthetic-provider'
SCHEMA_MAJOR = 1
DESCRIPTOR_FILE = 'provider.json'
PROGRESS_PROTOCOL = 'trainit-provider-progress/1'
TYPES = ('int', 'float', 'int_range', 'float_range', 'float_pair', 'choice')
GROUPS = ('objects', 'lighting', 'appearance', 'camera', 'scene', 'quality')
LEVELS = ('simple', 'advanced')
GENERAL = ('seed', 'runs', 'views')
RUNTIMES = ('isaac_sim_python',)


def _number(kind):
    return s.integer() if kind.startswith('int') else s.real


def _parameter(value, where, *, general=False):
    f = s.Fields(value, where)
    out = {} if general else {'key': f.req('key', s.token)}
    out['type'] = 'int' if general else f.req('type', s.one_of(TYPES))
    if general:
        f.opt('type', s.one_of(('int',)))
    if out['type'] == 'choice':
        out['values'] = list(f.req('values', s.list_of(s.json_value, min_length=1)))
    else:
        out['min'] = f.req('min', _number(out['type']))
        out['max'] = f.req('max', _number(out['type']))
        if out['min'] > out['max']:
            s.fail(INVALID_VALUE, 'min > max', where)
    if general:
        out['default'] = f.req('default', s.integer())
    out['label'] = f.req('label', s.string)
    s.put(out, 'unit', f.opt('unit', s.string))
    s.put(out, 'help', f.opt('help', s.string))
    if not general:
        out['group'] = f.req('group', s.one_of(GROUPS))
        out['level'] = f.req('level', s.one_of(LEVELS))
    f.done()
    if not general and any(not part for part in out['key'].split('.')):
        s.fail(INVALID_VALUE, f'{out["key"]!r} is not a dotted key path', f'{where}.key')
    if general:
        problem = value_problem(out, out['default'])
        if problem:
            s.fail(INVALID_VALUE, f'default {problem}', where)
    return out


def _runtime(value, where):
    f = s.Fields(value, where)
    out = {'kind': f.req('kind', s.one_of(RUNTIMES))}
    s.put(out, 'min_version', f.opt('min_version', s.token))
    f.done()
    return out


def _outputs(value, where):
    f = s.Fields(value, where)
    out = {name: list(f.req(name, s.list_of(s.token, unique=True)))
           for name in ('modalities', 'annotations', 'classes')}
    f.done()
    return out


def parse_descriptor(document, where='provider.json') -> dict:
    """The strict, normalised descriptor (unknown fields refused)."""
    f = s.Fields(document, where)
    out = {'schema': f.req('schema', s.schema(SCHEMA_NAME, SCHEMA_MAJOR)), 'id': f.req('id', s.ident),
           'name': f.req('name', s.string), 'version': f.req('version', s.token),
           'description': f.req('description', s.string), 'runtime': f.req('runtime', _runtime),
           'backends': list(f.req('backends', s.list_of(s.ident, min_length=1, unique=True))),
           'entry': f.req('entry', s.relative_path), 'base_config': f.req('base_config', s.relative_path)}
    general = s.Fields(f.req('general', s.opaque), f'{where}.general')
    out['general'] = {name: general.req(name, lambda v, w: _parameter(v, w, general=True)) for name in GENERAL}
    general.done()
    out['parameters'] = list(f.req('parameters', s.list_of(_parameter)))
    keys = [p['key'] for p in out['parameters']]
    if len(set(keys)) != len(keys):
        s.fail(INVALID_VALUE, 'a parameter key is declared twice', f'{where}.parameters')
    out['outputs'] = f.req('outputs', _outputs)
    progress = f.req('progress', s.string)
    if progress != PROGRESS_PROTOCOL:
        s.fail(INVALID_VALUE, f'{progress!r}: this reader speaks {PROGRESS_PROTOCOL}', f'{where}.progress')
    out['progress'] = progress
    f.done()
    return out


def load_descriptor(directory) -> dict:
    """The descriptor of the provider in ``directory`` (``provider.json``), parsed strictly."""
    path = Path(directory) / DESCRIPTOR_FILE
    try:
        document = json.loads(path.read_text())
    except (OSError, ValueError) as exc:
        raise DataError(INVALID_VALUE, f'{path}: not a readable JSON descriptor', context={'path': str(path)},
                        cause=exc) from None
    return parse_descriptor(document, str(path))


def get_path(config: dict, key: str):
    """The value at a dotted key path (``KeyError`` when absent)."""
    value = config
    for part in key.split('.'):
        if not isinstance(value, dict):
            raise KeyError(key)
        value = value[part]
    return value


def _is_number(value, kind):
    if isinstance(value, bool):
        return False
    return isinstance(value, int) if kind == 'int' else isinstance(value, (int, float))


def value_problem(parameter: dict, value):
    """None if ``value`` meets the parameter's type and bounds, else a short reason."""
    kind = parameter['type']
    if kind == 'choice':
        return None if value in parameter['values'] else f'{value!r} is not one of {parameter["values"]}'
    lo, hi = parameter['min'], parameter['max']
    scalar = 'int' if kind.startswith('int') else 'float'
    if kind in ('int', 'float'):
        if not _is_number(value, scalar):
            return f'{value!r} is not {"an integer" if scalar == "int" else "a number"}'
        return None if lo <= value <= hi else f'{value} is outside [{lo}, {hi}]'
    if not isinstance(value, list) or len(value) != 2 or not all(_is_number(v, scalar) for v in value):
        return f'{value!r} is not a pair of {"integers" if scalar == "int" else "numbers"}'
    if not all(lo <= v <= hi for v in value):
        return f'{value} is outside [{lo}, {hi}]'
    if kind != 'float_pair' and value[0] > value[1]:
        return f'{value}: the lower bound exceeds the upper one'
    return None


def config_problems(descriptor: dict, config: dict) -> list:
    """Every exposed parameter of a resolved configuration that is absent or out of bounds (empty = valid)."""
    problems = []
    for parameter in descriptor['parameters']:
        try:
            value = get_path(config, parameter['key'])
        except KeyError:
            problems.append(f'{parameter["key"]}: absent from the configuration')
            continue
        problem = value_problem(parameter, value)
        if problem:
            problems.append(f'{parameter["key"]}: {problem}')
    return problems


def general_problems(descriptor: dict, **values) -> list:
    """``seed`` / ``runs`` / ``views`` against their declared bounds (empty = valid)."""
    problems = []
    for name in GENERAL:
        problem = value_problem(descriptor['general'][name], values[name])
        if problem:
            problems.append(f'{name}: {problem}')
    return problems


__all__ = ['DESCRIPTOR_FILE', 'PROGRESS_PROTOCOL', 'config_problems', 'general_problems', 'get_path',
           'load_descriptor', 'parse_descriptor', 'value_problem']
