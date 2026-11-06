"""Split assignment: group-aware, deterministic, persisted (PUBLIC, D0.6).

``trainit-split/1.0`` is stored in each dataset version (``versions/<n>/splits.json``). It assigns GROUPS, never
single samples, to ``train`` / ``val`` / ``test``: everything of one group lands in one split, which is what
keeps correlated samples (one session: one scene, one lighting, one pile) from leaking across splits.

Strategy ``GROUP_HASH`` (decision D10): ``bucket = int(sha256("<seed>:<group>")[:16 hex], 16) mod 10000``,
compared with the cumulative ratios (``train`` below ``round(train * 10000)``, then ``val``, then ``test``).
It depends only on the seed and the group key, so adding a group never moves another one. ``overrides`` pin a
group to a split explicitly (recorded, never silent).

A derived version (ADD_SESSIONS, FILTER, RELABEL) INHERITS its parent's split: the same seed, ratios and
overrides, every kept group in exactly the split it had (copied from the parent, not recomputed), and only NEW
groups assigned. The memory is LINEAGE-WIDE: a group a FILTER removed is kept in ``retired`` with its split, so a
later version that adds it back puts it where it was (a session evaluated in ``test`` never comes back as
``train``). Moving a group between splits is a RESPLIT, a new split with no inheritance and nothing retired.

``group_by`` (decision D10) names what a group is: ``session`` (D0.6, the default) or ``randomization_run`` (D0.7:
a generated capture's randomization run -- one randomized scene and every view of it -- and, for a capture no
generator fired, its session, so a live camera's grouping never changes). The key function is
``synthetic.group_of``; nothing else here knows what a group is. A grouping is part of the split: a derived version
keeps its parent's, changing it is a RESPLIT. A split grouped by anything but ``session`` is written as
``trainit-split/1.1`` (a D0.6 split stays 1.0, byte for byte).
"""
import re

from . import _strict as s
from ._canonical import sha256_bytes, sha256_json
from .errors import DERIVATION_INVALID, INVALID_VALUE, DataError

SCHEMA_NAME = 'trainit-split'
SCHEMA_MAJOR = 1
SCHEMA = 'trainit-split/1.0'
SCHEMA_1_1 = 'trainit-split/1.1'             # group_by other than session (D0.7)

SPLITS = ('train', 'val', 'test')
STRATEGIES = frozenset({'GROUP_HASH'})
GROUP_BY = frozenset({'session', 'randomization_run'})
DEFAULT_GROUP_BY = 'session'
BUCKETS = 10000
RATIO_SUM_TOLERANCE = 1e-9
DEFAULT_RATIOS = {'train': 0.8, 'val': 0.1, 'test': 0.1}

split_id = s.matching(re.compile(r'^sp-[0-9a-f]{16}$'), 'split id (sp-<16 hex>)')


def _ratios(value, where):
    f = s.Fields(value, where)
    out = {name: f.req(name, s.real) for name in SPLITS}
    f.done()
    if any(r < 0.0 or r > 1.0 for r in out.values()):
        s.fail(INVALID_VALUE, 'every ratio lies in [0, 1]', where)
    if abs(sum(out.values()) - 1.0) > RATIO_SUM_TOLERANCE:
        s.fail(INVALID_VALUE, f'the ratios sum to {sum(out.values())}, not 1', where)
    return out


def _assignment(value, where):
    value = s.opaque(value, where)
    for group, split in value.items():
        s.string(group, f'{where}.<key>')
        s.one_of(set(SPLITS))(split, f'{where}.{group}')
    return value


def _inherited(value, where):
    f = s.Fields(value, where)
    out = {'version': f.req('version', s.integer(1)), 'split_id': f.req('split_id', split_id)}
    f.done()
    return out


def _counts(value, where):
    f = s.Fields(value, where)
    out = {}
    for name in SPLITS:
        cf = s.Fields(f.req(name, s.opaque), f'{where}.{name}')
        out[name] = {'groups': cf.req('groups', s.integer(0)), 'captures': cf.req('captures', s.integer(0))}
        cf.done()
    f.done()
    return out


def bucket_of(seed: int, group: str) -> int:
    digest = sha256_bytes(f'{seed}:{group}'.encode('utf-8'))[len('sha256:'):]
    return int(digest[:16], 16) % BUCKETS


def hashed_split(seed: int, ratios: dict, group: str) -> str:
    bucket = bucket_of(seed, group)
    if bucket < round(ratios['train'] * BUCKETS):
        return 'train'
    if bucket < round((ratios['train'] + ratios['val']) * BUCKETS):
        return 'val'
    return 'test'


class SplitAssignment:
    """One persisted assignment. Build with :func:`build_split`; parse with :meth:`from_dict`."""

    def __init__(self, record: dict):
        self.record = record

    @classmethod
    def from_dict(cls, value, where='splits') -> 'SplitAssignment':
        f = s.Fields(value, where)
        record = {'schema': f.req('schema', s.schema(SCHEMA_NAME, SCHEMA_MAJOR)),
                  'split_id': f.req('split_id', split_id),
                  'strategy': f.req('strategy', s.one_of(STRATEGIES)),
                  'group_by': f.req('group_by', s.one_of(GROUP_BY)),
                  'seed': f.req('seed', s.integer(0)),
                  'ratios': f.req('ratios', _ratios),
                  'overrides': f.req('overrides', _assignment),
                  'assignment': f.req('assignment', _assignment),
                  'counts': f.req('counts', _counts)}
        s.put(record, 'inherited', f.opt('inherited', _inherited))
        retired = f.opt('retired', _assignment)
        if retired:
            record['retired'] = retired
        f.done()
        both = sorted(set(record.get('retired', {})) & set(record['assignment']))
        if both:
            s.fail(INVALID_VALUE, f'groups {both} are both assigned and retired', where)
        if record.get('retired') and 'inherited' not in record:
            s.fail(INVALID_VALUE, 'only an inherited split retires groups', where)
        if record['schema'] == SCHEMA and record['group_by'] != 'session':
            s.fail(INVALID_VALUE, f'group_by {record["group_by"]} needs {SCHEMA_1_1}', where)
        if record['split_id'] != _split_id(record):
            s.fail(INVALID_VALUE, f'{record["split_id"]} is not the id of this assignment', where)
        stray = sorted(set(record['overrides']) - set(record['assignment']))
        if stray:
            s.fail(INVALID_VALUE, f'overrides name groups that are not assigned: {stray}', where)
        for group, split in record['overrides'].items():
            if record['assignment'][group] != split:
                s.fail(INVALID_VALUE, f'group {group} is overridden to {split} but assigned elsewhere', where)
        return cls(record)

    def to_dict(self) -> dict:
        return dict(self.record)

    @property
    def split_id(self) -> str:
        return self.record['split_id']

    @property
    def group_by(self) -> str:
        return self.record['group_by']

    @property
    def assignment(self) -> dict:
        return self.record['assignment']

    @property
    def history(self) -> dict:
        """Every group this split's lineage ever assigned: the current ones and the retired ones."""
        return {**self.record.get('retired', {}), **self.record['assignment']}

    def split_of(self, group: str) -> str:
        return self.record['assignment'][group]


def _split_id(record: dict) -> str:
    identity = {k: record[k] for k in ('schema', 'strategy', 'group_by', 'seed', 'ratios', 'overrides', 'assignment')}
    if record.get('retired'):
        identity['retired'] = record['retired']
    return 'sp-' + sha256_json(identity)[len('sha256:'):][:16]


def build_split(groups: dict, *, seed: int, ratios: dict = None, overrides: dict = None, parent=None,
                parent_version: int = None, group_by: str = DEFAULT_GROUP_BY) -> SplitAssignment:
    """Assign ``groups`` (``{group: number of captures}``, keyed as ``group_by`` says).

    With ``parent`` (a :class:`SplitAssignment`) the split is INHERITED: ``seed`` / ``ratios`` / ``group_by`` must
    be the parent's, the parent's overrides are kept, a group of the lineage (current or retired) keeps its split, a
    group left out is retired with its split, and ``overrides`` may name NEW groups only (moving an inherited group
    is a RESPLIT: ``DERIVATION_INVALID``)."""
    group_by = s.one_of(GROUP_BY)(group_by, 'split.group_by')
    ratios = _ratios(dict(DEFAULT_RATIOS if ratios is None else ratios), 'split.ratios')
    seed = s.integer(0)(seed, 'split.seed')
    overrides = _assignment(dict(overrides or {}), 'split.overrides')
    unknown = sorted(set(overrides) - set(groups))
    if unknown:
        raise DataError(INVALID_VALUE, f'split overrides name groups that are not in the version: {unknown}',
                        context={'groups': unknown})
    assignment = {}
    if parent is not None:
        p = parent.record
        if seed != p['seed'] or ratios != p['ratios'] or group_by != p['group_by']:
            raise DataError(DERIVATION_INVALID, 'a derived version inherits its parent\'s split (seed, ratios and '
                            'grouping); changing them is a RESPLIT', context={'parent_split_id': p['split_id']})
        history = parent.history
        moved = sorted(g for g in overrides if g in history and history[g] != overrides[g])
        if moved:
            raise DataError(DERIVATION_INVALID, f'groups {moved} would move between splits: a group of the lineage '
                            'keeps its split (a RESPLIT moves groups)', context={'groups': moved})
        kept_overrides = {g: v for g, v in p['overrides'].items() if g in groups}
        overrides = {**kept_overrides, **{g: v for g, v in overrides.items() if g not in history}}
        assignment.update({g: v for g, v in history.items() if g in groups})
        retired = {g: v for g, v in history.items() if g not in groups}
    for group in sorted(groups):
        if group not in assignment:
            assignment[group] = overrides.get(group) or hashed_split(seed, ratios, group)
    counts = {name: {'groups': 0, 'captures': 0} for name in SPLITS}
    for group, split in assignment.items():
        counts[split]['groups'] += 1
        counts[split]['captures'] += groups[group]
    record = {'schema': SCHEMA if group_by == 'session' else SCHEMA_1_1, 'strategy': 'GROUP_HASH',
              'group_by': group_by, 'seed': seed, 'ratios': ratios,
              'overrides': dict(sorted(overrides.items())), 'assignment': dict(sorted(assignment.items())),
              'counts': counts}
    if parent is not None:
        record['inherited'] = {'version': parent_version, 'split_id': parent.split_id}
        if retired:
            record['retired'] = dict(sorted(retired.items()))
    record['split_id'] = _split_id(record)
    return SplitAssignment.from_dict(record, 'split')        # one validation path


def check_split_rules(split: SplitAssignment, parent: SplitAssignment = None) -> list:
    """Problems of a stored split (``verify``): a group new to the lineage is assigned by its override or its
    hash; a group of the parent's lineage is in exactly the split it had there, and none is forgotten; an inherited
    split has its parent's seed and ratios."""
    r = split.record
    problems = []
    history = parent.history if parent is not None else {}
    if parent is not None:
        for key in ('seed', 'ratios', 'group_by'):
            if r[key] != parent.record[key]:
                problems.append(f'an inherited split changes the {key} ({parent.record[key]} -> {r[key]})')
        forgotten = sorted(set(history) - set(split.history))
        if forgotten:
            problems.append(f'groups {forgotten} of the lineage are neither assigned nor retired')
    for group, assigned in split.history.items():
        if group in history:
            expected = history[group]
        elif group in r.get('retired', {}):
            expected = None
        else:
            expected = r['overrides'].get(group) or hashed_split(r['seed'], r['ratios'], group)
        if assigned != expected:
            problems.append(f'group {group} is in {assigned}, the rule gives {expected}')
    return problems


__all__ = ['SplitAssignment', 'build_split', 'hashed_split', 'bucket_of', 'check_split_rules', 'SPLITS',
           'DEFAULT_RATIOS', 'GROUP_BY', 'DEFAULT_GROUP_BY']
