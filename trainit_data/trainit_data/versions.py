"""Dataset versions: immutable, content-hashed, derived with an explicit op (PUBLIC, D0.6).

A DatasetVersion (``trainit-dataset/1.0``) is an immutable statement of WHICH canonical content a dataset holds
at version ``n``::

    datasets/<dataset_id>/versions/<n>/selection.jsonl   the selected captures (OBSERVATIONS), sorted
    datasets/<dataset_id>/versions/<n>/splits.json       trainit-split/1.0
    datasets/<dataset_id>/versions/<n>/manifest.json     the manifest, written LAST (its completeness mark)

The manifest has a CONTENT-KIND NEUTRAL header -- identity (dataset, version), ``content_kind``, derivation
(op + parent version and content hash), rights and their basis, software, created, ``content_hash`` -- and a
``content`` block whose reader is chosen by ``content_kind``. D0.6 implements ``OBSERVATIONS`` (sessions,
calibrations, the capture selection, label sets, the split); ``EPISODES`` (D1) is reserved and refused with
``CONTENT_KIND_UNSUPPORTED`` until its reader exists. A D1 episode will reference D0 observations by id, never
copy them; a D3 adapter maps a canonical episode to an external format (LeRobot is never the source of truth).

``content_hash`` = sha256 of the canonical manifest without ``content_hash``, ``created`` and ``software``: a
deterministic function of the content (the same inputs give the same hash, whoever writes them, whenever).

Immutability: a version directory is never written twice (``VERSION_EXISTS``), a session is never modified, and
the version pins every session by its sealed index hash, every calibration and label set by its hash, and its
selection and split files by theirs; ``verify`` re-checks them all. A change is a NEW version derived from its
parent:

==============  ======================================================================================
CREATE          no parent: sessions, the capture-outcome filter, label sets, a split
ADD_SESSIONS    the parent + new sessions (and optionally new label sets); the split is inherited
FILTER          the parent minus sessions and / or a NARROWER outcome filter; the split is inherited
RELABEL         the parent with label sets added / dropped (a correction: a new set that supersedes one)
RESPLIT         the parent with a new split (seed / ratios / overrides -- the parent's overrides are NOT
                carried: a RESPLIT states its overrides in full); nothing else changes
==============  ======================================================================================

The op set is scoped by ``content_kind``: these five are the OBSERVATIONS ops (the vocabulary's
``derivation_ops`` since D0.1). A content kind that adds ops of its own (D1: e.g. adding episodes) adds them with
its reader, by a minor schema bump; an older reader refuses an op it does not know, never misreads it.

The helpers below that read ``content['sessions']`` / ``content['label_sets']`` (``session_ids``,
``label_set_ids``, ``read_selection``, ``load_split``, ``labelled_observations``, ``create_version``) are
OBSERVATIONS-scoped; the header (identity, derivation, rights, hash) is what every kind shares.

Known limit: versions are not chained in a store-wide ledger, so deleting the HIGHEST version directory by hand is
not detectable and its number would be reused (a deleted lower version is detected: its children name it).

Source independence: nothing here branches on a backend or a transport. A session's source is carried as
provenance, so a simulated session and a physical camera's session are selected, labelled and split the
same way.

Grouping (D0.7): a CREATE / RESPLIT chooses the split's ``group_by`` (``session``, the default, or
``randomization_run``); every other op inherits it. Under a grouping other than ``session`` each selection line
records its capture's ``group`` (``synthetic.group_of``: the capture's randomization run, or its session when no
generator fired it), the manifest's split reference names the grouping, and the manifest is written as
``trainit-dataset/1.1``; a version grouped by session is the D0.6 record, byte for byte.
"""
import re
from dataclasses import dataclass, field

from . import __version__
from . import _strict as s
from ._canonical import canonical_json, parse_json, pretty_json, sha256_bytes, sha256_json
from ._fsio import makedirs_durable, write_new_file
from ._sealed import SealedSessions, capture_key, capture_of, observation_key
from .capture import OUTCOMES
from .dataset import (combine_rights, created, created_now, dataset_id as dataset_id_check, dataset_rel,
                      load_dataset, make_read_only, rights as rights_check)
from .errors import (CONTENT_KIND_UNSUPPORTED, DATASET_INCONSISTENT, DERIVATION_INVALID, HASH_MISMATCH,
                     INVALID_VALUE, LABEL_CONFLICT, LABEL_SET_NOT_FOUND, VERSION_EXISTS, VERSION_NOT_FOUND,
                     WRITE_FAILED, DataError)
from .labels import label_set_id, load_label_set, read_entries
from .reader import index_lines
from .session_record import STATUSES
from .splits import DEFAULT_GROUP_BY, GROUP_BY, SplitAssignment, build_split, split_id
from .synthetic import TRIGGER_KIND as SYNTHETIC_FRAME, group_of

SCHEMA_NAME = 'trainit-dataset'
SCHEMA_MAJOR = 1
SCHEMA = 'trainit-dataset/1.0'
SCHEMA_1_1 = 'trainit-dataset/1.1'           # a split grouped by anything but session (D0.7)
SELECTION_FILE = 'selection.jsonl'
SPLIT_FILE = 'splits.json'
MANIFEST_FILE = 'manifest.json'

CONTENT_KINDS = frozenset({'OBSERVATIONS', 'EPISODES'})      # EPISODES: reserved for D1 (no reader)
DERIVATION_OPS = frozenset({'CREATE', 'ADD_SESSIONS', 'FILTER', 'RELABEL', 'RESPLIT'})
DEFAULT_OUTCOMES = ('COMPLETE',)
_NOT_HASHED = ('content_hash', 'created', 'software')
VERSION_DIR_RE = re.compile(r'^[1-9][0-9]*$')            # ASCII digits, no leading zero
_NUMBER_ATTEMPTS = 16


# ---- manifest header -------------------------------------------------------------------------------------

def _parent(value, where):
    f = s.Fields(value, where)
    out = {'version': f.req('version', s.integer(1)), 'content_hash': f.req('content_hash', s.sha256)}
    f.done()
    return out


def _derivation(value, where):
    f = s.Fields(value, where)
    out = {'op': f.req('op', s.one_of(DERIVATION_OPS))}
    s.put(out, 'parent', f.opt('parent', _parent))
    f.done()
    if (out['op'] == 'CREATE') != ('parent' not in out):
        s.fail(INVALID_VALUE, 'CREATE has no parent; every other op names its parent', where)
    return out


def _basis(value, where):
    f = s.Fields(value, where)
    out = {'source': f.req('source', s.string), 'rights': f.req('rights', rights_check)}
    f.done()
    return out


def _software(value, where):
    f = s.Fields(value, where)
    out = {'trainit_data': f.req('trainit_data', s.string)}
    f.done()
    return out


# ---- OBSERVATIONS content ----------------------------------------------------------------------------------

def _session_ref(value, where):
    f = s.Fields(value, where)
    out = {'session_id': f.req('session_id', s.session_id),
           'status': f.req('status', s.one_of(STATUSES)),
           'index_sha256': f.req('index_sha256', s.sha256),
           'vocab_sha256': f.req('vocab_sha256', s.sha256),
           'source': f.req('source', s.str_map),          # {transport, backend}: provenance, never a branch
           'clock': f.req('clock', s.str_map),            # {session_domain, sim_source?}
           'calibration_ids': list(f.req('calibration_ids', s.list_of(s.calibration_id, unique=True))),
           'captures_selected': f.req('captures_selected', s.integer(0))}
    f.done()
    return out


def _calibration_ref(value, where):
    f = s.Fields(value, where)
    out = {'sha256': f.req('sha256', s.sha256), 'sensor_id': f.req('sensor_id', s.ident),
           'stream': f.req('stream', s.ident), 'intrinsics_kind': f.req('intrinsics_kind', s.token),
           'extrinsics_kinds': list(f.req('extrinsics_kinds', s.list_of(s.token)))}
    f.done()
    return out


def _calibrations(value, where):
    value = s.opaque(value, where)
    return {s.calibration_id(k, f'{where}.<key>'): _calibration_ref(v, f'{where}.{k}') for k, v in value.items()}


def _outcome_counts(value, where):
    f = s.Fields(value, where)
    out = {name: f.req(name, s.integer(0)) for name in sorted(OUTCOMES)}
    f.done()
    return out


def _selection(value, where):
    f = s.Fields(value, where)
    ff = s.Fields(f.req('filter', s.opaque), f'{where}.filter')
    include = list(ff.req('include_outcomes', s.list_of(s.one_of(OUTCOMES), min_length=1, unique=True)))
    ff.done()
    out = {'filter': {'include_outcomes': sorted(include)}, 'path': f.req('path', s.one_of({SELECTION_FILE})),
           'sha256': f.req('sha256', s.sha256), 'captures': f.req('captures', s.integer(0)),
           'excluded': f.req('excluded', _outcome_counts)}
    f.done()
    if include != sorted(include):
        s.fail(INVALID_VALUE, 'include_outcomes is sorted', f'{where}.filter')
    return out


def _label_ref(value, where):
    f = s.Fields(value, where)
    out = {'label_set_id': f.req('label_set_id', label_set_id), 'sha256': f.req('sha256', s.sha256),
           'kind': f.req('kind', s.token), 'source': f.req('source', s.token),
           'entries_selected': f.req('entries_selected', s.integer(0))}
    f.done()
    return out


def _split_ref(value, where):
    f = s.Fields(value, where)
    out = {'split_id': f.req('split_id', split_id), 'path': f.req('path', s.one_of({SPLIT_FILE})),
           'sha256': f.req('sha256', s.sha256), 'counts': f.req('counts', s.opaque)}
    group_by = f.opt('group_by', s.one_of(GROUP_BY - {DEFAULT_GROUP_BY}))     # absent = session (D0.6)
    s.put(out, 'group_by', group_by)
    f.done()
    return out


def _observations_content(value, where):
    f = s.Fields(value, where)
    out = {'sessions': [dict(x) for x in f.req('sessions', s.list_of(_session_ref, min_length=1))],
           'calibrations': f.req('calibrations', _calibrations),
           'selection': f.req('selection', _selection),
           'label_sets': [dict(x) for x in f.req('label_sets', s.list_of(_label_ref))],
           'split': f.req('split', _split_ref)}
    f.done()
    ids = [x['session_id'] for x in out['sessions']]
    if ids != sorted(ids) or len(set(ids)) != len(ids):
        s.fail(INVALID_VALUE, 'sessions are unique and sorted by id', f'{where}.sessions')
    ls = [x['label_set_id'] for x in out['label_sets']]
    if ls != sorted(ls) or len(set(ls)) != len(ls):
        s.fail(INVALID_VALUE, 'label sets are unique and sorted by id', f'{where}.label_sets')
    used = sorted({c for x in out['sessions'] for c in x['calibration_ids']})
    if used != sorted(out['calibrations']):
        s.fail(INVALID_VALUE, 'calibrations are exactly those of the sessions', f'{where}.calibrations')
    if sum(x['captures_selected'] for x in out['sessions']) != out['selection']['captures']:
        s.fail(INVALID_VALUE, 'the sessions\' selected captures do not add up to the selection', where)
    return out


# The content readers this core implements, chosen by content_kind (the seam of D1's EPISODES).
_CONTENT_READERS = {'OBSERVATIONS': _observations_content}


@dataclass(frozen=True)
class DatasetVersion:
    schema: str
    dataset_id: str
    version: int
    content_kind: str
    derivation: dict
    rights: dict
    rights_basis: tuple
    content: dict
    software: dict
    created: dict
    content_hash: str
    extensions: dict = field(default_factory=dict)

    @classmethod
    def from_dict(cls, value, where='manifest') -> 'DatasetVersion':
        f = s.Fields(value, where)
        kind = f.req('content_kind', s.one_of(CONTENT_KINDS))
        reader = _CONTENT_READERS.get(kind)
        if reader is None:
            raise DataError(CONTENT_KIND_UNSUPPORTED, f'{where}: content kind {kind} has no reader in this core '
                            '(D0.6 reads OBSERVATIONS)', context={'where': where, 'content_kind': kind})
        record = cls(schema=f.req('schema', s.schema(SCHEMA_NAME, SCHEMA_MAJOR)),
                     dataset_id=f.req('dataset_id', dataset_id_check),
                     version=f.req('version', s.integer(1)),
                     content_kind=kind,
                     derivation=f.req('derivation', _derivation),
                     rights=f.req('rights', rights_check),
                     rights_basis=f.req('rights_basis', s.list_of(_basis, min_length=1)),
                     content=f.req('content', reader),
                     software=f.req('software', _software),
                     created=f.req('created', created),
                     content_hash=f.req('content_hash', s.sha256),
                     extensions=f.extensions())
        f.done()
        if record.schema == SCHEMA and record.content_kind == 'OBSERVATIONS' \
                and 'group_by' in record.content['split']:
            s.fail(INVALID_VALUE, f'a split grouped by {record.content["split"]["group_by"]} needs {SCHEMA_1_1}', where)
        parent = record.derivation.get('parent')
        if parent is not None and parent['version'] >= record.version:
            s.fail(INVALID_VALUE, 'a parent version precedes its child', f'{where}.derivation')
        expected = combine_rights(list(record.rights_basis))
        if expected != record.rights:
            s.fail(INVALID_VALUE, f'rights {record.rights} are not the combination of their basis ({expected})',
                   f'{where}.rights')
        actual = content_hash_of(value)               # the recorded JSON itself, as written
        if record.content_hash != actual:
            raise DataError(HASH_MISMATCH, f'{where}: content_hash does not match the manifest content',
                            context={'where': where, 'recorded': record.content_hash, 'actual': actual})
        return record

    def to_dict(self) -> dict:
        out = {'schema': self.schema, 'dataset_id': self.dataset_id, 'version': self.version,
               'content_kind': self.content_kind, 'derivation': dict(self.derivation), 'rights': dict(self.rights),
               'rights_basis': [dict(b) for b in self.rights_basis], 'content': self.content,
               'software': dict(self.software), 'created': dict(self.created), 'content_hash': self.content_hash}
        if self.extensions:
            out['extensions'] = dict(self.extensions)
        return out

    @property
    def session_ids(self) -> list:
        return [x['session_id'] for x in self.content['sessions']]

    @property
    def label_set_ids(self) -> list:
        return [x['label_set_id'] for x in self.content['label_sets']]

    @property
    def group_by(self) -> str:
        return self.content['split'].get('group_by', DEFAULT_GROUP_BY)


def content_hash_of(manifest: dict) -> str:
    return sha256_json({k: v for k, v in manifest.items() if k not in _NOT_HASHED})


# ---- store layout and reads ------------------------------------------------------------------------------

def versions_rel(dataset: str) -> str:
    return f'{dataset_rel(dataset)}/versions'


def version_rel(dataset: str, version: int) -> str:
    return f'{versions_rel(dataset)}/{s.integer(1)(version, "version")}'


def _version_numbers(store, dataset: str, *, complete_only: bool) -> list:
    root = store.path(versions_rel(dataset))
    if not root.is_dir():
        return []
    numbers = [int(p.name) for p in root.iterdir() if p.is_dir() and VERSION_DIR_RE.match(p.name)]
    if complete_only:
        numbers = [n for n in numbers if (root / str(n) / MANIFEST_FILE).is_file()]
    return sorted(numbers)


def list_versions(store, dataset: str) -> list:
    return _version_numbers(store, dataset, complete_only=True)


def load_version(store, dataset: str, version: int) -> DatasetVersion:
    rel = f'{version_rel(dataset, version)}/{MANIFEST_FILE}'
    path = store.path(rel)
    if not path.is_file():
        raise DataError(VERSION_NOT_FOUND, f'dataset {dataset} has no version {version}',
                        context={'dataset_id': dataset, 'version': version})
    record = DatasetVersion.from_dict(parse_json(path.read_bytes(), rel), rel)
    if record.dataset_id != dataset or record.version != version:
        s.fail(DATASET_INCONSISTENT, f'the manifest names {record.dataset_id} version {record.version}', rel)
    return record


def _checked_file(store, rel: str, expected: str) -> bytes:
    path = store.path(rel)
    data = path.read_bytes() if path.is_file() else None
    if data is None or sha256_bytes(data) != expected:
        raise DataError(HASH_MISMATCH, f'{rel} does not hash to the recorded {expected}',
                        context={'path': rel, 'expected': expected})
    return data


def read_selection(store, version: DatasetVersion) -> list:
    """``[{capture_id, outcome, group?}]`` of an OBSERVATIONS version, in canonical order (hash-checked); ``group``
    is present exactly when the version is not grouped by session."""
    sel = version.content['selection']
    rel = f'{version_rel(version.dataset_id, version.version)}/{SELECTION_FILE}'
    _checked_file(store, rel, sel['sha256'])
    lines = []
    for number, _, raw, complete in index_lines(store.path(rel)):
        where = f'{rel}:{number}'
        if not complete:
            s.fail(DATASET_INCONSISTENT, 'the selection ends with an incomplete line', where)
        f = s.Fields(parse_json(raw, where), where)
        line = {'capture_id': f.req('capture_id', s.string), 'outcome': f.req('outcome', s.one_of(OUTCOMES))}
        s.put(line, 'group', f.opt('group', s.string))
        f.done()
        if ('group' in line) != (version.group_by != DEFAULT_GROUP_BY):
            s.fail(DATASET_INCONSISTENT, f'a selection line records its group exactly when the version is grouped by '
                   f'something other than session (here: {version.group_by})', where)
        lines.append(line)
    return lines


def group_of_line(line: dict) -> str:
    """The split group of a selection line: its recorded group, or its session."""
    return line.get('group') or line['capture_id'].split('/')[0]


def load_split(store, version: DatasetVersion) -> SplitAssignment:
    ref = version.content['split']
    rel = f'{version_rel(version.dataset_id, version.version)}/{SPLIT_FILE}'
    data = _checked_file(store, rel, ref['sha256'])
    split = SplitAssignment.from_dict(parse_json(data, rel), rel)
    if split.split_id != ref['split_id']:
        s.fail(DATASET_INCONSISTENT, f'{rel} is split {split.split_id}, the manifest pins {ref["split_id"]}', rel)
    if split.group_by != version.group_by:
        s.fail(DATASET_INCONSISTENT, f'{rel} groups by {split.group_by}, the manifest by {version.group_by}', rel)
    if version.group_by == DEFAULT_GROUP_BY:
        groups = set(version.session_ids)
    else:
        groups = {group_of_line(x) for x in read_selection(store, version)}
    if set(split.assignment) != groups:
        s.fail(DATASET_INCONSISTENT, f'the split does not assign exactly the version\'s groups ({version.group_by})',
               rel)
    return split


def capture_splits(store, version: DatasetVersion, split: SplitAssignment = None) -> dict:
    """``{capture_id: split name}`` of every selected capture: its group's split (the exporters' only source)."""
    split = split or load_split(store, version)
    return {x['capture_id']: split.split_of(group_of_line(x)) for x in read_selection(store, version)}


# ---- create ----------------------------------------------------------------------------------------------

def _derive_spec(op, parent, parent_split, *, sessions, remove_sessions, include_outcomes, label_sets,
                 drop_label_sets, seed, ratios, overrides, group_by):
    """The full spec of the new version: ``(sessions, outcomes, label_sets, split_args, split_parent)``.
    Refuses (``DERIVATION_INVALID``) an argument the op does not take, and an op that changes nothing."""
    def refuse(message, **context):
        raise DataError(DERIVATION_INVALID, f'{op}: {message}', context={'op': op, **context})

    given = {'sessions': bool(sessions), 'remove_sessions': bool(remove_sessions),
             'include_outcomes': include_outcomes is not None, 'label_sets': bool(label_sets),
             'drop_label_sets': bool(drop_label_sets), 'seed': seed is not None, 'ratios': ratios is not None,
             'overrides': bool(overrides), 'group_by': group_by is not None}
    allowed = {'CREATE': {'sessions', 'include_outcomes', 'label_sets', 'seed', 'ratios', 'overrides', 'group_by'},
               'ADD_SESSIONS': {'sessions', 'label_sets', 'overrides'},
               'FILTER': {'remove_sessions', 'include_outcomes'},
               'RELABEL': {'label_sets', 'drop_label_sets'},
               'RESPLIT': {'seed', 'ratios', 'overrides', 'group_by'}}[op]
    extra = sorted(k for k, v in given.items() if v and k not in allowed)
    if extra:
        refuse(f'does not take {extra}', arguments=extra)
    if op == 'CREATE':
        if parent is not None:
            refuse('a CREATE has no parent')
        if not sessions:
            refuse('a version references at least one session')
        outcomes = sorted(set(include_outcomes or DEFAULT_OUTCOMES))
        split_args = {'seed': 0 if seed is None else seed, 'ratios': ratios, 'overrides': overrides,
                      'group_by': group_by or DEFAULT_GROUP_BY}
        return sorted(set(sessions)), outcomes, sorted(set(label_sets)), split_args, None
    if parent is None:
        refuse('names its parent version')
    p_sessions, p_outcomes = parent.session_ids, parent.content['selection']['filter']['include_outcomes']
    p_labels, p_split = parent.label_set_ids, parent_split.record
    inherited = {'seed': p_split['seed'], 'ratios': p_split['ratios'], 'overrides': overrides,
                 'group_by': p_split['group_by']}
    if op == 'ADD_SESSIONS':
        if not sessions:
            refuse('adds at least one session')
        already = sorted(set(sessions) & set(p_sessions))
        if already:
            refuse(f'sessions {already} are already in version {parent.version}', sessions=already)
        return (sorted(set(p_sessions) | set(sessions)), p_outcomes, sorted(set(p_labels) | set(label_sets)),
                inherited, parent_split)
    if op == 'FILTER':
        unknown = sorted(set(remove_sessions) - set(p_sessions))
        if unknown:
            refuse(f'sessions {unknown} are not in version {parent.version}', sessions=unknown)
        outcomes = p_outcomes if include_outcomes is None else sorted(set(include_outcomes))
        if not set(outcomes) <= set(p_outcomes):
            refuse(f'outcomes {outcomes} widen the parent\'s {p_outcomes}: a FILTER only narrows')
        kept = sorted(set(p_sessions) - set(remove_sessions))
        if not kept:
            refuse('removes every session')
        if kept == p_sessions and outcomes == p_outcomes:
            refuse('removes nothing')
        return kept, outcomes, p_labels, inherited, parent_split
    if op == 'RELABEL':
        unknown = sorted(set(drop_label_sets) - set(p_labels))
        if unknown:
            refuse(f'label sets {unknown} are not in version {parent.version}', label_sets=unknown)
        labels = sorted((set(p_labels) - set(drop_label_sets)) | set(label_sets))
        if labels == p_labels:
            refuse('changes no label set')
        return p_sessions, p_outcomes, labels, inherited, parent_split
    # RESPLIT: a new split, nothing inherited
    split_args = {'seed': p_split['seed'] if seed is None else seed,
                  'ratios': p_split['ratios'] if ratios is None else ratios, 'overrides': overrides,
                  'group_by': p_split['group_by'] if group_by is None else group_by}
    return p_sessions, p_outcomes, p_labels, split_args, None


def derivation_problems(version: DatasetVersion, parent: DatasetVersion) -> list:
    """What a stored derived version breaks of its op's rule relative to its parent (``verify``)."""
    op = version.derivation['op']
    s_new, s_old = set(version.session_ids), set(parent.session_ids)
    o_new = set(version.content['selection']['filter']['include_outcomes'])
    o_old = set(parent.content['selection']['filter']['include_outcomes'])
    l_new, l_old = set(version.label_set_ids), set(parent.label_set_ids)
    same_split = version.content['split']['split_id'] == parent.content['split']['split_id']
    rules = {'ADD_SESSIONS': [(s_new > s_old, 'adds sessions and keeps the parent\'s'), (o_new == o_old, 'keeps the '
                              'outcome filter'), (l_new >= l_old, 'keeps the parent\'s label sets')],
             'FILTER': [(s_new <= s_old and o_new <= o_old and (s_new, o_new) != (s_old, o_old), 'only narrows'),
                        (l_new == l_old, 'keeps the label sets')],
             'RELABEL': [(s_new == s_old and o_new == o_old, 'keeps sessions and outcomes'),
                         (l_new != l_old, 'changes the label sets')],
             'RESPLIT': [(s_new == s_old and o_new == o_old and l_new == l_old, 'changes the split only'),
                         (not same_split, 'changes the split')]}[op]
    return [f'a {op} version {what}: version {version.version} does not' for ok, what in rules if not ok]


def create_version(store, dataset: str, *, op: str = 'CREATE', parent: int = None, sessions=(), remove_sessions=(),
                   include_outcomes=None, label_sets=(), drop_label_sets=(), seed: int = None, ratios: dict = None,
                   overrides: dict = None, version: int = None, group_by: str = None) -> DatasetVersion:
    """Build and write a new immutable version of ``dataset``; returns it.

    ``version`` defaults to the next free number; an existing one is refused with ``VERSION_EXISTS``. Every session
    must be sealed with an intact index; label sets must belong to the dataset; the rights are the most restrictive
    of the dataset's and every session's (``RIGHTS_CONFLICT`` when that needs a guess). ``group_by`` (CREATE /
    RESPLIT only; default ``session``) chooses what a split group is."""
    identity = load_dataset(store, dataset)
    op = s.one_of(DERIVATION_OPS)(op, 'op')
    parent_version = parent_split = None
    if parent is not None:
        parent_version = load_version(store, dataset, parent)
        if parent_version.content_kind != 'OBSERVATIONS':
            raise DataError(CONTENT_KIND_UNSUPPORTED, 'only OBSERVATIONS versions are derived in D0.6',
                            context={'content_kind': parent_version.content_kind})
        parent_split = load_split(store, parent_version)
    session_ids, outcomes, ls_ids, split_args, split_parent = _derive_spec(
        op, parent_version, parent_split, sessions=list(sessions), remove_sessions=list(remove_sessions),
        include_outcomes=None if include_outcomes is None else list(include_outcomes), label_sets=list(label_sets),
        drop_label_sets=list(drop_label_sets), seed=seed, ratios=ratios, overrides=overrides, group_by=group_by)
    grouping = s.one_of(GROUP_BY)(split_args['group_by'], 'group_by')
    for sid in session_ids:
        s.session_id(sid, 'sessions')
    outcomes = list(s.list_of(s.one_of(OUTCOMES), min_length=1, unique=True)(outcomes, 'include_outcomes'))

    # sessions -> selection
    sealed = SealedSessions(store)
    session_refs, calibrations, selection, groups, generated = [], {}, [], {}, set()
    excluded = {name: 0 for name in sorted(OUTCOMES)}
    basis = [{'source': f'dataset:{dataset}', 'rights': identity.rights}]
    for sid in session_ids:
        session = sealed.get(sid)                                   # SESSION_NOT_FOUND / NOT_SEALED / HASH_MISMATCH
        header = session.header
        chosen = [c for c in session.captures if c.outcome in outcomes]
        for c in session.captures:
            if c.outcome not in outcomes:
                excluded[c.outcome] += 1
        if any(c.trigger.kind == SYNTHETIC_FRAME for c in chosen):
            generated.add(sid)
        for c in chosen:
            line = {'capture_id': c.capture_id, 'outcome': c.outcome}
            group = group_of(c, grouping)                          # INVALID_VALUE: a malformed generated frame
            if grouping != DEFAULT_GROUP_BY:
                line['group'] = group
            selection.append(line)
            groups[group] = groups.get(group, 0) + 1
        if grouping == DEFAULT_GROUP_BY:
            groups[sid] = len(chosen)                             # D0.6: a session is a group, even with 0 captures
        clock = {'session_domain': header.clock.session_domain}
        s.put(clock, 'sim_source', header.clock.sim_source)
        session_refs.append({'session_id': sid, 'status': header.status, 'index_sha256': header.index['sha256'],
                             'vocab_sha256': header.vocab['sha256'],
                             'source': {'transport': header.source.transport, 'backend': header.source.backend},
                             'clock': clock, 'calibration_ids': list(header.calibration_ids),
                             'captures_selected': len(chosen)})
        basis.append({'source': sid, 'rights': dict(header.rights)})
        for cal_id in header.calibration_ids:
            cal = store.get_calibration(cal_id, session.vocab)
            calibrations[cal_id] = {'sha256': cal.sha256, 'sensor_id': cal.sensor_id, 'stream': cal.stream,
                                    'intrinsics_kind': cal.intrinsics_provenance.kind,
                                    'extrinsics_kinds': [e.provenance.kind for e in cal.extrinsics]}
    if grouping == DEFAULT_GROUP_BY and generated:
        raise DataError(INVALID_VALUE, f'sessions {sorted(generated)} hold generated (synthetic) captures: group the '
                        'split by randomization_run (a session group would let one randomized scene, regenerated in '
                        'another session, sit in two splits)', context={'sessions': sorted(generated)})
    selection.sort(key=lambda x: capture_key(x['capture_id']))
    selected = {x['capture_id'] for x in selection}
    rights = combine_rights(basis)

    # label sets
    label_refs, labelled = [], {}
    for ls_id in ls_ids:
        label_set = load_label_set(store, dataset, ls_id)
        inside = 0
        for entry in read_entries(store, label_set):
            obs = entry['observation_id']
            if capture_of(obs) not in selected:
                continue
            inside += 1
            other = labelled.get((label_set.kind, obs))
            if other is not None:
                raise DataError(LABEL_CONFLICT, f'{obs} is labelled ({label_set.kind}) by both {other} and {ls_id}',
                                context={'observation_id': obs, 'label_sets': [other, ls_id]})
            labelled[(label_set.kind, obs)] = ls_id
        label_refs.append({'label_set_id': ls_id, 'sha256': label_set.sha256, 'kind': label_set.kind,
                           'source': label_set.provenance['source'], 'entries_selected': inside})

    # split
    split = build_split(groups, parent=split_parent, parent_version=parent, **split_args)
    if op == 'RESPLIT' and split.split_id == parent_split.split_id:
        raise DataError(DERIVATION_INVALID, 'RESPLIT: the new split is the parent\'s split (same seed, ratios, '
                        'overrides)',
                        context={'split_id': split.split_id})
    selection_bytes = b''.join(canonical_json(x) + b'\n' for x in selection)
    split_bytes = pretty_json(split.to_dict())
    derivation = {'op': op}
    if parent_version is not None:
        derivation['parent'] = {'version': parent_version.version, 'content_hash': parent_version.content_hash}
    content = {'sessions': session_refs, 'calibrations': dict(sorted(calibrations.items())),
               'selection': {'filter': {'include_outcomes': sorted(outcomes)}, 'path': SELECTION_FILE,
                             'sha256': sha256_bytes(selection_bytes), 'captures': len(selection), 'excluded': excluded},
               'label_sets': label_refs,
               'split': {'split_id': split.split_id, 'path': SPLIT_FILE, 'sha256': sha256_bytes(split_bytes),
                         'counts': split.record['counts']}}
    if grouping != DEFAULT_GROUP_BY:
        content['split']['group_by'] = grouping
    versions_dir = store.path(versions_rel(dataset))
    try:
        makedirs_durable(versions_dir)
    except OSError as exc:
        raise DataError(WRITE_FAILED, f'creating {versions_rel(dataset)} failed', cause=exc) from None
    for _ in range(_NUMBER_ATTEMPTS):
        number = version if version is not None else (_version_numbers(store, dataset, complete_only=False)
                                                      or [0])[-1] + 1
        manifest = {'schema': SCHEMA if grouping == DEFAULT_GROUP_BY else SCHEMA_1_1, 'dataset_id': dataset,
                    'version': number, 'content_kind': 'OBSERVATIONS',
                    'derivation': derivation, 'rights': rights, 'rights_basis': basis, 'content': content,
                    'software': {'trainit_data': __version__}, 'created': created_now()}
        manifest['content_hash'] = content_hash_of(manifest)
        record = DatasetVersion.from_dict(manifest, f'{version_rel(dataset, number)}/{MANIFEST_FILE}')   # one path
        directory = versions_dir / str(number)
        try:
            directory.mkdir()                                       # exclusive: the number is taken once
            break
        except FileExistsError:
            if version is not None:
                raise DataError(VERSION_EXISTS, f'dataset {dataset} version {number} already exists: a version is '
                                'never overwritten (derive a new one)',
                                context={'dataset_id': dataset, 'version': number}) from None
            # another writer took the next number meanwhile: take the following one
        except OSError as exc:
            raise DataError(WRITE_FAILED, f'creating {version_rel(dataset, number)} failed', cause=exc) from None
    else:
        raise DataError(VERSION_EXISTS, f'no free version number for {dataset} after {_NUMBER_ATTEMPTS} attempts',
                        context={'dataset_id': dataset})
    try:
        for name, data in ((SELECTION_FILE, selection_bytes), (SPLIT_FILE, split_bytes),
                           (MANIFEST_FILE, pretty_json(manifest))):
            write_new_file(directory / name, data)
            make_read_only(directory / name)
    except OSError as exc:
        raise DataError(WRITE_FAILED, f'writing {version_rel(dataset, number)} failed (its number stays taken; '
                        'verify reports the incomplete directory)', cause=exc,
                        context={'dataset_id': dataset, 'version': number}) from None
    return record


def labelled_observations(store, version: DatasetVersion, label_set_ids) -> dict:
    """``{observation_id: [(label_set, entry)]}`` of the version's label sets ``label_set_ids``, restricted to the
    selection; each label set is re-checked against the hash the version pins."""
    selected = {x['capture_id'] for x in read_selection(store, version)}
    unknown = sorted(set(label_set_ids) - set(version.label_set_ids))
    if unknown:
        raise DataError(LABEL_SET_NOT_FOUND, f'label sets {unknown} are not in {version.dataset_id} version '
                        f'{version.version}', context={'label_sets': unknown})
    out = {}
    for ref in version.content['label_sets']:
        if ref['label_set_id'] not in label_set_ids:
            continue
        label_set = load_label_set(store, version.dataset_id, ref['label_set_id'])
        if label_set.sha256 != ref['sha256']:
            raise DataError(HASH_MISMATCH, f'label set {ref["label_set_id"]} no longer hashes to the pinned '
                            f'{ref["sha256"]}', context={'label_set_id': ref['label_set_id']})
        for entry in read_entries(store, label_set):
            if capture_of(entry['observation_id']) in selected:
                out.setdefault(entry['observation_id'], []).append((label_set, entry))
    return dict(sorted(out.items(), key=lambda kv: observation_key(kv[0])))


__all__ = ['DatasetVersion', 'create_version', 'load_version', 'list_versions', 'read_selection', 'load_split',
           'capture_splits', 'group_of_line', 'labelled_observations', 'content_hash_of', 'derivation_problems',
           'CONTENT_KINDS', 'DERIVATION_OPS']
