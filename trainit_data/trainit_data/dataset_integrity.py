"""verify for datasets: identity, label sets, versions (PUBLIC, D0.6).

``verify_dataset`` INSPECTS and never changes a byte. Beyond each record's own parse (strict fields, the label
set's content id, the manifest's content hash) it re-checks everything a version pins against what the store
holds now: every session is sealed with the index hash pinned and still verifies, every calibration and label
set hashes to its pin, the selection is exactly the captures of the pinned sessions that pass the outcome filter,
every label still lies inside its image, no two label sets of one kind label one observation, every group is in
exactly one split by its rule (or its parent's split when inherited), and the rights are still the combination
of their basis. A version directory or label set without its header (an interrupted creation) is a WARNING.
"""
from dataclasses import dataclass, field

from ._sealed import SealedSessions, capture_key, capture_of
from .dataset import dataset_rel, load_dataset
from .errors import DATASET_INCONSISTENT, DERIVATION_INVALID, HASH_MISMATCH, LABEL_CONFLICT, LABEL_INVALID, DataError
from .ground_truth import rederive_problems
from .integrity import Finding, verify_session
from .labels import check_entry_in_store, labels_rel, load_label_set, read_entries
from .splits import DEFAULT_GROUP_BY, check_split_rules
from .synthetic import TRIGGER_KIND, group_of
from .versions import (MANIFEST_FILE, SELECTION_FILE, SPLIT_FILE, VERSION_DIR_RE, derivation_problems, group_of_line,
                       list_versions, load_split, load_version, read_selection, version_rel, versions_rel)


@dataclass
class DatasetReport:
    dataset_id: str
    versions: list = field(default_factory=list)
    label_sets: list = field(default_factory=list)
    findings: list = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not any(f.severity == 'ERROR' for f in self.findings)

    def to_dict(self):
        return {'dataset_id': self.dataset_id, 'ok': self.ok, 'versions': list(self.versions),
                'label_sets': list(self.label_sets), 'findings': [f.to_dict() for f in self.findings]}


class _Check:
    def __init__(self, store, dataset, session_ok=None):
        self.store = store
        self.dataset = dataset
        self.sessions = SealedSessions(store)
        self.session_reports = dict(session_ok or {})
        self.findings = []

    def add(self, code, artifact, detail, severity='ERROR', record=None):
        self.findings.append(Finding(code, severity, artifact, detail, record))

    def guard(self, artifact, fn, *args):
        try:
            return fn(*args)
        except DataError as exc:
            self.add(exc.code, artifact, exc.message)
        except OSError as exc:
            self.add(DATASET_INCONSISTENT, artifact, f'unreadable: {exc}')
        return None

    def session_ok(self, session_id) -> bool:
        if session_id not in self.session_reports:
            report = verify_session(self.store, session_id)
            self.session_reports[session_id] = report.ok
        return self.session_reports[session_id]


def verify_dataset(store, dataset: str, session_ok: dict = None) -> DatasetReport:
    """Inspect one dataset (``DATASET_NOT_FOUND`` if it does not exist); never modifies anything. ``session_ok``
    (``{session_id: bool}``) passes results of sessions already verified, so they are not verified twice."""
    identity = load_dataset(store, dataset)
    report = DatasetReport(dataset)
    check = _Check(store, dataset, session_ok)
    base = dataset_rel(dataset)
    known = {p.name for p in store.path(base).iterdir()}
    for name in sorted(known - {'dataset.json', 'labels', 'versions'}):
        check.add(DATASET_INCONSISTENT, f'{base}/{name}', 'not part of a dataset', 'WARNING')

    labels_root = store.path(labels_rel(dataset))
    if labels_root.is_dir():
        for directory in sorted(labels_root.iterdir()):
            rel = f'{labels_rel(dataset)}/{directory.name}'
            if not (directory / 'labelset.json').is_file():
                check.add(DATASET_INCONSISTENT, rel, 'a label set directory without labelset.json (an interrupted '
                          'creation); nothing references it', 'WARNING')
                continue
            if check.guard(rel, _verify_label_set, check, directory.name) is not None:
                report.label_sets.append(directory.name)

    versions_root = store.path(versions_rel(dataset))
    complete = list_versions(store, dataset)
    if versions_root.is_dir():
        for directory in sorted(versions_root.iterdir()):
            numbered = bool(VERSION_DIR_RE.match(directory.name))
            if numbered and int(directory.name) not in complete:
                check.add(DATASET_INCONSISTENT, f'{versions_rel(dataset)}/{directory.name}', 'a version directory '
                          'without manifest.json (an interrupted creation); its number is never reused', 'WARNING')
            elif not numbered:
                check.add(DATASET_INCONSISTENT, f'{versions_rel(dataset)}/{directory.name}', 'not a version', 'WARNING')
    for number in complete:
        rel = version_rel(dataset, number)
        if check.guard(rel, _verify_version, check, identity, number) is not None:
            report.versions.append(number)
    report.findings = check.findings
    return report


def _verify_label_set(check, ls_id):
    store, dataset = check.store, check.dataset
    rel = labels_rel(dataset, ls_id)
    label_set = load_label_set(store, dataset, ls_id)
    extra = sorted(p.name for p in store.path(rel).iterdir()) != ['entries.jsonl', 'labelset.json']
    if extra:
        check.add(DATASET_INCONSISTENT, rel, 'holds files other than labelset.json and entries.jsonl', 'WARNING')
    if label_set.supersedes is not None:
        check.guard(f'{rel}/labelset.json', load_label_set, store, dataset, label_set.supersedes)
    for problem in rederive_problems(store, label_set):
        check.add(LABEL_INVALID, f'{rel}/entries.jsonl', problem)
    for entry in read_entries(store, label_set):
        where = f'{rel}/entries.jsonl'
        try:
            check_entry_in_store(label_set.kind, entry, check.sessions, store, where)
        except DataError as exc:
            check.add(exc.code, where, exc.message, record=entry['observation_id'])
    return label_set


def _verify_version(check, identity, number):
    store, dataset = check.store, check.dataset
    rel = version_rel(dataset, number)
    version = load_version(store, dataset, number)
    names = sorted(p.name for p in store.path(rel).iterdir())
    if names != sorted([MANIFEST_FILE, SELECTION_FILE, SPLIT_FILE]):
        check.add(DATASET_INCONSISTENT, rel, f'holds {names}', 'WARNING')
    content = version.content

    # derivation
    parent = version.derivation.get('parent')
    parent_split = None
    if parent is not None:
        try:
            parent_version = load_version(store, dataset, parent['version'])
            if parent_version.content_hash != parent['content_hash']:
                check.add(DATASET_INCONSISTENT, f'{rel}/{MANIFEST_FILE}', f'parent version {parent["version"]} has '
                          f'content hash {parent_version.content_hash}, not the pinned {parent["content_hash"]}')
            parent_split = load_split(store, parent_version)
            for problem in derivation_problems(version, parent_version):
                check.add(DERIVATION_INVALID, f'{rel}/{MANIFEST_FILE}', problem)
        except DataError as exc:
            check.add(exc.code, f'{rel}/{MANIFEST_FILE}', f'parent version {parent["version"]}: {exc.message}')

    # sessions, calibrations, rights
    basis = [{'source': f'dataset:{dataset}', 'rights': identity.rights}]
    expected = []
    outcomes = set(content['selection']['filter']['include_outcomes'])
    excluded = {name: 0 for name in content['selection']['excluded']}
    for ref in content['sessions']:
        sid = ref['session_id']
        try:
            session = check.sessions.get(sid)
        except DataError as exc:
            check.add(exc.code, f'{rel}/{MANIFEST_FILE}', f'session {sid}: {exc.message}', record=sid)
            continue
        header = session.header
        if (header.index['sha256'], header.status, header.vocab['sha256']) != \
                (ref['index_sha256'], ref['status'], ref['vocab_sha256']):
            check.add(DATASET_INCONSISTENT, f'{rel}/{MANIFEST_FILE}', f'session {sid} is not the session the version '
                      'pins (index hash, status or vocabulary)', record=sid)
        clock = {'session_domain': header.clock.session_domain}
        if header.clock.sim_source is not None:
            clock['sim_source'] = header.clock.sim_source
        if ref['source'] != {'transport': header.source.transport, 'backend': header.source.backend} \
                or ref['clock'] != clock:
            check.add(DATASET_INCONSISTENT, f'{rel}/{MANIFEST_FILE}', f'session {sid}: the recorded source / clock '
                      'is not the session\'s', record=sid)
        if list(header.calibration_ids) != ref['calibration_ids']:
            check.add(DATASET_INCONSISTENT, f'{rel}/{MANIFEST_FILE}', f'session {sid} calibration ids differ',
                      record=sid)
        if not check.session_ok(sid):
            check.add(DATASET_INCONSISTENT, f'{rel}/{MANIFEST_FILE}', f'session {sid} does not verify (run verify '
                      f'{sid})', record=sid)
        basis.append({'source': sid, 'rights': dict(header.rights)})
        if version.group_by == DEFAULT_GROUP_BY and any(c.trigger.kind == TRIGGER_KIND and c.outcome in outcomes
                                                        for c in session.captures):
            check.add(DATASET_INCONSISTENT, f'{rel}/{MANIFEST_FILE}', f'session {sid} holds generated captures in a '
                      'version grouped by session: one randomized scene could sit in two splits', record=sid)
        chosen = [c.capture_id for c in session.captures if c.outcome in outcomes]
        if len(chosen) != ref['captures_selected']:
            check.add(DATASET_INCONSISTENT, f'{rel}/{MANIFEST_FILE}', f'session {sid}: {len(chosen)} captures pass '
                      f'the filter, the manifest records {ref["captures_selected"]}', record=sid)
        for c in session.captures:
            if c.outcome not in outcomes:
                continue
            line = {'capture_id': c.capture_id, 'outcome': c.outcome}
            if version.group_by != DEFAULT_GROUP_BY:
                group = check.guard(f'sessions/{sid}/captures.jsonl', group_of, c, version.group_by)
                line['group'] = group if group is not None else '<invalid>'
            expected.append(line)
        for c in session.captures:
            if c.outcome not in outcomes:
                excluded[c.outcome] += 1
        for cal_id in header.calibration_ids:
            pinned = content['calibrations'].get(cal_id)
            cal = check.guard(f'calibrations/{cal_id}.json', store.get_calibration, cal_id, session.vocab)
            if cal is not None and pinned is not None and cal.sha256 != pinned['sha256']:
                check.add(HASH_MISMATCH, f'calibrations/{cal_id}.json', 'not the calibration the version pins')
    # the rights themselves are their basis's combination by the manifest's own parse rule; here: the basis is still
    # the dataset's and the sessions' rights
    if basis != [dict(b) for b in version.rights_basis]:
        check.add(DATASET_INCONSISTENT, f'{rel}/{MANIFEST_FILE}', 'the rights basis is not the dataset\'s and the '
                  'sessions\' rights')

    if excluded != content['selection']['excluded']:
        check.add(DATASET_INCONSISTENT, f'{rel}/{MANIFEST_FILE}', f'selection.excluded is '
                  f'{content["selection"]["excluded"]}, the sessions give {excluded}')

    # selection
    selection = check.guard(f'{rel}/{SELECTION_FILE}', read_selection, store, version)
    if selection is not None:
        expected.sort(key=lambda x: capture_key(x['capture_id']))
        if selection != expected:
            check.add(DATASET_INCONSISTENT, f'{rel}/{SELECTION_FILE}', 'the selection is not exactly the pinned '
                      'sessions\' captures that pass the outcome filter, with their groups')
    selected = {x['capture_id'] for x in selection or ()}

    # label sets
    labelled = {}
    for ref in content['label_sets']:
        ls_id = ref['label_set_id']
        try:
            label_set = load_label_set(store, dataset, ls_id)
            if label_set.sha256 != ref['sha256']:
                check.add(HASH_MISMATCH, f'{labels_rel(dataset, ls_id)}/labelset.json', f'not the label set version '
                          f'{number} pins')
                continue
            inside = 0
            for entry in read_entries(store, label_set):
                obs = entry['observation_id']
                if capture_of(obs) not in selected:
                    continue
                inside += 1
                other = labelled.setdefault((label_set.kind, obs), ls_id)
                if other != ls_id:
                    check.add(LABEL_CONFLICT, f'{rel}/{MANIFEST_FILE}', f'{obs} is labelled by {other} and {ls_id}',
                              record=obs)
            if inside != ref['entries_selected']:
                check.add(DATASET_INCONSISTENT, f'{rel}/{MANIFEST_FILE}', f'{ls_id}: {inside} entries in the '
                          f'selection, the manifest records {ref["entries_selected"]}')
        except DataError as exc:
            check.add(exc.code, f'{rel}/{MANIFEST_FILE}', f'label set {ls_id}: {exc.message}')

    # split
    split = check.guard(f'{rel}/{SPLIT_FILE}', load_split, store, version)
    if split is not None:
        # load_split refused a split that does not assign exactly the version's groups
        if version.group_by == DEFAULT_GROUP_BY:
            groups = {x['session_id']: x['captures_selected'] for x in content['sessions']}
        else:
            groups = {}
            for line in selection or ():
                groups[group_of_line(line)] = groups.get(group_of_line(line), 0) + 1
        counts = {n: {'groups': 0, 'captures': 0} for n in ('train', 'val', 'test')}
        for group, name in split.assignment.items():
            counts[name]['groups'] += 1
            counts[name]['captures'] += groups.get(group, 0)
        if counts != split.record['counts'] or counts != content['split']['counts']:
            check.add(DATASET_INCONSISTENT, f'{rel}/{SPLIT_FILE}', 'split counts disagree with the selection')
        inherited = split.record.get('inherited')
        if (inherited is not None) != (version.derivation['op'] in ('ADD_SESSIONS', 'FILTER', 'RELABEL')):
            check.add(DATASET_INCONSISTENT, f'{rel}/{SPLIT_FILE}', f'a {version.derivation["op"]} version '
                      f'{"must" if inherited is None else "must not"} inherit its parent\'s split')
        if inherited is not None and parent_split is not None and inherited['split_id'] != parent_split.split_id:
            check.add(DATASET_INCONSISTENT, f'{rel}/{SPLIT_FILE}', 'inherits a split that is not its parent\'s')
        for problem in check_split_rules(split, parent_split if inherited is not None else None):
            check.add(DATASET_INCONSISTENT, f'{rel}/{SPLIT_FILE}', problem)
    return version


__all__ = ['verify_dataset', 'DatasetReport']
