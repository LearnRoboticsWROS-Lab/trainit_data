"""The ``trainit-data`` command line (D0.1: inspect, verify, recover; D0.6: dataset, version, labels, export,
verify --dataset, verify-export; D0.7: labels from-synthetic, version create --group-by, compare-generation).

The Python API (``Store``, ``verify_session``, ``recover_session``) is authoritative; the CLI is a thin
front end. ``--json`` prints ONE object of the stable machine schema ``trainit-cli/1``::

    {"schema": "trainit-cli/1", "command": <str>, "ok": <bool>, "trainit_data": <version>,
     "result": <object | null>, "error": {"code", "message", "context"[, "cause"]} | null}

``result`` keys per command are pinned by ``test/test_cli.py``. Exit status: 0 ok; 1 refused or findings
with errors; 2 usage error (argparse). Human output is for people and may change. ``command`` of a two-word
command is both words (``"dataset create"``). D0.6 evolves ``trainit-cli/1`` compatibly: new commands, and one new
key (``datasets``) in the result of a store ``verify``; no existing key changed. One consumer-visible change of
MEANING: a store ``verify`` is ``ok`` only if every dataset verifies too (so a store that used to pass can fail
because of a dataset). D0.7 evolves it compatibly again: two new commands (``labels from-synthetic``,
``compare-generation``) and one new option (``version create --group-by``); no existing key changed. D0.8 evolves it
compatibly again: two read-only ``inspect`` options for the TSA Data panel -- ``--detail`` (new keys in each store
session entry: ``opened_wall_utc_ns``, ``classification``, ``sensors``, ``outcomes``, ``synthetic``) and ``--captures``
(new result keys ``root``, ``captures``, ``captures_total``); without them the result is unchanged.
"""
import argparse
import json
import sys
from pathlib import Path

from . import __version__
from .dataset import create_dataset, list_datasets, load_dataset
from .dataset_integrity import verify_dataset
from .errors import INVALID_VALUE, WRITE_FAILED, DataError
from .exports import verify_export
from .formats import get_exporter
from .formats.yolo import import_yolo_labels
from .ground_truth import BOX_SOURCES, KINDS as SYNTHETIC_KINDS, create_synthetic_label_set
from .integrity import recover_session, verify_session, verify_store
from .labels import list_label_sets, load_label_set
from .reader import SessionReader
from .splits import GROUP_BY, SPLITS
from .store import Store
from .synthetic import compare_generations
from .versions import create_version, list_versions, load_version

CLI_SCHEMA = 'trainit-cli/1'


def _session_detail(header) -> dict:
    """``inspect --detail``: what a session table shows, from the header alone (D0.8)."""
    counts = header.counters['captures'] if header.counters else None
    return {'opened_wall_utc_ns': header.opened['wall_utc_ns'], 'classification': header.rights['classification'],
            'sensors': [d.sensor_id for d in header.sensors],
            'outcomes': None if counts is None else {k: counts[k] for k in ('complete', 'partial', 'failed')},
            'synthetic': 'randomization' in header.provenance}


def _inspect(store, args):
    if args.session is None:
        if args.captures:
            raise DataError(INVALID_VALUE, 'inspect --captures lists the captures of ONE session')
        sessions = []
        for session_id in store.list_sessions():
            try:
                header = SessionReader(store, session_id).header
                entry = {'session_id': session_id, 'status': header.status, 'transport': header.source.transport,
                         'backend': header.source.backend,
                         'captures': header.index['lines'] if header.index else None}
                if args.detail:
                    entry.update(_session_detail(header))
            except DataError as exc:
                entry = {'session_id': session_id, 'status': None, 'error': exc.code}
            sessions.append(entry)
        result = {'root': str(store.root), 'store_id': store.store_id, 'schemas': store.header['schemas'],
                  'sessions': sessions}
        return True, result
    if args.detail:
        raise DataError(INVALID_VALUE, 'inspect --detail summarises the store\'s sessions; give no session')
    if args.limit is not None and (not args.captures or args.limit < 0):
        raise DataError(INVALID_VALUE, '--limit is a count (>= 0) of inspect --captures')
    reader = store.open_session(args.session)
    header = reader.header
    result = {'session': header.to_dict()}
    outcomes = {'complete': 0, 'partial': 0, 'failed': 0}
    captures = []
    for capture in (reader.captures() if args.captures or not header.sealed else ()):
        outcomes[capture.outcome.lower()] += 1
        if args.captures and (args.limit is None or len(captures) < args.limit):
            captures.append(capture.to_dict())
    if not header.sealed:
        result['committed'] = outcomes
    if args.captures:
        # Read-only: the recorded capture lines, payloads named by their store-relative path under ``root``.
        result['root'] = str(store.root)
        result['captures'] = captures
        result['captures_total'] = sum(outcomes.values())
    return True, result


def _verify(store, args):
    if args.dataset and args.session:
        raise DataError(INVALID_VALUE, 'verify takes a session or --dataset, not both')
    if args.dataset:
        report = verify_dataset(store, args.dataset)
    else:
        report = verify_session(store, args.session) if args.session else verify_store(store)
    return report.ok, report.to_dict()


# ---- D0.6 ------------------------------------------------------------------------------------------------

def _dataset_create(store, args):
    rights = {'classification': args.classification}
    for key, value in (('owner_org', args.owner_org), ('license', args.license), ('notes', args.rights_notes)):
        if value is not None:
            rights[key] = value
    domain = {k: v for k, v in (('object', args.object), ('task', args.task), ('environment', args.environment))
              if v is not None}
    identity = create_dataset(store, args.dataset, name=args.name, rights=rights, description=args.description,
                              domain=domain)
    return True, {'dataset': identity.to_dict()}


def _dataset_show(store, args):
    identity = load_dataset(store, args.dataset)
    return True, {'dataset': identity.to_dict(), 'versions': list_versions(store, args.dataset),
                  'label_sets': list_label_sets(store, args.dataset)}


def _dataset_list(store, args):
    out = []
    for dataset_id in list_datasets(store):
        try:
            identity = load_dataset(store, dataset_id)
            out.append({'dataset_id': dataset_id, 'name': identity.name,
                        'classification': identity.rights['classification'],
                        'versions': list_versions(store, dataset_id)})
        except DataError as exc:
            out.append({'dataset_id': dataset_id, 'error': exc.code})
    return True, {'datasets': out}


def _assignments(values):
    out = {}
    for value in values or ():
        group, sep, split = value.partition('=')
        if not sep or split not in SPLITS or group in out:
            raise DataError(INVALID_VALUE, f'--assign {value!r}: expected GROUP=train|val|test, once per group')
        out[group] = split
    return out


def _version_create(store, args):
    ratios = None
    given = [args.train, args.val, args.test]
    if any(r is not None for r in given):
        if any(r is None for r in given):
            raise DataError(INVALID_VALUE, '--train, --val and --test are given together')
        ratios = {'train': args.train, 'val': args.val, 'test': args.test}
    version = create_version(store, args.dataset, op=args.op, parent=args.parent, sessions=args.session or (),
                             remove_sessions=args.remove_session or (), include_outcomes=args.include_outcome,
                             label_sets=args.label_set or (), drop_label_sets=args.drop_label_set or (),
                             seed=args.seed, ratios=ratios, overrides=_assignments(args.assign) or None,
                             version=args.number, group_by=args.group_by)
    return True, {'version': version.to_dict()}


def _version_show(store, args):
    return True, {'version': load_version(store, args.dataset, args.version).to_dict()}


def _labels_import_yolo(store, args):
    label_set = import_yolo_labels(store, args.dataset, args.directory, args.class_name, source=args.source,
                                   tool=args.tool, author=args.author, notes=args.notes, supersedes=args.supersedes)
    return True, {'label_set': label_set.to_dict()}


def _labels_from_synthetic(store, args):
    label_set = create_synthetic_label_set(store, args.dataset, args.session, args.kind, box_source=args.box_source,
                                           classes=args.class_name, image_stream=args.image_stream,
                                           mask_stream=args.mask_stream, supersedes=args.supersedes)
    return True, {'label_set': label_set.to_dict()}


def _compare_generation(store, args):
    report = compare_generations(store, args.session_a, args.session_b)
    return report['reproducible'], report


def _labels_show(store, args):
    if args.label_set is None:
        return True, {'label_sets': [load_label_set(store, args.dataset, i).to_dict()
                                     for i in list_label_sets(store, args.dataset)]}
    return True, {'label_set': load_label_set(store, args.dataset, args.label_set).to_dict()}


def _export_yolo(store, args):
    config = {'task': args.task, 'allow_mixed_provenance': args.allow_mixed_provenance}
    if args.label_set:
        config['label_sets'] = args.label_set
    sidecar = get_exporter('yolo').export(store, args.dataset, args.version, args.destination, config)
    return True, {'export': {'destination': str(Path(args.destination).expanduser().resolve()),
                             'export_id': sidecar['export_id'], 'exporter': sidecar['exporter'],
                             'source': sidecar['source'], 'rights': sidecar['rights'],
                             'summary': sidecar['summary'], 'files': len(sidecar['files'])}}


def _verify_export(store, args):
    report = verify_export(args.directory, store)
    return report.ok, report.to_dict()


def _recover(store, args):
    return True, recover_session(store, args.session)


def _human(command, ok, result, error):
    if error is not None:
        return f'{command}: REFUSED {error["code"]}: {error["message"]}'
    lines = []
    if command == 'inspect' and 'sessions' in result:
        lines.append(f'store {result["store_id"]} at {result["root"]}: {len(result["sessions"])} session(s)')
        lines += [f'  {e["session_id"]}  {e["status"]}  {e.get("transport", "")}/{e.get("backend", "")}  '
                  f'captures={e.get("captures")}'
                  + (f'  outcomes={e["outcomes"]}  {e["classification"]}{"  generated" if e["synthetic"] else ""}'
                     if 'outcomes' in e else '') for e in result['sessions']]
    elif command == 'inspect':
        h = result['session']
        lines.append(f'{h["session_id"]}  {h["status"]}  {h["source"]["transport"]}/{h["source"]["backend"]}  '
                     f'clock={h["clock"]["session_domain"]}')
        lines.append(f'  sensors: {", ".join(d["sensor_id"] for d in h["sensors"])}; calibrations: '
                     f'{", ".join(h["calibration_ids"]) or "-"}')
        if 'counters' in h:
            lines.append(f'  captures: {h["counters"]["captures"]}  index lines: {h["index"]["lines"]}')
        else:
            lines.append(f'  ACTIVE, committed so far: {result["committed"]}')
        if 'captures' in result:
            lines.append(f'  showing {len(result["captures"])} of {result["captures_total"]} capture(s):')
            lines += [f'    {c["capture_id"]}  {c["outcome"]}  {c["trigger"]["trigger_id"]}  '
                      + ', '.join(o['payload']['path'] for o in c['observations'] if 'payload' in o)
                      for c in result['captures']]
    elif command == 'verify' and 'dataset_id' in result:
        lines.append(f'dataset {result["dataset_id"]}  {"OK" if result["ok"] else "FAILED"}  '
                     f'versions={result["versions"]}  label sets={len(result["label_sets"])}')
        lines += [f'  {f["severity"]} {f["code"]} {f["artifact"]}: {f["detail"]}' for f in result['findings']]
        lines.append('verify: OK' if ok else 'verify: FAILED')
    elif command == 'verify':
        reports = result.get('sessions', [result])
        for r in reports:
            lines.append(f'{r["session_id"]}  {r["status"]}  {"OK" if r["ok"] else "FAILED"}  '
                         f'captures={r["captures"]} observations={r["observations"]}')
            lines += [f'  {f["severity"]} {f["code"]} {f["artifact"]}'
                      f'{" [" + f["record"] + "]" if "record" in f else ""}: {f["detail"]}' for f in r['findings']]
        lines += [f'  {f["severity"]} {f["code"]} {f["artifact"]}: {f["detail"]}' for f in result.get('findings', [])
                  if 'sessions' in result]
        for d in result.get('datasets', []):
            lines.append(f'dataset {d["dataset_id"]}  {"OK" if d["ok"] else "FAILED"}  versions={d["versions"]}')
            lines += [f'  {f["severity"]} {f["code"]} {f["artifact"]}: {f["detail"]}' for f in d['findings']]
        lines.append('verify: OK' if ok else 'verify: FAILED')
    elif command == 'recover':
        lines.append(f'{result["session_id"]} sealed INTERRUPTED: {result["committed_captures"]} committed capture(s)')
        if 'torn_tail' in result:
            tail = result['torn_tail']
            lines.append(f'  torn index tail ({tail["bytes"]} B) kept as {tail["saved_as"]}')
        lines += [f'  moved {q["from"]} -> {q["to"]}' for q in result['quarantined']]
        lines.append('  nothing was deleted')
    elif command in ('dataset create', 'dataset show'):
        d = result['dataset']
        lines.append(f'{d["dataset_id"]}  "{d["name"]}"  rights={d["rights"]["classification"]}')
        if 'versions' in result:
            lines.append(f'  versions: {result["versions"] or "-"}; '
                         f'label sets: {", ".join(result["label_sets"]) or "-"}')
    elif command == 'dataset list':
        lines += [f'{d["dataset_id"]}  {d.get("classification", d.get("error"))}  versions={d.get("versions")}'
                  for d in result['datasets']] or ['no dataset']
    elif command in ('version create', 'version show'):
        v = result['version']
        c = v['content']
        lines.append(f'{v["dataset_id"]} version {v["version"]}  {v["derivation"]["op"]}  {v["content_kind"]}  '
                     f'rights={v["rights"]["classification"]}  {v["content_hash"]}')
        lines.append(f'  sessions: {", ".join(x["session_id"] for x in c["sessions"])}')
        outcomes = '+'.join(c['selection']['filter']['include_outcomes'])
        lines.append(f'  captures: {c["selection"]["captures"]} ({outcomes}; excluded {c["selection"]["excluded"]})')
        lines.append(f'  label sets: {", ".join(x["label_set_id"] for x in c["label_sets"]) or "-"}')
        lines.append(f'  split {c["split"]["split_id"]}: ' + ', '.join(
            f'{k} {n["groups"]} group(s) / {n["captures"]} capture(s)' for k, n in c['split']['counts'].items()))
    elif command in ('labels import-yolo', 'labels from-synthetic', 'labels show') and 'label_set' in result:
        ls = result['label_set']
        lines.append(f'{ls["label_set_id"]}  {ls["kind"]}  {ls["provenance"]["source"]}  '
                     f'entries={ls["entries"]["count"]}  classes={ls["ontology"]["classes"]}')
    elif command == 'labels show':
        lines += [f'{ls["label_set_id"]}  {ls["kind"]}  {ls["provenance"]["source"]}  entries={ls["entries"]["count"]}'
                  for ls in result['label_sets']] or ['no label set']
    elif command == 'export yolo':
        e = result['export']
        lines.append(f'{e["export_id"]} -> {e["destination"]}  ({e["files"]} files, rights '
                     f'{e["rights"]["classification"]})')
        lines += [f'  {k}: {n["images"]} image(s), {n["boxes"]} box(es)' for k, n in e['summary']['splits'].items()]
    elif command == 'compare-generation':
        lines.append(f'{"REPRODUCIBLE" if result["reproducible"] else "NOT REPRODUCIBLE"} (claim SAME_DRAWS): '
                     f'{result["frames"]["a"]} / {result["frames"]["b"]} frames, {result["runs"]} run(s)')
        lines += [f'  {"=" if same else "!"} {name}' for name, same in result['checks'].items()]
        render = result['render']
        lines.append(f'  rendered bytes identical: {render["bit_identical"]} (not promised) {render["by_stream"]}')
        lines.append(f'  same environment: {result["environment"]["same"]}')
    elif command == 'verify-export':
        lines.append(f'{result["destination"]}  {result["export_id"]}  files={result["files"]}')
        lines += [f'  {f["severity"]} {f["code"]} {f["artifact"]}: {f["detail"]}' for f in result['findings']]
        lines.append('verify-export: OK' if ok else 'verify-export: FAILED')
    return '\n'.join(lines)


_HANDLERS = {'inspect': _inspect, 'verify': _verify, 'recover': _recover, 'dataset create': _dataset_create,
             'dataset show': _dataset_show, 'dataset list': _dataset_list, 'version create': _version_create,
             'version show': _version_show, 'labels import-yolo': _labels_import_yolo, 'labels show': _labels_show,
             'export yolo': _export_yolo, 'verify-export': _verify_export,
             'labels from-synthetic': _labels_from_synthetic, 'compare-generation': _compare_generation}


def build_parser():
    parser = argparse.ArgumentParser(prog='trainit-data', description='TrainIt Data Foundation: the canonical '
                                     'data store (inspect, verify, recover; datasets, versions, labels, exports).')
    parser.add_argument('--root', help='the data store (default: $TRAINIT_DATA_ROOT, else ~/TrainItData)')
    parser.add_argument('--json', action='store_true', help=f'print one {CLI_SCHEMA} JSON object')
    parser.add_argument('--version', action='version', version=f'trainit-data {__version__}')
    sub = parser.add_subparsers(dest='command', required=True)
    p = sub.add_parser('inspect', help='summarise the store, or one session (read-only)')
    p.add_argument('session', nargs='?')
    p.add_argument('--detail', action='store_true', help='store summary: also each session\'s opening time, rights, '
                   'sensors, outcomes and whether it was generated')
    p.add_argument('--captures', action='store_true', help='one session: also its capture records, in order')
    p.add_argument('--limit', type=int, help='--captures: at most this many (the first ones)')
    p = sub.add_parser('verify', help='check every hash and record of the store, of one session or of one dataset '
                       '(read-only)')
    p.add_argument('session', nargs='?')
    p.add_argument('--dataset', help='verify this dataset (identity, label sets, versions)')
    p = sub.add_parser('recover', help='seal an interrupted ACTIVE session INTERRUPTED (moves, never deletes)')
    p.add_argument('session')

    p = sub.add_parser('dataset', help='dataset identities').add_subparsers(dest='action', required=True)
    c = p.add_parser('create', help='create a dataset identity (immutable)')
    c.add_argument('dataset')
    c.add_argument('--name', required=True)
    c.add_argument('--description')
    c.add_argument('--object')
    c.add_argument('--task')
    c.add_argument('--environment')
    c.add_argument('--classification', default='INTERNAL',
                   choices=['INTERNAL', 'CUSTOMER_RESTRICTED', 'PUBLIC_APPROVED'],
                   help='default INTERNAL; PUBLIC_APPROVED only as the owner\'s explicit statement')
    c.add_argument('--owner-org')
    c.add_argument('--license')
    c.add_argument('--rights-notes')
    c = p.add_parser('show', help='a dataset, its versions and label sets')
    c.add_argument('dataset')
    p.add_parser('list', help='every dataset of the store')

    p = sub.add_parser('version', help='immutable dataset versions').add_subparsers(dest='action', required=True)
    c = p.add_parser('create', help='build a new immutable version (CREATE, or derived from --from)')
    c.add_argument('dataset')
    c.add_argument('--op', default='CREATE', choices=['CREATE', 'ADD_SESSIONS', 'FILTER', 'RELABEL', 'RESPLIT'])
    c.add_argument('--from', dest='parent', type=int, help='the parent version (every op but CREATE)')
    c.add_argument('--session', action='append', help='a sealed session to include (CREATE) or add (ADD_SESSIONS)')
    c.add_argument('--remove-session', action='append', help='FILTER: a session to leave out')
    c.add_argument('--include-outcome', action='append', choices=['COMPLETE', 'PARTIAL', 'FAILED'],
                   help='capture outcomes to select (CREATE: default COMPLETE; FILTER: a narrower set)')
    c.add_argument('--label-set', action='append', help='a label set to include (CREATE, ADD_SESSIONS, RELABEL)')
    c.add_argument('--drop-label-set', action='append', help='RELABEL: a label set to leave out')
    c.add_argument('--seed', type=int, help='split seed (CREATE: default 0; RESPLIT)')
    c.add_argument('--train', type=float)
    c.add_argument('--val', type=float)
    c.add_argument('--test', type=float)
    c.add_argument('--assign', action='append', metavar='GROUP=SPLIT',
                   help='pin a group (a session, or a randomization run) to a split (recorded); a RESPLIT states all '
                   'its pins again, it inherits none')
    c.add_argument('--group-by', choices=sorted(GROUP_BY),
                   help='what a split group is (CREATE: default session; RESPLIT): randomization_run groups a '
                   'generated capture by its randomization run and any other capture by its session')
    c.add_argument('--number', type=int, help='the version number (default: the next free one)')
    c = p.add_parser('show', help='one version\'s manifest')
    c.add_argument('dataset')
    c.add_argument('version', type=int)

    p = sub.add_parser('labels', help='label sets (immutable, content-addressed)')
    p = p.add_subparsers(dest='action', required=True)
    c = p.add_parser('import-yolo', help='import YOLO detection label files (<stem>.txt) as one label set')
    c.add_argument('dataset')
    c.add_argument('directory')
    c.add_argument('--class', dest='class_name', action='append', required=True,
                   help='a class name, in YOLO index order (repeat: --class a --class b)')
    c.add_argument('--source', default='IMPORTED', choices=['IMPORTED', 'HUMAN'])
    c.add_argument('--tool')
    c.add_argument('--author')
    c.add_argument('--notes')
    c.add_argument('--supersedes', help='the label set this one corrects')
    c = p.add_parser('from-synthetic', help='derive a SYNTHETIC_GROUND_TRUTH label set from the annotator output '
                     'recorded in synthetic sessions')
    c.add_argument('dataset')
    c.add_argument('--session', action='append', required=True, help='a sealed synthetic session (repeat)')
    c.add_argument('--kind', required=True, choices=list(SYNTHETIC_KINDS))
    c.add_argument('--box-source', default='ANNOTATOR', choices=sorted(BOX_SOURCES),
                   help='BBOX_2D: the annotator\'s tight boxes (default) or the masks\' extents (recorded as derived)')
    c.add_argument('--class', dest='class_name', action='append',
                   help='the ontology in order (repeat; default: the sorted classes seen); must include every class')
    c.add_argument('--image-stream', default='color')
    c.add_argument('--mask-stream', default='instances')
    c.add_argument('--supersedes', help='the label set this one corrects')
    c = p.add_parser('show', help='every label set of a dataset, or one')
    c.add_argument('dataset')
    c.add_argument('label_set', nargs='?')

    p = sub.add_parser('export', help='export a dataset version').add_subparsers(dest='action', required=True)
    c = p.add_parser('yolo', help='YOLO detection export (segmentation is deferred)')
    c.add_argument('dataset')
    c.add_argument('version', type=int)
    c.add_argument('destination', help='a missing or empty directory outside any repository and the store')
    c.add_argument('--label-set', action='append', help='BBOX_2D label sets to export (default: all of the version)')
    c.add_argument('--task', default='detect', choices=['detect', 'segment'])
    c.add_argument('--allow-mixed-provenance', action='store_true')

    p = sub.add_parser('verify-export', help='check an export against its sidecar and the store (read-only)')
    p.add_argument('directory')
    p = sub.add_parser('compare-generation', help='D0-G: do two synthetic sessions hold the same recorded generation '
                       '(runs, ordering, draws, semantic assignments, calibrations)? (read-only)')
    p.add_argument('session_a')
    p.add_argument('session_b')
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    command = args.command if getattr(args, 'action', None) is None else f'{args.command} {args.action}'
    handler = _HANDLERS[command]
    ok, result, error = False, None, None
    try:
        ok, result = handler(Store.open(args.root), args)
    except DataError as exc:
        error = exc.to_dict()
    except OSError as exc:              # an I/O failure the core did not classify: still one structured refusal
        error = DataError(WRITE_FAILED, 'an I/O operation failed', cause=exc).to_dict()
    if args.json:
        print(json.dumps({'schema': CLI_SCHEMA, 'command': command, 'ok': ok, 'trainit_data': __version__,
                          'result': result, 'error': error}, sort_keys=True, indent=2))
    else:
        print(_human(command, ok, result, error), file=sys.stdout if error is None else sys.stderr)
    return 0 if ok and error is None else 1


if __name__ == '__main__':      # pragma: no cover
    sys.exit(main())
