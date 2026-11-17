"""Export artifacts: one exporter run over one dataset version (PUBLIC, D0.6).

An export is a DERIVED artifact in an external format, written where the user says, never part of the store's
canonical content: it can be deleted and recreated. Its sidecar ``trainit_export.json`` (``trainit-export/1.0``)
says exactly what it is::

    {schema, export_id, exporter {name, version, config, config_sha256}, source {dataset_id, version,
     content_kind, content_hash, label_sets [{label_set_id, sha256, source}], split {split_id, sha256}},
     rights, summary, files [{path, sha256, bytes, observation_id?, label_of?}], software {trainit_data}}

A file with ``observation_id`` IS that observation's payload, byte for byte; a file with ``label_of`` holds the labels
of that observation.

``export_id`` = ``"exp-"`` + 16 hex of the sha256 of (exporter name, version, config, source): the same version
exported with the same configuration has the same id. The sidecar carries no timestamp and no machine path, so
two runs produce byte-identical trees. An export is written into a sibling working directory and renamed into place
when complete, so a destination holds a whole export or nothing.

This module is format-neutral: the destination rules, the export plan and its streamed writer, the sidecar and
``verify_export``. The formats themselves (YOLO detection in D0.6, YOLO segmentation in M4.5) live in
``trainit_data.formats``, register there by name, and only PLAN an export (``plan(store, dataset, version, config)
-> ExportPlan``): what to write is decided by the format, how it is written and checked is decided here, once.
"""
import os
import re
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path

from . import __version__
from . import _strict as s
from ._canonical import parse_json, pretty_json, sha256_bytes, sha256_file, sha256_json
from ._fsio import fsync_dir, makedirs_durable, write_new_file
from .dataset import dataset_id as dataset_id_check, rights as rights_check
from .errors import (DATASET_INCONSISTENT, EXPORT_DESTINATION_INVALID, HASH_MISMATCH, INVALID_VALUE,
                     PAYLOAD_MISSING, WRITE_FAILED, DataError)
from ._sealed import observation_id
from .labels import label_set_id
from .splits import split_id
from .store import enclosing_repository

SCHEMA_NAME = 'trainit-export'
SCHEMA_MAJOR = 1
SCHEMA = 'trainit-export/1.0'
SIDECAR = 'trainit_export.json'

export_id = s.matching(re.compile(r'^exp-[0-9a-f]{16}$'), 'export id (exp-<16 hex>)')


def _exporter(value, where):
    f = s.Fields(value, where)
    out = {'name': f.req('name', s.ident), 'version': f.req('version', s.string),
           'config': f.req('config', s.opaque), 'config_sha256': f.req('config_sha256', s.sha256)}
    f.done()
    if out['config_sha256'] != sha256_json(out['config']):
        s.fail(INVALID_VALUE, 'config_sha256 is not the sha256 of the config', where)
    return out


def _label_ref(value, where):
    f = s.Fields(value, where)
    out = {'label_set_id': f.req('label_set_id', label_set_id), 'sha256': f.req('sha256', s.sha256),
           'source': f.req('source', s.token)}
    f.done()
    return out


def _split_ref(value, where):
    f = s.Fields(value, where)
    out = {'split_id': f.req('split_id', split_id), 'sha256': f.req('sha256', s.sha256)}
    f.done()
    return out


def _source(value, where):
    f = s.Fields(value, where)
    out = {'dataset_id': f.req('dataset_id', dataset_id_check), 'version': f.req('version', s.integer(1)),
           'content_kind': f.req('content_kind', s.token), 'content_hash': f.req('content_hash', s.sha256),
           'label_sets': [dict(x) for x in f.req('label_sets', s.list_of(_label_ref))],
           'split': f.req('split', _split_ref)}
    f.done()
    return out


def _file(value, where):
    f = s.Fields(value, where)
    out = {'path': f.req('path', s.relative_path), 'sha256': f.req('sha256', s.sha256),
           'bytes': f.req('bytes', s.integer(0))}
    s.put(out, 'observation_id', f.opt('observation_id', observation_id))
    s.put(out, 'label_of', f.opt('label_of', observation_id))
    f.done()
    if 'observation_id' in out and 'label_of' in out:
        s.fail(INVALID_VALUE, 'a file is a payload or labels, not both', where)
    return out


def _software(value, where):
    f = s.Fields(value, where)
    out = {'trainit_data': f.req('trainit_data', s.string)}
    f.done()
    return out


def parse_sidecar(value, where=SIDECAR) -> dict:
    f = s.Fields(value, where)
    out = {'schema': f.req('schema', s.schema(SCHEMA_NAME, SCHEMA_MAJOR)),
           'export_id': f.req('export_id', export_id),
           'exporter': f.req('exporter', _exporter),
           'source': f.req('source', _source),
           'rights': f.req('rights', rights_check),
           'summary': f.req('summary', s.opaque),
           'files': [dict(x) for x in f.req('files', s.list_of(_file))],
           'software': f.req('software', _software)}
    f.done()
    if out['export_id'] != export_id_of(out['exporter'], out['source']):
        s.fail(INVALID_VALUE, f'{out["export_id"]} is not the id of this exporter and source', where)
    paths = [x['path'] for x in out['files']]
    if paths != sorted(paths) or len(set(paths)) != len(paths) or SIDECAR in paths:
        s.fail(INVALID_VALUE, 'files are unique, sorted by path, and never list the sidecar', where)
    return out


def export_id_of(exporter: dict, source: dict) -> str:
    identity = {'exporter': {k: exporter[k] for k in ('name', 'version', 'config')}, 'source': source}
    return 'exp-' + sha256_json(identity)[len('sha256:'):][:16]


def prepare_destination(destination, store) -> Path:
    """A missing or EMPTY directory outside every source repository and outside the store
    (``EXPORT_DESTINATION_INVALID``): an export never overwrites, never lands in git, never mixes with the store."""
    dest = Path(destination).expanduser().resolve()
    repository = enclosing_repository(dest)
    if repository is not None:
        raise DataError(EXPORT_DESTINATION_INVALID, f'{dest} is inside the source repository {repository}: exported '
                        'data never lives in a software repository',
                        context={'destination': str(dest), 'repository': str(repository)})
    root = store.root.resolve()
    if dest == root or root in dest.parents:
        raise DataError(EXPORT_DESTINATION_INVALID, f'{dest} is inside the data store {root}: an export is not store '
                        'content', context={'destination': str(dest)})
    if dest.exists() and (not dest.is_dir() or any(dest.iterdir())):
        raise DataError(EXPORT_DESTINATION_INVALID, f'{dest} exists and is not an empty directory: an export never '
                        'overwrites', context={'destination': str(dest)})
    return dest


@dataclass(frozen=True)
class PlannedFile:
    """One file of an export, decided before anything is written: either generated ``data`` or the store file
    ``source`` copied byte for byte, which must hash to ``sha256`` (checked while it is copied)."""
    path: str
    sha256: str
    bytes: int
    data: bytes = None
    source: str = None                   # store-relative path (a payload)
    link: dict = None                    # {'observation_id': id} (the file IS that payload) | {'label_of': id}

    @classmethod
    def generated(cls, path: str, data: bytes, link: dict = None) -> 'PlannedFile':
        return cls(path, sha256_bytes(data), len(data), data=data, link=link)


@dataclass(frozen=True)
class ExportPlan:
    exporter: dict                       # {name, version, config}
    source: dict
    rights: dict
    summary: dict
    files: tuple                         # PlannedFile, any order

    def sidecar(self) -> dict:
        return build_sidecar(exporter=self.exporter, source=self.source, rights=self.rights, summary=self.summary,
                             files=[_listed(f) for f in self.files])


def _listed(planned: PlannedFile) -> dict:
    entry = {'path': planned.path, 'sha256': planned.sha256, 'bytes': planned.bytes}
    entry.update(planned.link or {})
    return entry


def write_export(store, destination, plan: ExportPlan) -> dict:
    """Write ``plan`` to ``destination`` (see :func:`prepare_destination`) and return the sidecar.

    The files are STREAMED one at a time (a payload is read, checked against its sha256 and written; nothing is
    held for the whole export) into a sibling ``.<name>.partial-<hex>`` directory, the sidecar last, and that
    directory is then renamed onto the destination: an export appears complete or not at all. On a failure the
    partial directory -- this call's own temporary -- is removed and the destination stays as it was."""
    dest = prepare_destination(destination, store)
    sidecar = plan.sidecar()
    try:
        makedirs_durable(dest.parent)
        tmp = Path(tempfile.mkdtemp(prefix=f'.{dest.name}.partial-', dir=dest.parent))
    except OSError as exc:
        raise DataError(WRITE_FAILED, f'cannot create a working directory beside {dest}', cause=exc) from None
    try:
        for planned in sorted(plan.files, key=lambda f: f.path):
            data = planned.data if planned.data is not None else _read_source(store, planned)
            path = tmp.joinpath(*planned.path.split('/'))
            makedirs_durable(path.parent)
            write_new_file(path, data)
        write_new_file(tmp / SIDECAR, pretty_json(sidecar))
        os.rename(tmp, dest)                       # onto a missing or EMPTY directory only
        fsync_dir(dest.parent)
    except BaseException as exc:
        shutil.rmtree(tmp, ignore_errors=True)
        if isinstance(exc, OSError):
            raise DataError(WRITE_FAILED, f'writing the export to {dest} failed; nothing was left there', cause=exc,
                            context={'destination': str(dest)}) from None
        raise
    return sidecar


def _read_source(store, planned: PlannedFile) -> bytes:
    path = store.path(planned.source)
    if not path.is_file():
        raise DataError(PAYLOAD_MISSING, f'{planned.source} is missing', context={'path': planned.source})
    data = path.read_bytes()
    if sha256_bytes(data) != planned.sha256 or len(data) != planned.bytes:
        raise DataError(HASH_MISMATCH, f'{planned.source} does not match its recorded sha256',
                        context={'path': planned.source})
    return data


def build_sidecar(*, exporter: dict, source: dict, rights: dict, summary: dict, files: list) -> dict:
    """``files``: ``[{path, sha256, bytes, observation_id? | label_of?}]``, any order."""
    exporter = dict(exporter, config_sha256=sha256_json(exporter['config']))
    sidecar = {'schema': SCHEMA, 'export_id': export_id_of(exporter, source), 'exporter': exporter,
               'source': source, 'rights': dict(rights), 'summary': summary,
               'files': sorted((dict(f) for f in files), key=lambda f: f['path']),
               'software': {'trainit_data': __version__}}
    parse_sidecar(sidecar, 'sidecar')                                 # one validation path
    return sidecar


def read_sidecar(dest) -> dict:
    dest = Path(dest)
    path = dest / SIDECAR
    if not path.is_file():
        raise DataError(PAYLOAD_MISSING, f'{dest} has no {SIDECAR}: not a TrainIt export, or an incomplete one',
                        context={'destination': str(dest)})
    return parse_sidecar(parse_json(path.read_bytes(), SIDECAR))


def verify_export(dest, store=None):
    """Inspect an export; never modifies anything. Returns an :class:`ExportReport`.

    Always: the sidecar parses, every listed file exists with its sha256 and size, nothing unlisted is present,
    and the format's own checks pass. With ``store``: the exporter PLANS the same export again from the store
    (the source version, its label sets and split, the recorded config) and every file must be the one planned --
    the same paths, the same label bytes, each image the payload of its observation, the same data.yaml."""
    from .formats import get_exporter
    from .integrity import Finding
    dest = Path(dest).expanduser().resolve()
    report = ExportReport(str(dest))
    try:
        sidecar = read_sidecar(dest)
    except DataError as exc:
        report.findings.append(Finding(exc.code, 'ERROR', SIDECAR, exc.message))
        return report
    report.export_id = sidecar['export_id']
    listed = {x['path']: x for x in sidecar['files']}
    present = {p.relative_to(dest).as_posix() for p in dest.rglob('*') if p.is_file() or p.is_symlink()}
    for rel in sorted(present - set(listed) - {SIDECAR}):
        report.findings.append(Finding(DATASET_INCONSISTENT, 'ERROR', rel, 'a file the sidecar does not list'))
    for rel, entry in listed.items():
        path = dest.joinpath(*rel.split('/'))
        if path.is_symlink() or not path.is_file():
            report.findings.append(Finding(PAYLOAD_MISSING, 'ERROR', rel, 'listed in the sidecar, missing'))
            continue
        actual, size = sha256_file(path)
        report.files += 1
        if (actual, size) != (entry['sha256'], entry['bytes']):
            report.findings.append(Finding(HASH_MISMATCH, 'ERROR', rel, f'sha256 {actual} ({size} B), the sidecar '
                                           f'records {entry["sha256"]} ({entry["bytes"]} B)'))
    try:
        exporter = get_exporter(sidecar['exporter']['name'])
    except DataError as exc:
        report.findings.append(Finding(exc.code, 'ERROR', SIDECAR, exc.message))
        return report
    report.findings.extend(_guarded(SIDECAR, exporter.verify, dest, sidecar))
    if store is not None:
        report.findings.extend(_guarded(SIDECAR, _verify_against_store, store, exporter, sidecar))
    return report


def _guarded(artifact, fn, *args) -> list:
    """A check over untrusted files reports, it never crashes: an unexpected failure is a finding."""
    from .integrity import Finding
    try:
        return fn(*args)
    except DataError as exc:
        return [Finding(exc.code, 'ERROR', artifact, exc.message)]
    except (KeyError, TypeError, ValueError, AttributeError, OSError) as exc:
        return [Finding(DATASET_INCONSISTENT, 'ERROR', artifact, f'{type(exc).__name__}: {exc}')]


def _verify_against_store(store, exporter, sidecar) -> list:
    from .integrity import Finding
    src = sidecar['source']
    if sidecar['exporter']['version'] != exporter.version:
        return [Finding(DATASET_INCONSISTENT, 'WARNING', SIDECAR, f'made by {exporter.name} exporter version '
                        f'{sidecar["exporter"]["version"]}, this core has {exporter.version}: the export cannot be '
                        'regenerated to compare it with the store')]
    plan = exporter.plan(store, src['dataset_id'], src['version'], sidecar['exporter']['config'])
    expected = plan.sidecar()
    findings = []
    for key in ('export_id', 'exporter', 'source', 'rights', 'summary'):
        if expected[key] != sidecar[key]:
            findings.append(Finding(DATASET_INCONSISTENT, 'ERROR', SIDECAR, f'{key} is not what the store gives for '
                                    f'{src["dataset_id"]} version {src["version"]}'))
    want = {f['path']: f for f in expected['files']}
    have = {f['path']: f for f in sidecar['files']}
    for path in sorted(set(want) | set(have)):
        if want.get(path) != have.get(path):
            detail = ('not in the export the store gives' if path not in want else
                      'missing from the export' if path not in have else
                      'not the file the store gives (content, payload or observation differ)')
            findings.append(Finding(DATASET_INCONSISTENT, 'ERROR', path, detail))
    return findings


class ExportReport:
    def __init__(self, destination: str):
        self.destination = destination
        self.export_id = None
        self.files = 0
        self.findings = []

    @property
    def ok(self) -> bool:
        return not any(f.severity == 'ERROR' for f in self.findings)

    def to_dict(self) -> dict:
        return {'destination': self.destination, 'export_id': self.export_id, 'ok': self.ok, 'files': self.files,
                'findings': [f.to_dict() for f in self.findings]}


__all__ = ['SIDECAR', 'verify_export', 'read_sidecar', 'parse_sidecar', 'prepare_destination', 'write_export',
           'ExportPlan', 'PlannedFile', 'ExportReport']
