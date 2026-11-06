"""The local data store (PUBLIC).

Default root ``~/TrainItData``, override ``TRAINIT_DATA_ROOT`` (ADR-0029 Decision 5, D0_DESIGN.md §12).
The store never lives inside a source repository: a root at or under a directory holding ``.git`` (a work
tree, a worktree or a submodule) is refused with ``STORE_INSIDE_REPOSITORY``. Nothing else is refused
for its location -- a directory under the user's home is fine.

Layout (paths inside records are relative to the root)::

    store.json                      trainit-store/1.0: store id, the record schemas this core writes
    vocab/vocab-<16 hex>.json       every vocabulary a session pinned (content-addressed)
    calibrations/cal-<16 hex>.json  immutable, content-addressed calibration records
    sessions/ses-<utc>-<8 hex>/     one directory per session (see ``writer``)
    datasets/<dataset_id>/          D0.6: identity, label sets, immutable versions (see ``versions``)

A store is created explicitly (``Store.open(create=True)``, which the writer's first session uses) and only
in a missing or EMPTY directory: an existing directory with foreign content is never adopted.
"""
import datetime
import os
import re
import secrets
import time
from pathlib import Path

from . import _strict as s
from ._canonical import parse_json, pretty_json
from ._fsio import fsync_dir, write_new_file
from .calibration import CalibrationRecord
from .errors import (CALIBRATION_CONFLICT, CALIBRATION_NOT_FOUND, HASH_MISMATCH, MALFORMED_RECORD,
                     SESSION_NOT_FOUND, STORE_INSIDE_REPOSITORY, STORE_INVALID, STORE_NOT_INITIALIZED, DataError)
from .vocab import Vocabulary

ENV_VAR = 'TRAINIT_DATA_ROOT'
DEFAULT_DIRNAME = 'TrainItData'
STORE_SCHEMA = 'trainit-store/1.0'
_STORE_ID_RE = re.compile(r'^store-[0-9a-f]{8}$')
WRITTEN_SCHEMAS = ('trainit-vocab/1.0', 'trainit-calibration/1.0', 'trainit-capture/1.0', 'trainit-session/1.1',
                   'trainit-recovery/1.0', 'trainit-dataset-identity/1.0', 'trainit-labels/1.0', 'trainit-labels/1.1',
                   'trainit-split/1.0', 'trainit-split/1.1', 'trainit-dataset/1.0', 'trainit-dataset/1.1',
                   'trainit-randomization/1.0', 'trainit-synthetic-scene/1.0')


def resolve_store_root(environ=None) -> Path:
    """``$TRAINIT_DATA_ROOT`` when set and non-empty, else ``~/TrainItData``; absolute, symlinks resolved."""
    environ = os.environ if environ is None else environ
    value = environ.get(ENV_VAR, '').strip()
    root = Path(value).expanduser() if value else Path.home() / DEFAULT_DIRNAME
    return root.resolve()


def enclosing_repository(path) -> Path:
    """The nearest directory at or above ``path`` that holds ``.git`` (dir or file), else ``None``."""
    path = Path(path).resolve()
    for candidate in (path, *path.parents):
        if (candidate / '.git').exists():
            return candidate
    return None


def _store_header(value, where='store.json'):
    f = s.Fields(value, where)
    header = {'schema': f.req('schema', s.schema('trainit-store', 1)),
              'store_id': f.req('store_id', s.matching(_STORE_ID_RE, 'store id')),
              'created': f.req('created', s.string),
              'schemas': list(f.req('schemas', s.list_of(s.string, unique=True)))}
    extensions = f.extensions()
    if extensions:
        header['extensions'] = extensions
    f.done()
    return header


def utc_iso(wall_ns: int = None) -> str:
    seconds = (time.time_ns() if wall_ns is None else wall_ns) // 1_000_000_000
    return datetime.datetime.fromtimestamp(seconds, datetime.timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')


class Store:
    """An opened data store. Obtain one with :meth:`Store.open`."""

    def __init__(self, root: Path, header: dict):
        self.root = root
        self.header = header

    @property
    def store_id(self) -> str:
        return self.header['store_id']

    @classmethod
    def open(cls, root=None, *, create: bool = False) -> 'Store':
        root = resolve_store_root() if root is None else Path(root).expanduser().resolve()
        repository = enclosing_repository(root)
        if repository is not None:
            raise DataError(STORE_INSIDE_REPOSITORY, f'the data store {root} is inside the source repository '
                            f'{repository}; data never lives in a software repository (set {ENV_VAR} elsewhere)',
                            context={'root': str(root), 'repository': str(repository)})
        header_path = root / 'store.json'
        if root.exists() and not root.is_dir():
            raise DataError(STORE_INVALID, f'{root} is not a directory', context={'root': str(root)})
        if header_path.is_file():
            try:
                header = _store_header(parse_json(header_path.read_bytes(), 'store.json'))
            except DataError as exc:
                raise DataError(STORE_INVALID, f'{header_path} is not a valid store header',
                                context={'root': str(root), 'detail': exc.to_dict()}) from None
            return cls(root, header)
        if not create:
            raise DataError(STORE_NOT_INITIALIZED, f'no data store at {root} (store.json missing)',
                            context={'root': str(root)})
        if root.exists() and any(root.iterdir()):
            raise DataError(STORE_INVALID, f'{root} exists, is not empty and holds no store.json: a store is never '
                            'created over foreign content', context={'root': str(root)})
        return cls._initialise(root)

    @classmethod
    def _initialise(cls, root: Path) -> 'Store':
        root.mkdir(parents=True, exist_ok=True)
        header = {'schema': STORE_SCHEMA, 'store_id': f'store-{secrets.token_hex(4)}', 'created': utc_iso(),
                  'schemas': list(WRITTEN_SCHEMAS)}
        for sub in ('vocab', 'calibrations', 'sessions'):
            (root / sub).mkdir(exist_ok=True)
        write_new_file(root / 'store.json', pretty_json(header))
        fsync_dir(root.parent)
        return cls(root, header)

    # ---- paths ---------------------------------------------------------------------------------------

    def path(self, relative: str) -> Path:
        return self.root.joinpath(*relative.split('/'))

    def session_dir(self, session_id: str) -> Path:
        return self.root / 'sessions' / s.session_id(session_id, 'session_id')

    def list_sessions(self) -> list:
        sessions = self.root / 'sessions'
        if not sessions.is_dir():
            return []
        return sorted(p.name for p in sessions.iterdir() if p.is_dir() and s.SESSION_ID_RE.match(p.name))

    # ---- vocabulary ----------------------------------------------------------------------------------

    @staticmethod
    def _vocab_rel(sha256: str) -> str:
        return f'vocab/vocab-{sha256[len("sha256:"):][:16]}.json'

    def pin_vocabulary(self, vocab: Vocabulary) -> dict:
        """Keep a copy of ``vocab`` in the store (idempotent); returns the session's reference to it."""
        path = self.path(self._vocab_rel(vocab.sha256))
        ref = {'schema': vocab.schema, 'sha256': vocab.sha256}
        if path.exists():
            self.load_vocabulary(ref)   # HASH_MISMATCH if the pinned copy was altered
        else:
            path.parent.mkdir(exist_ok=True)
            write_new_file(path, pretty_json(vocab.to_dict()))
        return ref

    def load_vocabulary(self, ref: dict) -> Vocabulary:
        """The pinned vocabulary ``ref`` names; ``HASH_MISMATCH`` if its file no longer hashes to it."""
        rel = self._vocab_rel(ref['sha256'])
        path = self.path(rel)
        if not path.is_file():
            raise DataError(MALFORMED_RECORD, f'pinned vocabulary {rel} is missing', context={'path': rel})
        vocab = Vocabulary.from_dict(parse_json(path.read_bytes(), rel))
        if vocab.sha256 != ref['sha256']:
            raise DataError(HASH_MISMATCH, f'{rel} does not hash to the pinned {ref["sha256"]}',
                            context={'path': rel, 'expected': ref['sha256'], 'actual': vocab.sha256})
        return vocab

    # ---- calibrations --------------------------------------------------------------------------------

    @staticmethod
    def calibration_rel(calibration_id: str) -> str:
        return f'calibrations/{calibration_id}.json'

    def put_calibration(self, record: CalibrationRecord) -> str:
        """Store an immutable calibration record (idempotent for identical content); returns its id."""
        rel = self.calibration_rel(record.calibration_id)
        path = self.path(rel)
        if not path.exists():
            path.parent.mkdir(exist_ok=True)
            try:
                write_new_file(path, pretty_json(record.to_dict()))
                return record.calibration_id
            except FileExistsError:
                pass                    # written concurrently: compare below
        stored = parse_json(path.read_bytes(), rel)
        if stored.get('sha256') != record.sha256:
            raise DataError(CALIBRATION_CONFLICT, f'{rel} exists with other content',
                            context={'path': rel, 'calibration_id': record.calibration_id})
        return record.calibration_id

    def has_calibration(self, calibration_id: str) -> bool:
        return self.path(self.calibration_rel(calibration_id)).is_file()

    def get_calibration(self, calibration_id: str, vocab: Vocabulary) -> CalibrationRecord:
        rel = self.calibration_rel(s.calibration_id(calibration_id, 'calibration_id'))
        path = self.path(rel)
        if not path.is_file():
            raise DataError(CALIBRATION_NOT_FOUND, f'calibration {calibration_id} is not in the store',
                            context={'calibration_id': calibration_id, 'path': rel})
        return CalibrationRecord.from_dict(parse_json(path.read_bytes(), rel), vocab, where=rel)

    # ---- sessions ------------------------------------------------------------------------------------

    def create_session(self, **spec):
        """Open a new ACTIVE session and return its :class:`~trainit_data.writer.SessionWriter`."""
        from .writer import SessionWriter
        return SessionWriter.create(self, **spec)

    def open_session(self, session_id: str):
        """A read-only :class:`~trainit_data.reader.SessionReader` over an existing session."""
        from .reader import SessionReader
        if not (self.session_dir(session_id) / 'session.json').is_file():
            raise DataError(SESSION_NOT_FOUND, f'no session {session_id} in {self.root}',
                            context={'session_id': session_id})
        return SessionReader(self, session_id)
