"""Read-only access to SEALED sessions for datasets, labels and exports (INTERNAL, D0.6).

A dataset references a session by its id AND its sealed index hash. Before anything is built on a session,
its header must be sealed (``SESSION_NOT_SEALED``) and its index must still hash to what the header sealed
(``HASH_MISMATCH``): a dataset is never built on an open or altered session. Payload bytes are checked
against their sha256 where they are read (the export), and by ``verify``.
"""
import re

from . import _strict as s
from ._canonical import sha256_file
from .errors import HASH_MISMATCH, OBSERVATION_NOT_FOUND, SESSION_NOT_SEALED, DataError

OBSERVATION_ID_RE = re.compile(r'^(ses-\d{8}T\d{6}Z-[0-9a-f]{8})/(\d{6,})/([a-z0-9]+(?:_[a-z0-9]+)*)/'
                               r'([a-z0-9]+(?:_[a-z0-9]+)*)$')
observation_id = s.matching(OBSERVATION_ID_RE, 'observation id (<session_id>/<seq>/<sensor_id>/<stream>)')


def observation_key(value: str) -> tuple:
    """The canonical order of observation ids: session, then the capture's NUMBER, then sensor and stream (a
    plain string sort would put capture 1000000 before 999999)."""
    m = OBSERVATION_ID_RE.match(value)
    return (m.group(1), int(m.group(2)), m.group(3), m.group(4))


def capture_of(obs_id: str) -> str:
    """The capture id of an observation id (``<session_id>/<seq>``): the one place that splits the id grammar."""
    m = OBSERVATION_ID_RE.match(obs_id)
    if m is None:
        raise DataError(OBSERVATION_NOT_FOUND, f'{obs_id!r} is not an observation id',
                        context={'observation_id': obs_id})
    return f'{m.group(1)}/{m.group(2)}'


def capture_key(capture_id: str) -> tuple:
    session, seq = capture_id.split('/')
    return (session, int(seq))


class SealedSession:
    """One sealed session: its header, its captures (parsed once) and an observation lookup."""

    def __init__(self, store, session_id: str):
        reader = store.open_session(session_id)          # SESSION_NOT_FOUND
        header = reader.header
        if not header.sealed:
            raise DataError(SESSION_NOT_SEALED, f'session {session_id} is {header.status}: a dataset references '
                            'sealed sessions only (close it, or recover it if it was interrupted)',
                            context={'session_id': session_id, 'status': header.status})
        index_rel = f'sessions/{session_id}/captures.jsonl'
        actual, _ = sha256_file(store.path(index_rel))
        if actual != header.index['sha256']:
            raise DataError(HASH_MISMATCH, f'{index_rel} no longer hashes to the sealed {header.index["sha256"]}',
                            context={'path': index_rel, 'expected': header.index['sha256'], 'actual': actual})
        self.store = store
        self.session_id = session_id
        self.header = header
        self.vocab = reader.vocab
        self.reader = reader
        self.captures = list(reader.captures())
        self._observations = {o.observation_id: (c, o) for c in self.captures for o in c.observations}

    def observation(self, obs_id: str, where: str):
        """``(capture, observation)``; ``OBSERVATION_NOT_FOUND`` if the session has no such observation."""
        found = self._observations.get(obs_id)
        if found is None:
            raise DataError(OBSERVATION_NOT_FOUND, f'{where}: session {self.session_id} has no observation '
                            f'{obs_id}', context={'where': where, 'observation_id': obs_id})
        return found


class SealedSessions:
    """A cache of :class:`SealedSession` (each session's index is read once per operation)."""

    def __init__(self, store):
        self.store = store
        self._open = {}

    def get(self, session_id: str) -> SealedSession:
        if session_id not in self._open:
            self._open[session_id] = SealedSession(self.store, session_id)
        return self._open[session_id]

    def observation(self, obs_id: str, where: str):
        return self.get(observation_key(obs_id)[0]).observation(obs_id, where)
