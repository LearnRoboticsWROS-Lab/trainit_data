"""SessionReader: read-only access to one session (PUBLIC).

Reads the header with the session's PINNED vocabulary (authoritative), streams the index one capture at
a time, and decodes payloads after checking their sha256. It never writes.
"""
import numpy as np

from . import _codecs
from ._canonical import parse_json, sha256_bytes
from .capture import Capture
from .errors import (HASH_MISMATCH, INDEX_CORRUPT, INDEX_TORN, INVALID_VALUE, MALFORMED_RECORD, PAYLOAD_INVALID,
                     PAYLOAD_MISSING, DataError)
from .session_record import SessionHeader

_CHUNK = 1 << 20


def index_lines(path):
    """Yield ``(line_number, offset, raw_bytes, complete)`` of an index file, streamed.

    ``complete`` is False only for a trailing fragment without its newline (a torn or in-flight write).
    """
    with open(path, 'rb') as handle:
        number, offset, pending = 0, 0, b''
        while True:
            chunk = handle.read(_CHUNK)
            if not chunk:
                break
            pending += chunk
            *lines, pending = pending.split(b'\n')
            for line in lines:
                number += 1
                yield number, offset, line, True
                offset += len(line) + 1
        if pending:
            yield number + 1, offset, pending, False


def parse_index_line(raw: bytes, vocab, where: str) -> Capture:
    try:
        return Capture.from_dict(parse_json(raw, where), vocab, where)
    except DataError as exc:
        raise DataError(INDEX_CORRUPT, f'{where}: not a valid capture ({exc.code})',
                        context={'where': where, 'detail': exc.to_dict()}) from None


def read_header(store, session_id: str):
    """(header, vocabulary): the header parsed with the vocabulary it pins."""
    rel = f'sessions/{session_id}/session.json'
    raw = parse_json(store.path(rel).read_bytes(), rel)
    vocab_ref = raw.get('vocab') if isinstance(raw, dict) else None
    if not isinstance(vocab_ref, dict) or 'sha256' not in vocab_ref:
        raise DataError(MALFORMED_RECORD, f'{rel}: no pinned vocabulary reference',
                        context={'where': rel})
    vocab = store.load_vocabulary(vocab_ref)
    return SessionHeader.from_dict(raw, vocab, where=rel), vocab


class SessionReader:
    def __init__(self, store, session_id: str):
        self.store = store
        self.session_id = session_id
        self.header, self.vocab = read_header(store, session_id)

    def captures(self):
        """Committed captures, in order, one at a time. A sealed session's index must end cleanly
        (``INDEX_TORN``); an ACTIVE one may end with an in-flight fragment, which is skipped."""
        path = self.store.session_dir(self.session_id) / 'captures.jsonl'
        for number, _, raw, complete in index_lines(path):
            where = f'sessions/{self.session_id}/captures.jsonl:{number}'
            if not complete:
                if self.header.sealed:
                    raise DataError(INDEX_TORN, f'{where}: the sealed index ends with an incomplete line',
                                    context={'where': where})
                return
            yield parse_index_line(raw, self.vocab, where)

    def load_payload(self, observation) -> np.ndarray:
        """The observation's array, bit-exact; ``HASH_MISMATCH`` if the file is not the recorded bytes."""
        ref = observation.payload
        if ref is None:
            raise DataError(INVALID_VALUE, f'{observation.observation_id} has no payload: a {observation.data.block!r} '
                            'observation holds its values in the record (observation.data)',
                            context={'observation_id': observation.observation_id})
        path = self.store.path(ref.path)
        if not path.is_file():
            raise DataError(PAYLOAD_MISSING, f'{ref.path} is missing', context={'path': ref.path})
        data = path.read_bytes()
        if sha256_bytes(data) != ref.sha256 or len(data) != ref.bytes:
            raise DataError(HASH_MISMATCH, f'{ref.path} does not match its recorded sha256',
                            context={'path': ref.path, 'observation_id': observation.observation_id})
        array = _codecs.decode(data, ref.media_type, where=ref.path)
        encoding = self.vocab.encodings[observation.data.encoding]
        try:
            _codecs.check_array(array, encoding, observation.data.encoding)
        except DataError as exc:
            raise DataError(PAYLOAD_INVALID, f'{ref.path}: {exc.message}', context={'path': ref.path}) from None
        if array.shape[:2] != (observation.data.height, observation.data.width):
            raise DataError(PAYLOAD_INVALID, f'{ref.path}: {array.shape[1]}x{array.shape[0]} pixels, the observation '
                            f'declares {observation.data.width}x{observation.data.height}', context={'path': ref.path})
        return array
