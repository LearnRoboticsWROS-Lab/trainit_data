"""Canonical JSON and sha256: the ONE serialization / hashing mechanism of the core (INTERNAL).

Every content hash, content-addressed id, index line and pinned vocabulary goes through these functions
(D0_SCHEMA_PROPOSAL.md §0). The canonical form is UTF-8, sorted keys, separators ``(",", ":")``, no NaN /
Infinity. Python 3.10 and 3.11 produce identical bytes for it (float ``repr`` is the shortest round-trip
form in both), which ``test/test_canonical.py`` pins with literal hashes.
"""
import hashlib
import json

from .errors import INVALID_VALUE, MALFORMED_RECORD, DataError

_CHUNK = 1 << 20


def canonical_json(value) -> bytes:
    """The canonical bytes of a JSON value. Refuses NaN / Infinity and non-JSON Python objects."""
    try:
        return json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False,
                          allow_nan=False).encode('utf-8')
    except (TypeError, ValueError) as exc:
        raise DataError(INVALID_VALUE, 'value is not canonical JSON', cause=exc) from None


def pretty_json(value) -> bytes:
    """The on-disk form of a JSON record: sorted keys, indented, newline-terminated (never hashed)."""
    canonical_json(value)                     # same refusals as the canonical form
    return (json.dumps(value, sort_keys=True, indent=2, ensure_ascii=False, allow_nan=False) + '\n').encode('utf-8')


def _no_constants(name):
    raise ValueError(f'non-finite literal {name}')


def _no_duplicate_keys(pairs):
    out = {}
    for key, value in pairs:
        if key in out:
            raise ValueError(f'duplicate key {key!r}')
        out[key] = value
    return out


def parse_json(data: bytes, where: str):
    """Parse JSON bytes strictly: NaN / Infinity literals and duplicate keys are refused (two different
    files must never parse to the same record)."""
    try:
        return json.loads(data.decode('utf-8'), parse_constant=_no_constants, object_pairs_hook=_no_duplicate_keys)
    except (UnicodeDecodeError, ValueError, RecursionError) as exc:
        raise DataError(MALFORMED_RECORD, f'{where} is not valid JSON', context={'where': where},
                        cause=exc) from None


def sha256_bytes(data: bytes) -> str:
    return 'sha256:' + hashlib.sha256(data).hexdigest()


def sha256_json(value) -> str:
    """The content hash of a JSON value: the sha256 of its canonical form."""
    return sha256_bytes(canonical_json(value))


def sha256_file(path) -> tuple:
    """(``sha256:<hex>``, size) of a file, streamed in chunks (never read whole into memory)."""
    digest = hashlib.sha256()
    size = 0
    with open(path, 'rb') as handle:
        while True:
            chunk = handle.read(_CHUNK)
            if not chunk:
                break
            digest.update(chunk)
            size += len(chunk)
    return 'sha256:' + digest.hexdigest(), size
