"""Strict parsing helpers shared by every record (INTERNAL).

The TSA strictness pattern (D0_SCHEMA_PROPOSAL.md §0): every key of a record object must be consumed by
its parser, otherwise ``UNKNOWN_FIELD``. The only open map is a record's ``extensions``, whose keys are
namespaced (``trainit.d1.foo``) and whose values are carried verbatim. Optional fields are absent, never
``null``. Numbers that are physically real-valued (``K``, translations) are normalised to ``float`` so a
source that wrote ``424`` and one that wrote ``424.0`` hash identically.
"""
import copy
import math
import re

from .errors import (INVALID_VALUE, MALFORMED_RECORD, MISSING_FIELD, SCHEMA_MAJOR_UNSUPPORTED,
                     UNKNOWN_FIELD, DataError)

IDENT_RE = re.compile(r'^[a-z0-9]+(?:_[a-z0-9]+)*$')          # sensor / stream / trigger ids: no `__`, no edges
SHA256_RE = re.compile(r'^sha256:[0-9a-f]{64}$')
SESSION_ID_RE = re.compile(r'^ses-\d{8}T\d{6}Z-[0-9a-f]{8}$')
CALIBRATION_ID_RE = re.compile(r'^cal-[0-9a-f]{16}$')
EXTENSION_KEY_RE = re.compile(r'^[a-z0-9_]+(?:\.[a-z0-9_]+)+$')
SCHEMA_RE = re.compile(r'^(trainit-[a-z-]+)/(\d+)\.(\d+)$')
CODE_RE = re.compile(r'^[A-Z][A-Z0-9_]*$')


def fail(code, message, where, **context):
    raise DataError(code, f'{where}: {message}', context={'where': where, **context})


class Fields:
    """Reads one JSON object strictly. Call :meth:`done` once every known key has been read."""

    def __init__(self, value, where: str):
        if not isinstance(value, dict):
            fail(MALFORMED_RECORD, 'expected a JSON object', where)
        self._value = value
        self._where = where
        self._seen = set()

    def req(self, key, check):
        if key not in self._value:
            fail(MISSING_FIELD, f'required field {key!r} is absent', self._where, field=key)
        return self.opt(key, check)

    def opt(self, key, check, default=None):
        self._seen.add(key)
        if key not in self._value:
            return default
        value = self._value[key]
        if value is None:
            fail(INVALID_VALUE, f'{key!r} is null; optional fields are absent, never null', self._where, field=key)
        return check(value, f'{self._where}.{key}')

    def extensions(self) -> dict:
        return self.opt('extensions', extensions, {})

    def done(self):
        extra = sorted(set(self._value) - self._seen)
        if extra:
            fail(UNKNOWN_FIELD, f'unknown field(s) {extra}', self._where, fields=extra)


# ---- value checks: (value, where) -> normalised value -------------------------------------------------

def string(value, where):
    if not isinstance(value, str) or not value:
        fail(INVALID_VALUE, 'expected a non-empty string', where)
    return value


def text(value, where):
    """A string that may be empty (free-text notes)."""
    if not isinstance(value, str):
        fail(INVALID_VALUE, 'expected a string', where)
    return value


def token(value, where):
    """A non-empty string without whitespace (frame ids, media types, model names)."""
    value = string(value, where)
    if any(ch.isspace() for ch in value):
        fail(INVALID_VALUE, 'must not contain whitespace', where)
    return value


def ident(value, where):
    value = string(value, where)
    if not IDENT_RE.match(value):
        fail(INVALID_VALUE, f'{value!r} is not an identifier ([a-z0-9] words joined by single "_")', where)
    return value


def code(value, where):
    value = string(value, where)
    if not CODE_RE.match(value):
        fail(INVALID_VALUE, f'{value!r} is not an UPPER_SNAKE code', where)
    return value


def matching(regex, what):
    def check(value, where):
        value = string(value, where)
        if not regex.match(value):
            fail(INVALID_VALUE, f'{value!r} is not a valid {what}', where)
        return value
    return check


sha256 = matching(SHA256_RE, 'sha256 reference (sha256:<64 hex>)')
session_id = matching(SESSION_ID_RE, 'session id')
calibration_id = matching(CALIBRATION_ID_RE, 'calibration id')


def integer(minimum=None):
    def check(value, where):
        if isinstance(value, bool) or not isinstance(value, int):
            fail(INVALID_VALUE, 'expected an integer', where)
        if minimum is not None and value < minimum:
            fail(INVALID_VALUE, f'must be >= {minimum}', where)
        return value
    return check


def real(value, where):
    """A finite JSON number, normalised to float (and -0.0 to 0.0)."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        fail(INVALID_VALUE, 'expected a number', where)
    try:
        value = float(value)
    except OverflowError:
        value = math.inf
    if not math.isfinite(value):
        fail(INVALID_VALUE, 'must be finite', where)
    return value + 0.0


def positive_real(value, where):
    value = real(value, where)
    if value <= 0.0:
        fail(INVALID_VALUE, 'must be > 0', where)
    return value


def boolean(value, where):
    if not isinstance(value, bool):
        fail(INVALID_VALUE, 'expected true / false', where)
    return value


def one_of(allowed):
    """A CLOSED set owned by code (its semantics are implemented, so it is not vocabulary)."""
    def check(value, where):
        value = string(value, where)
        if value not in allowed:
            fail(INVALID_VALUE, f'{value!r} is not one of {sorted(allowed)}', where)
        return value
    return check


def json_value(value, where):
    """A plain JSON value (str keys, lists, finite numbers): what a parsed file holds. Content built in
    Python with tuples, int keys or NaN is refused instead of being silently coerced by the serializer."""
    if isinstance(value, dict):
        for key, item in value.items():
            if not isinstance(key, str):
                fail(INVALID_VALUE, f'object key {key!r} is not a string', where)
            json_value(item, f'{where}.{key}')
    elif isinstance(value, list):
        for i, item in enumerate(value):
            json_value(item, f'{where}[{i}]')
    elif isinstance(value, float):
        if not math.isfinite(value):
            fail(INVALID_VALUE, 'non-finite number', where)
    elif value is not None and not isinstance(value, (str, int, bool)):
        fail(INVALID_VALUE, f'{type(value).__name__} is not a JSON value', where)
    return value


def opaque(value, where):
    """An object the core stores, hashes and carries VERBATIM but does not interpret (e.g. ``backend_details``).
    Its numbers are not normalised: ``1`` and ``1.0`` inside it are different content."""
    if not isinstance(value, dict):
        fail(INVALID_VALUE, 'expected a JSON object', where)
    return copy.deepcopy(json_value(value, where))


def str_map(value, where):
    value = opaque(value, where)
    for key, item in value.items():
        string(item, f'{where}.{key}')
    return value


def list_of(check, *, length=None, min_length=0, unique=False):
    def checker(value, where):
        if not isinstance(value, list):
            fail(INVALID_VALUE, 'expected a JSON array', where)
        if length is not None and len(value) != length:
            fail(INVALID_VALUE, f'expected exactly {length} items', where)
        if len(value) < min_length:
            fail(INVALID_VALUE, f'expected at least {min_length} items', where)
        out = tuple(check(item, f'{where}[{i}]') for i, item in enumerate(value))
        if unique and len(set(out)) != len(out):
            fail(INVALID_VALUE, 'items must be unique', where)
        return out
    return checker


def extensions(value, where):
    if not isinstance(value, dict):
        fail(INVALID_VALUE, 'expected a JSON object', where)
    for key in value:
        if not EXTENSION_KEY_RE.match(key):
            fail(INVALID_VALUE, f'extension key {key!r} must be namespaced (e.g. "trainit.d1.name")', where)
    return copy.deepcopy(json_value(value, where))


def relative_path(value, where):
    """A store-relative POSIX path: no leading '/', no '..', no backslash, no empty segment."""
    value = string(value, where)
    parts = value.split('/')
    if value.startswith('/') or '\\' in value or any(p in ('', '.', '..') for p in parts):
        fail(INVALID_VALUE, f'{value!r} is not a store-relative POSIX path', where)
    return value


def schema(expected_name: str, supported_major: int):
    """``trainit-<record>/<major>.<minor>``: same name, same major; any minor of that major is read."""
    def check(value, where):
        value = string(value, where)
        match = SCHEMA_RE.match(value)
        if not match or match.group(1) != expected_name:
            fail(MALFORMED_RECORD, f'schema {value!r} is not a {expected_name}/<major>.<minor> record', where)
        major = int(match.group(2))
        if major != supported_major:
            fail(SCHEMA_MAJOR_UNSUPPORTED, f'{value!r}: this reader supports {expected_name} major '
                 f'{supported_major} only', where, schema=value)
        return value
    return check


def put(out: dict, key, value):
    """Serialise an optional field: absent stays absent."""
    if value is None:
        return
    if isinstance(value, tuple):
        value = list(value)
    out[key] = value


def vector(length):
    return list_of(real, length=length)


def quaternion(value, where):
    """A unit quaternion ``[x, y, z, w]`` (|q| within 1e-3 of 1; values are stored as given)."""
    q = vector(4)(value, where)
    norm = math.sqrt(sum(c * c for c in q))
    if abs(norm - 1.0) > 1e-3:
        fail(INVALID_VALUE, f'not a unit quaternion (|q| = {norm:.6f})', where)
    return q
