"""The versioned, extensible vocabulary ``trainit-vocab/1.0`` (PUBLIC).

Backend kinds, modalities, encodings, transports, trigger kinds / origins, context namespaces, sync
policies, stamp origins, missing-stream reasons, label sources, provenance kinds and derivation
operations are vocabulary DATA, not Python enums (D0_SCHEMA_PROPOSAL.md §1). Adding a row -- a ``gazebo``
backend, a sync policy, a new image modality -- is a data change with no schema-code change.

A vocabulary is identified by its content hash. A session pins the vocabulary it was written with (the
store keeps a copy under ``vocab/``) and that pinned vocabulary is AUTHORITATIVE when the session is
read or verified: a value that is not one of its rows is refused with ``UNKNOWN_VOCAB``.

Closed sets with implemented semantics (clock domains, capture outcomes, sync statuses, session states,
rights classifications) stay in code: adding one of those needs code, so a vocabulary row would lie. In
particular a sync policy added as a row must report its result in the existing statuses (SYNCED /
WITHIN_TOLERANCE / OUT_OF_TOLERANCE / SINGLE_STREAM / INCOMPATIBLE_CLOCKS), whose rules ``capture.py`` checks.
"""
import copy
import re
from pathlib import Path

from . import _strict as s
from ._canonical import parse_json, sha256_json
from .errors import INVALID_VALUE, UNKNOWN_VOCAB, DataError

SCHEMA_NAME = 'trainit-vocab'
SCHEMA_MAJOR = 1
DEFAULT_FILE = Path(__file__).with_name('vocab_data') / 'trainit-vocab-1.0.json'

_BLOCK_VERSION_RE = re.compile(r'^\d+\.\d+$')

# Array element types an encoding row may name: explicit, little-endian, platform-independent (never left to
# the local numpy's parser, so a pinned vocabulary means the same on every machine).
DTYPES = frozenset({'uint8', 'uint16', 'int16', 'int32', 'float16', 'float32', 'float64'})

# Rows the core's own rules use: a vocabulary without them cannot describe what the core records.
REQUIRED_ROWS = {'missing_reasons': ('INCOMPATIBLE_CLOCKS', 'WRITE_FAILED')}

LIST_CATEGORIES = (
    'transports', 'backends', 'sim_sources', 'payload_kinds_reserved', 'stamp_origins', 'trigger_kinds',
    'trigger_origins',
    'context_namespaces', 'missing_reasons', 'label_sources', 'intrinsic_provenance', 'extrinsic_provenance',
    'derivation_ops',
)


_MAP_CATEGORIES = ('modalities', 'encodings', 'data_blocks', 'sync_policies')


def _row_names(value, where):
    return s.list_of(s.token, unique=True)(value, where)


def _modality(value, where):
    f = s.Fields(value, where)
    row = {
        'data_block': f.req('data_block', s.token),
        'encodings': f.req('encodings', s.list_of(s.token, unique=True)),
        'payloads': f.req('payloads', s.list_of(s.token, unique=True)),
        'unit_scale_required': f.opt('unit_scale_required', s.boolean, False),
    }
    f.done()
    # A modality stores payload files (an encoding + a media type, e.g. an image) or none at all (its values are
    # in the record, e.g. the ``vector`` block of a low-rate IMU reading, test D0-J): both lists, or neither.
    # Which of the two a given block needs is its reader's rule (``observation.modality_problem``, checked when a
    # session declares the stream), not the vocabulary's: the vocabulary stays data.
    if bool(row['encodings']) != bool(row['payloads']):
        s.fail(INVALID_VALUE, 'encodings and payloads are both listed (a modality with payload files) or both empty '
               '(its values are held in the record)', where)
    return row


def _encoding(value, where):
    f = s.Fields(value, where)
    dtype = f.req('dtype', s.one_of(DTYPES))
    channels = f.req('channels', s.integer(1))
    f.done()
    return {'dtype': dtype, 'channels': channels}


def _data_block(value, where):
    f = s.Fields(value, where)
    version = f.req('version', s.matching(_BLOCK_VERSION_RE, 'block version (<major>.<minor>)'))
    f.done()
    return {'version': version}


def _sync_policy(value, where):
    f = s.Fields(value, where)
    versions = f.req('versions', s.list_of(s.integer(1), min_length=1, unique=True))
    f.done()
    return {'versions': versions}


def _named_rows(check):
    def checker(value, where):
        if not isinstance(value, dict) or not value:
            s.fail(INVALID_VALUE, 'expected a non-empty JSON object of rows', where)
        return {s.token(name, f'{where}.<key>'): check(row, f'{where}.{name}') for name, row in value.items()}
    return checker


class Vocabulary:
    """One parsed vocabulary document. Immutable by convention; identity = :attr:`sha256`."""

    def __init__(self, document: dict):
        f = s.Fields(document, 'vocab')
        self.schema = f.req('schema', s.schema(SCHEMA_NAME, SCHEMA_MAJOR))
        self._lists = {name: f.req(name, _row_names) for name in LIST_CATEGORIES}
        self.modalities = f.req('modalities', _named_rows(_modality))
        self.encodings = f.req('encodings', _named_rows(_encoding))
        self.data_blocks = f.req('data_blocks', _named_rows(_data_block))
        self.sync_policies = f.req('sync_policies', _named_rows(_sync_policy))
        self.extensions = f.extensions()
        f.done()
        for name, row in self.modalities.items():
            where = f'vocab.modalities.{name}'
            if row['data_block'] not in self.data_blocks:
                s.fail(INVALID_VALUE, f'data block {row["data_block"]!r} is not a data_blocks row', where)
            for enc in row['encodings']:
                if enc not in self.encodings:
                    s.fail(INVALID_VALUE, f'encoding {enc!r} is not an encodings row', where)
        for category, needed in REQUIRED_ROWS.items():
            absent = [row for row in needed if row not in self._lists[category]]
            if absent:
                s.fail(INVALID_VALUE, f'rows {absent} are required: the core records them', f'vocab.{category}')
        self._document = self.to_dict()
        self.sha256 = sha256_json(self._document)

    @classmethod
    def default(cls) -> 'Vocabulary':
        """The vocabulary shipped with this release of the core."""
        return cls(parse_json(DEFAULT_FILE.read_bytes(), str(DEFAULT_FILE.name)))

    @classmethod
    def from_dict(cls, document: dict) -> 'Vocabulary':
        return cls(document)

    def to_dict(self) -> dict:
        out = {'schema': self.schema}
        out.update({name: list(rows) for name, rows in self._lists.items()})
        out['modalities'] = {k: {'data_block': v['data_block'], 'encodings': list(v['encodings']),
                                 'payloads': list(v['payloads']), 'unit_scale_required': v['unit_scale_required']}
                             for k, v in self.modalities.items()}
        out['encodings'] = {k: dict(v) for k, v in self.encodings.items()}
        out['data_blocks'] = {k: dict(v) for k, v in self.data_blocks.items()}
        out['sync_policies'] = {k: {'versions': list(v['versions'])} for k, v in self.sync_policies.items()}
        if self.extensions:
            out['extensions'] = copy.deepcopy(self.extensions)
        return out

    def rows(self, category: str) -> tuple:
        if category in self._lists:
            return self._lists[category]
        if category in _MAP_CATEGORIES:
            return tuple(getattr(self, category))
        raise ValueError(f'{category!r} is not a vocabulary category')

    def require(self, category: str, value, where: str = 'value'):
        """``value`` if it is a row of ``category``, else ``UNKNOWN_VOCAB``."""
        if not isinstance(value, str) or value not in self.rows(category):
            raise DataError(UNKNOWN_VOCAB, f'{where}: {value!r} is not a {category} row of the vocabulary '
                            f'({self.schema} {self.sha256[:19]}…)',
                            context={'where': where, 'category': category, 'value': value,
                                     'vocab_sha256': self.sha256})
        return value

    def checker(self, category: str):
        """A ``_strict`` value check bound to a vocabulary category."""
        def check(value, where):
            return self.require(category, s.token(value, where), where)
        return check

    def require_sync_policy(self, name: str, version: int, where: str):
        self.require('sync_policies', name, f'{where}.name')
        if version not in self.sync_policies[name]['versions']:
            raise DataError(UNKNOWN_VOCAB, f'{where}: version {version} of {name} is not a vocabulary row',
                            context={'where': where, 'category': 'sync_policies', 'value': f'{name}/{version}',
                                     'vocab_sha256': self.sha256})


__all__ = ['Vocabulary']
