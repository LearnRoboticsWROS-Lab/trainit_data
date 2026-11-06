"""The stable error model of the canonical core (PUBLIC).

Every refusal of the core is a :class:`DataError` carrying a stable ``code`` (one of the constants below).
The code and the ``context`` keys are the contract; ``message`` is human text and ``cause`` keeps the
underlying diagnostic (an ``OSError`` text, a JSON decoder message) for people, never for programs.

The codes are plain strings on purpose: a ``str``-mixin ``Enum`` formats differently in Python 3.10 and
3.11, and the core must behave identically in both.
"""
from typing import Optional

# Parsing / schema evolution
UNKNOWN_FIELD = 'UNKNOWN_FIELD'                      # a key the schema does not define, outside `extensions`
MISSING_FIELD = 'MISSING_FIELD'                      # a required key is absent
INVALID_VALUE = 'INVALID_VALUE'                      # wrong type, range, format or cross-field rule
MALFORMED_RECORD = 'MALFORMED_RECORD'                # not JSON, not an object, or the wrong record schema
SCHEMA_MAJOR_UNSUPPORTED = 'SCHEMA_MAJOR_UNSUPPORTED'  # the record's major version is not this reader's
UNKNOWN_VOCAB = 'UNKNOWN_VOCAB'                      # a value that is not a row of the authoritative vocabulary
DATA_BLOCK_UNSUPPORTED = 'DATA_BLOCK_UNSUPPORTED'    # a vocabulary data block this core has no reader for

# Clocks / payloads
INCOMPATIBLE_CLOCKS = 'INCOMPATIBLE_CLOCKS'          # timestamps of different clock domains compared
ENCODING_REFUSED = 'ENCODING_REFUSED'                # an array / encoding / media type the modality does not allow

# Store / session lifecycle
STORE_INSIDE_REPOSITORY = 'STORE_INSIDE_REPOSITORY'  # the data store resolves inside a git work tree
STORE_NOT_INITIALIZED = 'STORE_NOT_INITIALIZED'      # no store.json and creation was not requested
STORE_INVALID = 'STORE_INVALID'                      # not a usable store (a file, foreign content, a bad header)
SESSION_NOT_FOUND = 'SESSION_NOT_FOUND'
SESSION_SEALED = 'SESSION_SEALED'                    # a write or a recovery targets a sealed session
SESSION_FAULTED = 'SESSION_FAULTED'                  # the writer lost the index; only `recover` may continue
SESSION_IN_USE = 'SESSION_IN_USE'                    # a live writer holds the session
CALIBRATION_NOT_FOUND = 'CALIBRATION_NOT_FOUND'
CALIBRATION_CONFLICT = 'CALIBRATION_CONFLICT'        # a calibration file exists with other content for the same id
CALIBRATION_MISMATCH = 'CALIBRATION_MISMATCH'        # an observation and its calibration disagree (sensor, size, frame)
WRITE_FAILED = 'WRITE_FAILED'                        # disk full / I/O error while persisting

# Integrity (verify / recover findings)
HASH_MISMATCH = 'HASH_MISMATCH'                      # bytes do not match their recorded sha256
PAYLOAD_MISSING = 'PAYLOAD_MISSING'
PAYLOAD_INVALID = 'PAYLOAD_INVALID'                  # a payload decodes to something its observation does not declare
INDEX_CORRUPT = 'INDEX_CORRUPT'                      # a complete index line is not a valid capture, or out of sequence
INDEX_TORN = 'INDEX_TORN'                            # the index ends with an incomplete line
ORPHAN_FILE = 'ORPHAN_FILE'                          # a payload-area file no committed capture references
SESSION_INCONSISTENT = 'SESSION_INCONSISTENT'        # the header disagrees with the index (counters, calibrations)
STALE_ACTIVE_MARKER = 'STALE_ACTIVE_MARKER'          # a sealed session still carries its `.active` marker
RECOVERY_UNSAFE = 'RECOVERY_UNSAFE'                  # recover cannot establish safety and changes nothing

# Acquisition configuration (trainit_data_ros)
SOURCE_DESCRIPTOR_INVALID = 'SOURCE_DESCRIPTOR_INVALID'  # the backend's source descriptor is missing or invalid

# Datasets, versions, labels, splits, exports (D0.6)
DATASET_EXISTS = 'DATASET_EXISTS'                    # a dataset identity is never overwritten
DATASET_NOT_FOUND = 'DATASET_NOT_FOUND'
VERSION_EXISTS = 'VERSION_EXISTS'                    # a dataset version is never overwritten
VERSION_NOT_FOUND = 'VERSION_NOT_FOUND'
LABEL_SET_NOT_FOUND = 'LABEL_SET_NOT_FOUND'
SESSION_NOT_SEALED = 'SESSION_NOT_SEALED'            # a dataset references sealed sessions only
OBSERVATION_NOT_FOUND = 'OBSERVATION_NOT_FOUND'      # a label references no observation of a sealed session
LABEL_INVALID = 'LABEL_INVALID'                      # a label (or an imported label line) breaks a rule; never clipped
LABEL_CONFLICT = 'LABEL_CONFLICT'                    # two label sets of one kind label the same observation
LABEL_KIND_UNSUPPORTED = 'LABEL_KIND_UNSUPPORTED'    # a representable label kind with no reader / producer yet
CONTENT_KIND_UNSUPPORTED = 'CONTENT_KIND_UNSUPPORTED'  # a dataset content kind this core has no reader for
DERIVATION_INVALID = 'DERIVATION_INVALID'            # a derived version does not match its derivation op
RIGHTS_CONFLICT = 'RIGHTS_CONFLICT'                  # rights that cannot be combined without a guess
DATASET_INCONSISTENT = 'DATASET_INCONSISTENT'        # a version / label set disagrees with what it references
EXPORT_DESTINATION_INVALID = 'EXPORT_DESTINATION_INVALID'  # not empty, inside a repository or inside the store
EXPORT_REFUSED = 'EXPORT_REFUSED'                    # the exporter cannot represent this content faithfully

CODES = frozenset({
    UNKNOWN_FIELD, MISSING_FIELD, INVALID_VALUE, MALFORMED_RECORD, SCHEMA_MAJOR_UNSUPPORTED, UNKNOWN_VOCAB,
    DATA_BLOCK_UNSUPPORTED, INCOMPATIBLE_CLOCKS, ENCODING_REFUSED, STORE_INSIDE_REPOSITORY,
    STORE_NOT_INITIALIZED, STORE_INVALID, SESSION_NOT_FOUND, SESSION_SEALED, SESSION_FAULTED, SESSION_IN_USE,
    CALIBRATION_NOT_FOUND, CALIBRATION_CONFLICT, CALIBRATION_MISMATCH, WRITE_FAILED, HASH_MISMATCH,
    PAYLOAD_MISSING, PAYLOAD_INVALID, INDEX_CORRUPT, INDEX_TORN, ORPHAN_FILE, SESSION_INCONSISTENT,
    STALE_ACTIVE_MARKER, RECOVERY_UNSAFE, SOURCE_DESCRIPTOR_INVALID, DATASET_EXISTS, DATASET_NOT_FOUND,
    VERSION_EXISTS, VERSION_NOT_FOUND, LABEL_SET_NOT_FOUND, SESSION_NOT_SEALED, OBSERVATION_NOT_FOUND, LABEL_INVALID,
    LABEL_CONFLICT, LABEL_KIND_UNSUPPORTED, CONTENT_KIND_UNSUPPORTED, DERIVATION_INVALID, RIGHTS_CONFLICT,
    DATASET_INCONSISTENT, EXPORT_DESTINATION_INVALID, EXPORT_REFUSED,
})


class DataError(Exception):
    """A refusal of the canonical core: a stable ``code`` + a JSON-safe ``context``."""

    def __init__(self, code: str, message: str, *, context: Optional[dict] = None,
                 cause: Optional[BaseException] = None):
        if code not in CODES:
            raise ValueError(f'unknown DataError code {code!r}')
        super().__init__(f'{code}: {message}')
        self.code = code
        self.message = message
        self.context = dict(context or {})
        self.cause = None if cause is None else f'{type(cause).__name__}: {cause}'

    def __reduce__(self):                # picklable (multiprocessing, test runners)
        return _rebuild, (self.code, self.message, self.context, self.cause)

    def to_dict(self) -> dict:
        out = {'code': self.code, 'message': self.message, 'context': self.context}
        if self.cause is not None:
            out['cause'] = self.cause
        return out


def _rebuild(code, message, context, cause):
    error = DataError(code, message, context=context)
    error.cause = cause
    return error


__all__ = ['DataError', 'CODES'] + sorted(CODES)
