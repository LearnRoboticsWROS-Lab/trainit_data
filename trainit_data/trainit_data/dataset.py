"""Dataset identity and rights (PUBLIC, D0.6).

A DATASET is a logical name with a description and declared rights, ``datasets/<dataset_id>/dataset.json``
(``trainit-dataset-identity/1.0``). It holds no data: its content is its immutable VERSIONS (``versions``),
built from sealed sessions and its LABEL SETS (``labels``). Session != Dataset != DatasetVersion: a dataset
never copies, moves or rewrites a session.

The identity is CONTENT-KIND NEUTRAL: nothing here is image-, observation- or YOLO-specific, so a future
``EPISODES`` dataset (D1) has the same identity record. It is immutable once written (``DATASET_EXISTS``);
metadata edits after creation are not in D0.6.

Rights (D0_DESIGN.md §11, decision D8) are a CLOSED order of restrictiveness, PUBLIC_APPROVED < INTERNAL <
CUSTOMER_RESTRICTED. :func:`combine_rights` gives a version the MOST restrictive of everything it is built
from and refuses (``RIGHTS_CONFLICT``) what it cannot combine without a guess: rights are never promoted,
and two owners are never merged.
"""
import os
from dataclasses import dataclass, field

from . import __version__
from . import _strict as s
from ._canonical import parse_json, pretty_json
from ._fsio import makedirs_durable, write_new_file
from .errors import DATASET_EXISTS, DATASET_NOT_FOUND, INVALID_VALUE, RIGHTS_CONFLICT, WRITE_FAILED, DataError
from .session_record import CLASSIFICATIONS

SCHEMA_NAME = 'trainit-dataset-identity'
SCHEMA_MAJOR = 1
SCHEMA = 'trainit-dataset-identity/1.0'

DATASET_ID_MAX = 64
# restrictiveness, least to most: a combination takes the maximum and never goes down
RIGHTS_ORDER = ('PUBLIC_APPROVED', 'INTERNAL', 'CUSTOMER_RESTRICTED')
assert set(RIGHTS_ORDER) == CLASSIFICATIONS


def dataset_id(value, where):
    value = s.ident(value, where)
    if len(value) > DATASET_ID_MAX:
        s.fail(INVALID_VALUE, f'a dataset id has at most {DATASET_ID_MAX} characters', where)
    return value


def rights(value, where):
    """``{classification, owner_org?, license?, notes?}`` -- the session record's rights block."""
    f = s.Fields(value, where)
    out = {'classification': f.req('classification', s.one_of(CLASSIFICATIONS))}
    for name in ('owner_org', 'license', 'notes'):
        s.put(out, name, f.opt(name, s.string))
    f.done()
    return out


def created(value, where):
    f = s.Fields(value, where)
    out = {'utc': f.req('utc', s.string), 'by': f.req('by', s.string)}
    f.done()
    return out


def created_now() -> dict:
    from .store import utc_iso
    return {'utc': utc_iso(), 'by': f'trainit-data {__version__}'}


def _domain(value, where):
    f = s.Fields(value, where)
    out = {}
    for name in ('object', 'task', 'environment'):
        s.put(out, name, f.opt(name, s.string))
    f.done()
    return out


@dataclass(frozen=True)
class DatasetIdentity:
    schema: str
    dataset_id: str
    name: str
    rights: dict
    created: dict
    description: str = None
    domain: dict = field(default_factory=dict)          # {object?, task?, environment?}: descriptive only
    extensions: dict = field(default_factory=dict)

    @classmethod
    def from_dict(cls, value, where='dataset') -> 'DatasetIdentity':
        f = s.Fields(value, where)
        identity = cls(schema=f.req('schema', s.schema(SCHEMA_NAME, SCHEMA_MAJOR)),
                       dataset_id=f.req('dataset_id', dataset_id),
                       name=f.req('name', s.string),
                       rights=f.req('rights', rights),
                       created=f.req('created', created),
                       description=f.opt('description', s.string),
                       domain=f.opt('domain', _domain, {}),
                       extensions=f.extensions())
        f.done()
        return identity

    def to_dict(self) -> dict:
        out = {'schema': self.schema, 'dataset_id': self.dataset_id, 'name': self.name, 'rights': dict(self.rights),
               'created': dict(self.created)}
        s.put(out, 'description', self.description)
        if self.domain:
            out['domain'] = dict(self.domain)
        if self.extensions:
            out['extensions'] = dict(self.extensions)
        return out


# ---- store layout ----------------------------------------------------------------------------------------

def dataset_rel(dataset: str) -> str:
    return f'datasets/{dataset_id(dataset, "dataset_id")}'


def identity_rel(dataset: str) -> str:
    return f'{dataset_rel(dataset)}/dataset.json'


def make_read_only(path):
    """Immutability is enforced by the API and checked by verify; read-only modes are an extra guard."""
    os.chmod(path, 0o444)


def create_dataset(store, dataset: str, *, name: str, rights: dict = None, description: str = None,
                   domain: dict = None) -> DatasetIdentity:
    """Write a new dataset identity; ``DATASET_EXISTS`` if the id is taken. Rights default to INTERNAL:
    PUBLIC_APPROVED is only ever the caller's explicit statement."""
    record = {'schema': SCHEMA, 'dataset_id': dataset, 'name': name,
              'rights': rights if rights is not None else {'classification': 'INTERNAL'}, 'created': created_now()}
    s.put(record, 'description', description)
    if domain:
        record['domain'] = domain
    identity = DatasetIdentity.from_dict(record, 'dataset')         # one validation path
    path = store.path(identity_rel(identity.dataset_id))
    exists = DataError(DATASET_EXISTS, f'dataset {identity.dataset_id} already exists in {store.root}',
                       context={'dataset_id': identity.dataset_id})
    # an EMPTY directory is an interrupted creation of nothing: it is completed; anything in it is a dataset
    if path.parent.exists() and any(path.parent.iterdir()):
        raise exists
    try:
        makedirs_durable(path.parent)
        write_new_file(path, pretty_json(identity.to_dict()))
        make_read_only(path)
    except FileExistsError:
        raise exists from None
    except OSError as exc:
        raise DataError(WRITE_FAILED, f'writing {identity_rel(identity.dataset_id)} failed', cause=exc,
                        context={'dataset_id': identity.dataset_id}) from None
    return identity


def load_dataset(store, dataset: str) -> DatasetIdentity:
    rel = identity_rel(dataset)
    path = store.path(rel)
    if not path.is_file():
        raise DataError(DATASET_NOT_FOUND, f'no dataset {dataset} in {store.root}', context={'dataset_id': dataset})
    identity = DatasetIdentity.from_dict(parse_json(path.read_bytes(), rel), rel)
    if identity.dataset_id != dataset:
        s.fail(INVALID_VALUE, f'the identity names dataset {identity.dataset_id!r}', rel)
    return identity


def list_datasets(store) -> list:
    root = store.root / 'datasets'
    if not root.is_dir():
        return []
    return sorted(p.name for p in root.iterdir() if p.is_dir() and s.IDENT_RE.match(p.name))


# ---- rights ----------------------------------------------------------------------------------------------

def combine_rights(basis: list) -> dict:
    """The rights of something built from ``basis`` (a list of ``{source, rights}``): the most restrictive
    classification, never promoted. Refused (``RIGHTS_CONFLICT``) when combining needs a guess:

    - CUSTOMER_RESTRICTED from different owners, or with an owner on one side and none on the other;
    - PUBLIC_APPROVED everywhere but under different licenses (or a license on one side only).

    The owner / license rules apply among the sources AT the resulting classification: those are the sources the
    result's owner and license come from. A less restrictive source does not constrain them (internal data of one
    owner combined with customer data of another is that customer's restricted data); every source's own rights
    stay in ``rights_basis``. The result keeps ``owner_org`` (CUSTOMER_RESTRICTED) and ``license``
    (PUBLIC_APPROVED) only where they are unambiguous; notes are not merged. Label sets carry no rights of their own
    in D0.6: an imported label set is covered by the dataset's declared rights."""
    if not basis:
        raise ValueError('combine_rights needs at least one source')
    level = max(RIGHTS_ORDER.index(b['rights']['classification']) for b in basis)
    classification = RIGHTS_ORDER[level]
    out = {'classification': classification}
    if classification == 'CUSTOMER_RESTRICTED':
        owners = {b['rights'].get('owner_org') for b in basis
                  if b['rights']['classification'] == 'CUSTOMER_RESTRICTED'}
        if len(owners) > 1:
            raise DataError(RIGHTS_CONFLICT, f'CUSTOMER_RESTRICTED content of different owners '
                            f'{sorted(o or "<none>" for o in owners)} cannot be combined',
                            context={'owners': sorted(o or '' for o in owners),
                                     'sources': [b['source'] for b in basis]})
        owner = owners.pop()
        s.put(out, 'owner_org', owner)
    elif classification == 'PUBLIC_APPROVED':
        licenses = {b['rights'].get('license') for b in basis}
        if len(licenses) > 1:
            raise DataError(RIGHTS_CONFLICT, f'PUBLIC_APPROVED content under different licenses '
                            f'{sorted(x or "<none>" for x in licenses)} cannot be combined',
                            context={'licenses': sorted(x or '' for x in licenses),
                                     'sources': [b['source'] for b in basis]})
        s.put(out, 'license', licenses.pop())
    return out


__all__ = ['DatasetIdentity', 'create_dataset', 'load_dataset', 'list_datasets', 'combine_rights', 'RIGHTS_ORDER']
