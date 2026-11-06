"""Label sets: annotations as a layer ABOVE acquisition (PUBLIC, D0.6).

A label set (``trainit-labels/1.0``) belongs to one dataset and lives beside it::

    datasets/<dataset_id>/labels/<label_set_id>/entries.jsonl   one line per labelled observation, sorted
    datasets/<dataset_id>/labels/<label_set_id>/labelset.json   the header, written LAST (its completeness mark)

Labels reference observations by id; they never rewrite a capture, a payload or a session. A label set is
IMMUTABLE and CONTENT-ADDRESSED: ``label_set_id = "ls-" + 16 hex`` of the sha256 of its canonical content
(everything but the id and ``created``), so importing the same labels twice is the same label set. A
correction is a NEW label set (``supersedes`` names the one it corrects) and a NEW dataset version (RELABEL).

Kinds (a CLOSED set, the core implements their rules): ``CLASS`` (one class per observation), ``BBOX_2D``
(axis-aligned pixel boxes ``[x, y, w, h]``, origin at the top-left pixel corner), ``INSTANCE_MASK`` (D0.7): an
entry names the labelled image AND the instance mask of the same capture (an ``instance_segmentation``
observation: a lossless 16-bit PNG of instance ids) and gives every id of that mask its class. The mask is checked
against the store: same capture, same size, its payload hash, and its pixel values exactly ``{0} | ids``.

Provenance ``source`` (decision D9): ``HUMAN``, ``IMPORTED``, ``SYNTHETIC_GROUND_TRUTH`` (requires its
``generator``), and ``MODEL_PREDICTION``, which is RESERVED: readable, never created in D0.6. Schema 1.1 (D0.7)
adds ``annotator`` (the synthetic annotator the labels come from) and ``derivation`` (``{rule}``: how they were
DERIVED from it, e.g. a box from a mask's extent, ``mask_extent/1``): a derived label is never indistinguishable
from a direct one. A record is written at the lowest minor that represents it (a D0.6 label set stays 1.0).

Rules checked against the store when a label set is created and again by ``verify``: every observation exists in
a SEALED session, carries an image data block, and every box lies inside its image. A box outside the image is
REFUSED, never clipped; the only tolerance is ``EDGE_TOLERANCE`` of the image size per edge, which absorbs the
rounding of a normalized 6-decimal format (YOLO) and changes no stored value. A box narrower or lower than
``MIN_RELATIVE_EXTENT`` of the image is refused too (an exchange format would round it to nothing).

Label sets are OBSERVATION-scoped (their entries name observation ids), not content-kind neutral: a future D1
episode is labelled either through the observations it references or by a new label kind of its own.
"""
import re
from dataclasses import dataclass, field

from . import _codecs
from . import _strict as s
from ._canonical import canonical_json, parse_json, pretty_json, sha256_bytes, sha256_file, sha256_json
from ._fsio import makedirs_durable, write_new_file
from ._sealed import SealedSessions, capture_of, observation_id, observation_key
from .dataset import created, created_now, dataset_id as dataset_id_check, dataset_rel, load_dataset, make_read_only
from .errors import (DATASET_INCONSISTENT, HASH_MISMATCH, INVALID_VALUE, LABEL_INVALID, LABEL_KIND_UNSUPPORTED,
                     LABEL_SET_NOT_FOUND, WRITE_FAILED, DataError)
from .reader import index_lines

SCHEMA_NAME = 'trainit-labels'
SCHEMA_MAJOR = 1
SCHEMA = 'trainit-labels/1.0'
SCHEMA_1_1 = 'trainit-labels/1.1'          # INSTANCE_MASK entries, provenance annotator / derivation (D0.7)
ENTRIES_FILE = 'entries.jsonl'
DERIVATION_RULES = frozenset({'mask_extent/1'})

KINDS = frozenset({'CLASS', 'BBOX_2D', 'INSTANCE_MASK'})
LABEL_SOURCES = frozenset({'HUMAN', 'IMPORTED', 'SYNTHETIC_GROUND_TRUTH', 'MODEL_PREDICTION'})
RESERVED_SOURCES = frozenset({'MODEL_PREDICTION'})
EDGE_TOLERANCE = 1e-6
MIN_RELATIVE_EXTENT = 1e-5                 # 10x the resolution of a 6-decimal normalized format

label_set_id = s.matching(re.compile(r'^ls-[0-9a-f]{16}$'), 'label set id (ls-<16 hex>)')


def _ontology(value, where):
    f = s.Fields(value, where)
    classes = f.req('classes', s.list_of(s.ident, min_length=1, unique=True))
    sha = f.req('sha256', s.sha256)
    f.done()
    if sha != sha256_json(list(classes)):
        s.fail(INVALID_VALUE, 'ontology.sha256 is not the sha256 of its class list', where)
    return {'classes': list(classes), 'sha256': sha}


def _derivation(value, where):
    f = s.Fields(value, where)
    out = {'rule': f.req('rule', s.one_of(DERIVATION_RULES))}
    f.done()
    return out


def ontology_of(classes) -> dict:
    return {'classes': list(classes), 'sha256': sha256_json(list(classes))}


def _provenance(value, where):
    f = s.Fields(value, where)
    out = {'source': f.req('source', s.one_of(LABEL_SOURCES))}
    for name in ('tool', 'author', 'generator', 'notes', 'source_ref'):
        s.put(out, name, f.opt(name, s.string))
    s.put(out, 'annotator', f.opt('annotator', s.token))
    derivation = f.opt('derivation', _derivation)
    if derivation is not None:
        out['derivation'] = derivation
    f.done()
    if 'derivation' in out and 'annotator' not in out:
        s.fail(INVALID_VALUE, 'a derivation names the annotator it derives from', where)
    if out['source'] == 'SYNTHETIC_GROUND_TRUTH' and 'generator' not in out:
        s.fail(INVALID_VALUE, 'SYNTHETIC_GROUND_TRUTH labels name their generator', where)
    return out


def _entries_ref(value, where):
    f = s.Fields(value, where)
    out = {'path': f.req('path', s.one_of({ENTRIES_FILE})), 'sha256': f.req('sha256', s.sha256),
           'count': f.req('count', s.integer(0))}
    f.done()
    return out


# ---- entries ---------------------------------------------------------------------------------------------

def _class_in(classes):
    def check(value, where):
        value = s.string(value, where)
        if value not in classes:
            s.fail(LABEL_INVALID, f'class {value!r} is not in the ontology {list(classes)}', where)
        return value
    return check


def _box(classes):
    def check(value, where):
        f = s.Fields(value, where)
        box = {'class': f.req('class', _class_in(classes)),
               'bbox_xywh_px': list(f.req('bbox_xywh_px', s.vector(4)))}
        f.done()
        x, y, w, h = box['bbox_xywh_px']
        if w <= 0.0 or h <= 0.0:
            s.fail(LABEL_INVALID, f'a box has a positive width and height, not {w} x {h}', where)
        return box
    return check


def _class_entry(classes):
    def check(value, where):
        f = s.Fields(value, where)
        entry = {'observation_id': f.req('observation_id', observation_id),
                 'class': f.req('class', _class_in(classes))}
        f.done()
        return entry
    return check


def _bbox_entry(classes):
    def check(value, where):
        f = s.Fields(value, where)
        entry = {'observation_id': f.req('observation_id', observation_id),
                 'boxes': [dict(b) for b in f.req('boxes', s.list_of(_box(classes)))]}
        f.done()
        return entry
    return check


def _mask_instance(classes):
    def check(value, where):
        f = s.Fields(value, where)
        out = {'id': f.req('id', s.integer(1)), 'class': f.req('class', _class_in(classes))}
        f.done()
        return out
    return check


def _mask_entry(classes):
    def check(value, where):
        f = s.Fields(value, where)
        entry = {'observation_id': f.req('observation_id', observation_id),
                 'mask_observation_id': f.req('mask_observation_id', observation_id),
                 'instances': [dict(i) for i in f.req('instances', s.list_of(_mask_instance(classes)))]}
        f.done()
        if capture_of(entry['observation_id']) != capture_of(entry['mask_observation_id']):
            s.fail(LABEL_INVALID, 'the mask belongs to the capture of the labelled image', where)
        if observation_key(entry['observation_id'])[2] != observation_key(entry['mask_observation_id'])[2]:
            s.fail(LABEL_INVALID, 'the mask belongs to the camera of the labelled image', where)
        if entry['mask_observation_id'] == entry['observation_id']:
            s.fail(LABEL_INVALID, 'the mask is another observation than the labelled image', where)
        ids = [i['id'] for i in entry['instances']]
        if ids != sorted(ids) or len(set(ids)) != len(ids):
            s.fail(LABEL_INVALID, 'instance ids are unique and sorted', where)
        return entry
    return check


# The entry readers this core implements. A representable kind without one is LABEL_KIND_UNSUPPORTED.
_ENTRY_READERS = {'CLASS': _class_entry, 'BBOX_2D': _bbox_entry, 'INSTANCE_MASK': _mask_entry}


def entry_reader(kind: str, classes, where: str):
    reader = _ENTRY_READERS.get(kind)
    if reader is None:
        raise DataError(LABEL_KIND_UNSUPPORTED, f'{where}: label kind {kind} is representable but has no reader '
                        'in this core', context={'where': where, 'kind': kind})
    return reader(tuple(classes))


def box_inside(bbox, width: int, height: int) -> bool:
    """``[x, y, w, h]`` lies inside a ``width`` x ``height`` image (``EDGE_TOLERANCE`` of the size per edge)."""
    x, y, w, h = bbox
    tx, ty = EDGE_TOLERANCE * width, EDGE_TOLERANCE * height
    return x >= -tx and y >= -ty and x + w <= width + tx and y + h <= height + ty


def box_large_enough(bbox, width: int, height: int) -> bool:
    """At least ``MIN_RELATIVE_EXTENT`` of the image per side: a box every exchange format can still represent
    (a normalized 6-decimal format rounds a smaller one to zero)."""
    return bbox[2] >= MIN_RELATIVE_EXTENT * width and bbox[3] >= MIN_RELATIVE_EXTENT * height


def check_entry_against(entry: dict, observation, where: str):
    """The rules that need the observation: an image data block, and every box inside the image."""
    data = observation.data
    if getattr(data, 'block', None) != 'image':
        s.fail(LABEL_INVALID, f'{observation.observation_id} has no image data block: it cannot carry image '
               'labels', where)
    for i, box in enumerate(entry.get('boxes', ())):
        if not box_inside(box['bbox_xywh_px'], data.width, data.height):
            s.fail(LABEL_INVALID, f'box {box["bbox_xywh_px"]} lies outside the {data.width}x{data.height} image of '
                   f'{observation.observation_id} (boxes are refused, never clipped)', f'{where}.boxes[{i}]')
        if not box_large_enough(box['bbox_xywh_px'], data.width, data.height):
            s.fail(LABEL_INVALID, f'box {box["bbox_xywh_px"]} is smaller than {MIN_RELATIVE_EXTENT} of the '
                   f'{data.width}x{data.height} image per side', f'{where}.boxes[{i}]')


def load_mask(store, mask_obs, where: str):
    """The decoded instance mask of an ``instance_image`` observation, its payload hash checked."""
    if getattr(mask_obs.data, 'block', None) != 'instance_image':
        s.fail(LABEL_INVALID, f'{mask_obs.observation_id} is not an instance mask (an instance_image observation)',
               where)
    ref = mask_obs.payload
    path = store.path(ref.path)
    data = path.read_bytes() if path.is_file() else None
    if data is None or sha256_bytes(data) != ref.sha256:
        raise DataError(HASH_MISMATCH, f'{where}: the mask payload {ref.path} is missing or does not hash to the '
                        f'recorded {ref.sha256}', context={'where': where, 'path': ref.path})
    return _codecs.decode(data, ref.media_type, where=where)


def check_mask_against(entry: dict, observation, mask_obs, store, where: str):
    """An INSTANCE_MASK entry against the store: the mask's block and size, every id of the mask given a class
    (exactly the mask's ids), and the decoded pixels exactly what the block says."""
    mask = load_mask(store, mask_obs, f'{where}.mask_observation_id')
    block = mask_obs.data
    data = observation.data
    if (block.width, block.height) != (data.width, data.height):
        s.fail(LABEL_INVALID, f'the {block.width}x{block.height} mask does not match the {data.width}x{data.height} '
               'image', where)
    listed = [i['id'] for i in entry['instances']]
    if listed != [i['id'] for i in block.instances]:
        s.fail(LABEL_INVALID, f'the entry classifies ids {listed}; the mask holds {[i["id"] for i in block.instances]}',
               where)
    problems = block.mask_problems(mask)
    if problems:
        s.fail(LABEL_INVALID, '; '.join(problems), where)


def check_entry_in_store(label_set_kind: str, entry: dict, sessions, store, where: str):
    """Every rule of an entry that needs the store (``create_label_set`` and ``verify``)."""
    _, obs = sessions.observation(entry['observation_id'], where)
    check_entry_against(entry, obs, where)
    if label_set_kind == 'INSTANCE_MASK':
        _, mask_obs = sessions.observation(entry['mask_observation_id'], where)
        check_mask_against(entry, obs, mask_obs, store, where)


# ---- the record ------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class LabelSet:
    schema: str
    label_set_id: str
    dataset_id: str
    kind: str
    ontology: dict
    provenance: dict
    entries: dict                        # {path, sha256, count}
    created: dict
    supersedes: str = None
    extensions: dict = field(default_factory=dict)

    @classmethod
    def from_dict(cls, value, where='labelset') -> 'LabelSet':
        f = s.Fields(value, where)
        record = cls(schema=f.req('schema', s.schema(SCHEMA_NAME, SCHEMA_MAJOR)),
                     label_set_id=f.req('label_set_id', label_set_id),
                     dataset_id=f.req('dataset_id', dataset_id_check),
                     kind=f.req('kind', s.one_of(KINDS)),
                     ontology=f.req('ontology', _ontology),
                     provenance=f.req('provenance', _provenance),
                     entries=f.req('entries', _entries_ref),
                     created=f.req('created', created),
                     supersedes=f.opt('supersedes', label_set_id),
                     extensions=f.extensions())
        f.done()
        if record.label_set_id != content_id(record.content()):
            s.fail(HASH_MISMATCH, f'{record.label_set_id} is not the id of this content '
                   f'({content_id(record.content())})', where)
        if record.supersedes == record.label_set_id:
            s.fail(INVALID_VALUE, 'a label set cannot supersede itself', where)
        if record.schema == SCHEMA and (record.kind == 'INSTANCE_MASK' or 'annotator' in record.provenance):
            s.fail(INVALID_VALUE, f'INSTANCE_MASK and provenance.annotator / derivation need {SCHEMA_1_1}', where)
        return record

    def content(self) -> dict:
        """Everything that identifies the label set: the record without its id and ``created``."""
        out = {'schema': self.schema, 'dataset_id': self.dataset_id, 'kind': self.kind,
               'ontology': dict(self.ontology), 'provenance': dict(self.provenance), 'entries': dict(self.entries)}
        s.put(out, 'supersedes', self.supersedes)
        if self.extensions:
            out['extensions'] = dict(self.extensions)
        return out

    def to_dict(self) -> dict:
        out = self.content()
        out['label_set_id'] = self.label_set_id
        out['created'] = dict(self.created)
        return out

    @property
    def sha256(self) -> str:
        """The pin a dataset version records: the sha256 of the whole canonical header."""
        return sha256_json(self.to_dict())

    @property
    def classes(self) -> tuple:
        return tuple(self.ontology['classes'])


def content_id(content: dict) -> str:
    return 'ls-' + sha256_json(content)[len('sha256:'):][:16]


def labels_rel(dataset: str, ls_id: str = None) -> str:
    base = f'{dataset_rel(dataset)}/labels'
    return base if ls_id is None else f'{base}/{label_set_id(ls_id, "label_set_id")}'


# ---- create / load ---------------------------------------------------------------------------------------

def create_label_set(store, dataset: str, *, kind: str, classes, provenance: dict, entries, supersedes=None,
                     extensions=None) -> LabelSet:
    """Validate ``entries`` (dicts in the entry form of ``kind``) against the store and write an immutable label
    set; returns it. The same content again returns the existing label set (content-addressed)."""
    load_dataset(store, dataset)                                      # DATASET_NOT_FOUND
    where = f'{labels_rel(dataset)}/<new>'
    kind = s.one_of(KINDS)(kind, f'{where}.kind')
    classes = s.list_of(s.ident, min_length=1, unique=True)(list(classes), f'{where}.ontology.classes')
    provenance = _provenance(provenance, f'{where}.provenance')
    if provenance['source'] in RESERVED_SOURCES:
        s.fail(INVALID_VALUE, f'label source {provenance["source"]} is reserved: no producer exists in D0.6',
               f'{where}.provenance.source')
    if provenance['source'] == 'SYNTHETIC_GROUND_TRUTH' and 'annotator' not in provenance:
        # D0.7: synthetic ground truth names the annotator it comes from, so verify can re-derive it from the sessions
        s.fail(INVALID_VALUE, 'SYNTHETIC_GROUND_TRUTH labels name their annotator (provenance.annotator)',
               f'{where}.provenance')
    reader = entry_reader(kind, classes, f'{where}.kind')
    parsed = [reader(e, f'{where}.entries[{i}]') for i, e in enumerate(entries)]
    parsed.sort(key=lambda e: observation_key(e['observation_id']))
    for a, b in zip(parsed, parsed[1:]):
        if a['observation_id'] == b['observation_id']:
            s.fail(LABEL_INVALID, f'{a["observation_id"]} is labelled twice', f'{where}.entries')
    sessions = SealedSessions(store)
    for entry in parsed:
        check_entry_in_store(kind, entry, sessions, store, f'{where}.entries[{entry["observation_id"]}]')
    if supersedes is not None:
        previous = load_label_set(store, dataset, supersedes)
        if previous.kind != kind:
            s.fail(INVALID_VALUE, f'{supersedes} is a {previous.kind} label set; a {kind} set cannot supersede it',
                   f'{where}.supersedes')
    data = b''.join(canonical_json(e) + b'\n' for e in parsed)
    newer = kind == 'INSTANCE_MASK' or 'annotator' in provenance or 'derivation' in provenance
    record = {'schema': SCHEMA_1_1 if newer else SCHEMA, 'dataset_id': dataset, 'kind': kind,
              'ontology': ontology_of(classes),
              'provenance': provenance, 'entries': {'path': ENTRIES_FILE, 'sha256': sha256_bytes(data),
                                                    'count': len(parsed)}}
    s.put(record, 'supersedes', supersedes)
    if extensions:
        record['extensions'] = extensions
    ls_id = content_id(record)
    record.update(label_set_id=ls_id, created=created_now())
    label_set = LabelSet.from_dict(record, f'{labels_rel(dataset)}/{ls_id}')    # one validation path
    directory = store.path(labels_rel(dataset, ls_id))
    header = directory / 'labelset.json'
    if header.is_file():
        return load_label_set(store, dataset, ls_id)                  # the same content: the same label set
    entries_path = directory / ENTRIES_FILE
    resumed = directory.exists()
    if resumed:
        # An interrupted creation of THIS content (the id is its hash). It is completed when what it left is exactly
        # the entries this call writes; anything else is refused, never guessed about or deleted.
        left = sorted(p.name for p in directory.iterdir())
        if left != [ENTRIES_FILE] or entries_path.read_bytes() != data:
            raise DataError(DATASET_INCONSISTENT, f'{labels_rel(dataset, ls_id)} holds {left} from an interrupted '
                            'creation that cannot be completed safely; move that directory out of the store (nothing '
                            'references it; verify reports it) and import again',
                            context={'path': labels_rel(dataset, ls_id), 'left': left})
    try:
        if not resumed:
            makedirs_durable(directory)
            write_new_file(entries_path, data)
        make_read_only(entries_path)
        write_new_file(header, pretty_json(label_set.to_dict()))
        make_read_only(header)
    except OSError as exc:
        raise DataError(WRITE_FAILED, f'writing {labels_rel(dataset, ls_id)} failed', cause=exc,
                        context={'path': labels_rel(dataset, ls_id)}) from None
    return label_set


def load_label_set(store, dataset: str, ls_id: str) -> LabelSet:
    rel = f'{labels_rel(dataset, ls_id)}/labelset.json'
    path = store.path(rel)
    if not path.is_file():
        raise DataError(LABEL_SET_NOT_FOUND, f'dataset {dataset} has no label set {ls_id}',
                        context={'dataset_id': dataset, 'label_set_id': ls_id})
    record = LabelSet.from_dict(parse_json(path.read_bytes(), rel), rel)
    if record.label_set_id != ls_id or record.dataset_id != dataset:
        s.fail(DATASET_INCONSISTENT, f'the header names {record.dataset_id}/{record.label_set_id}', rel)
    return record


def list_label_sets(store, dataset: str) -> list:
    root = store.path(labels_rel(dataset))
    if not root.is_dir():
        return []
    return sorted(p.name for p in root.iterdir() if (p / 'labelset.json').is_file())


def read_entries(store, label_set: LabelSet) -> list:
    """The entries, parsed; ``HASH_MISMATCH`` if the file is not the recorded bytes."""
    rel = f'{labels_rel(label_set.dataset_id, label_set.label_set_id)}/{ENTRIES_FILE}'
    path = store.path(rel)
    actual, _ = sha256_file(path) if path.is_file() else (None, 0)
    if actual != label_set.entries['sha256']:
        raise DataError(HASH_MISMATCH, f'{rel} does not hash to the recorded {label_set.entries["sha256"]}',
                        context={'path': rel, 'expected': label_set.entries['sha256'], 'actual': actual})
    reader = entry_reader(label_set.kind, label_set.classes, rel)
    entries = []
    for number, _, raw, complete in index_lines(path):
        where = f'{rel}:{number}'
        if not complete:
            s.fail(DATASET_INCONSISTENT, 'the entries end with an incomplete line', where)
        entry = reader(parse_json(raw, where), where)
        if canonical_json(entry) != raw:
            s.fail(DATASET_INCONSISTENT, 'an entry line is not in canonical form', where)
        entries.append(entry)
    if len(entries) != label_set.entries['count']:
        s.fail(DATASET_INCONSISTENT, f'{len(entries)} entries, the header records {label_set.entries["count"]}', rel)
    keys = [observation_key(e['observation_id']) for e in entries]
    if keys != sorted(keys) or len(set(keys)) != len(keys):
        s.fail(DATASET_INCONSISTENT, 'entries are not sorted by observation, or one is labelled twice', rel)
    return entries


__all__ = ['LabelSet', 'create_label_set', 'load_label_set', 'list_label_sets', 'read_entries', 'box_inside',
           'check_entry_in_store', 'load_mask', 'KINDS', 'LABEL_SOURCES', 'EDGE_TOLERANCE', 'DERIVATION_RULES']
