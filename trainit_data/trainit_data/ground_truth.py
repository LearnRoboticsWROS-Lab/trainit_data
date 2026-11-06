"""Synthetic ground truth -> canonical label sets (PUBLIC, D0.7).

A synthetic session records its camera's ANNOTATOR OUTPUT with every capture: an ``instance_segmentation``
observation (``instance_image`` block: a lossless mask of compact instance ids, each id's object key and the
scene's semantic class, and the 2-D box annotator's tight box). That is evidence, sealed with the session; it is
not a label. This module turns it into CANONICAL label sets of a dataset, with ``SYNTHETIC_GROUND_TRUTH``
provenance, by one deterministic rule per kind:

==============  ==============================================================================================
INSTANCE_MASK   the image + the mask of its capture; every id of the mask gets the block's class
BBOX_2D         ``box_source ANNOTATOR`` (preferred): the annotator's tight boxes, as recorded
                (``annotator`` = the block's ``box_annotator``); ``box_source MASK_EXTENT``: each mask instance's
                pixel extent (``annotator`` = the mask's annotator, ``derivation {rule: mask_extent/1}``) -- a
                derived box is never indistinguishable from a direct one
==============  ==============================================================================================

The rule is generic: whatever produced the ``instance_image`` observations (Isaac Replicator today, Gazebo
tomorrow) is named by the session's generation block (``provenance.randomization.generator``), never branched on.
Because the rule is deterministic, :func:`rederive_problems` (``verify``) re-derives a synthetic label set from the
sessions and compares it entry for entry: an edited synthetic label is found even when every hash is consistent.

The ontology defaults to the sorted union of the classes seen; ``classes`` may give an ordered SUPERSET (e.g. the
class map shared with a real camera's labels in a mixed version). Nothing is filtered: every visible labelled
instance is a label (a filter would be a recorded rule of its own).
"""
from . import _strict as s
from ._sealed import SealedSessions
from .errors import LABEL_INVALID, DataError
from .labels import create_label_set, load_mask, read_entries
from .observation import mask_extent
from .synthetic import generation_block

KINDS = ('INSTANCE_MASK', 'BBOX_2D')
BOX_SOURCES = frozenset({'ANNOTATOR', 'MASK_EXTENT'})
MASK_EXTENT_RULE = 'mask_extent/1'


def _pairs(store, session_ids, image_stream: str, mask_stream: str):
    """``(session, capture, image observation, mask observation)`` of every capture holding both streams."""
    sessions = SealedSessions(store)
    for sid in sorted(set(session_ids)):
        session = sessions.get(sid)
        for capture in session.captures:
            by_stream = {o.stream: o for o in capture.observations}
            image, mask = by_stream.get(image_stream), by_stream.get(mask_stream)
            if image is None or mask is None:
                continue
            if mask.data.block != 'instance_image' or mask.sensor_id != image.sensor_id:
                raise DataError(LABEL_INVALID, f'{mask.observation_id} is not an instance mask of the camera of '
                                f'{image.observation_id}', context={'observation_id': mask.observation_id})
            yield session, capture, image, mask


def _generator_of(sessions) -> str:
    names = set()
    for session in sessions:
        block = session.header.provenance.get('randomization')
        if block is None:
            raise DataError(LABEL_INVALID, f'session {session.session_id} has no generation block: it holds no '
                            'synthetic ground truth', context={'session_id': session.session_id})
        names.add(generation_block(block)['generator'])
    if len(names) != 1:
        raise DataError(LABEL_INVALID, f'the sessions come from different generators {sorted(names)}: derive one '
                        'label set per generator', context={'generators': sorted(names)})
    return names.pop()


def _annotators(blocks, field: str, kind_label: str) -> str:
    names = {getattr(b, field) for b in blocks}
    if None in names or len(names) != 1:
        raise DataError(LABEL_INVALID, f'{kind_label}: the masks name {sorted(map(str, names))} as their {field}: '
                        'one annotator per label set', context={'annotators': sorted(map(str, names))})
    return names.pop()


def derive_entries(store, session_ids, kind: str, *, box_source: str = 'ANNOTATOR', image_stream: str = 'color',
                   mask_stream: str = 'instances'):
    """``(entries, provenance, classes_seen)``: the deterministic derivation of one synthetic label set."""
    kind = s.one_of(set(KINDS))(kind, 'kind')
    box_source = s.one_of(BOX_SOURCES)(box_source, 'box_source')
    pairs = list(_pairs(store, session_ids, image_stream, mask_stream))
    if not pairs:
        raise DataError(LABEL_INVALID, f'no capture of {sorted(set(session_ids))} holds both {image_stream} and '
                        f'{mask_stream}', context={'sessions': sorted(set(session_ids))})
    generator = _generator_of({p[0].session_id: p[0] for p in pairs}.values())
    blocks = [mask.data for _, _, _, mask in pairs]
    entries, seen = [], set()
    if kind == 'INSTANCE_MASK':
        annotator = _annotators(blocks, 'annotator', kind)
        provenance = {'source': 'SYNTHETIC_GROUND_TRUTH', 'generator': generator, 'annotator': annotator}
        for _, _, image, mask in pairs:
            entries.append({'observation_id': image.observation_id, 'mask_observation_id': mask.observation_id,
                            'instances': [{'id': i['id'], 'class': i['class']} for i in mask.data.instances]})
            seen.update(i['class'] for i in mask.data.instances)
        return entries, provenance, seen
    if box_source == 'ANNOTATOR':
        annotator = _annotators(blocks, 'box_annotator', f'{kind} ({box_source})')
        provenance = {'source': 'SYNTHETIC_GROUND_TRUTH', 'generator': generator, 'annotator': annotator}
    else:
        annotator = _annotators(blocks, 'annotator', f'{kind} ({box_source})')
        provenance = {'source': 'SYNTHETIC_GROUND_TRUTH', 'generator': generator, 'annotator': annotator,
                      'derivation': {'rule': MASK_EXTENT_RULE}}
    for _, _, image, mask in pairs:
        boxes = []
        decoded = None
        for inst in mask.data.instances:
            if box_source == 'ANNOTATOR':
                box = inst.get('box_tight_xywh_px')
                if box is None:
                    raise DataError(LABEL_INVALID, f'{mask.observation_id}: instance {inst["id"]} has no annotator '
                                    'box; derive the boxes from the masks (box_source MASK_EXTENT) instead',
                                    context={'observation_id': mask.observation_id, 'instance': inst['id']})
            else:
                if decoded is None:
                    decoded = load_mask(store, mask, mask.observation_id)
                    problems = mask.data.mask_problems(decoded)
                    if problems:
                        raise DataError(LABEL_INVALID, f'{mask.observation_id}: ' + '; '.join(problems),
                                        context={'observation_id': mask.observation_id})
                box = mask_extent(decoded == inst['id'])
            boxes.append({'class': inst['class'], 'bbox_xywh_px': [float(v) for v in box]})
            seen.add(inst['class'])
        entries.append({'observation_id': image.observation_id, 'boxes': boxes})
    return entries, provenance, seen


def _ontology(classes, seen):
    if classes is None:
        return sorted(seen)
    classes = list(classes)
    missing = sorted(seen - set(classes))
    if missing:
        raise DataError(LABEL_INVALID, f'classes {missing} of the ground truth are not in the given ontology {classes}',
                        context={'classes': missing})
    return classes


def create_synthetic_label_set(store, dataset: str, session_ids, kind: str, *, box_source: str = 'ANNOTATOR',
                               classes=None, image_stream: str = 'color', mask_stream: str = 'instances',
                               supersedes: str = None):
    """Derive and write one canonical ``SYNTHETIC_GROUND_TRUTH`` label set of ``dataset`` from the sessions'
    recorded annotator output; returns it (content-addressed: the same derivation is the same label set)."""
    entries, provenance, seen = derive_entries(store, session_ids, kind, box_source=box_source,
                                               image_stream=image_stream, mask_stream=mask_stream)
    provenance['source_ref'] = 'sessions ' + ' '.join(sorted(set(session_ids)))
    return create_label_set(store, dataset, kind=kind, classes=_ontology(classes, seen), provenance=provenance,
                            entries=entries, supersedes=supersedes)


def rederive_problems(store, label_set) -> list:
    """``verify``: a SYNTHETIC_GROUND_TRUTH label set with a known rule, re-derived from its sessions, must be
    exactly its entries (same images, same instances / boxes, same classes). ``[]`` for other label sets."""
    prov = label_set.provenance
    if prov['source'] != 'SYNTHETIC_GROUND_TRUTH':
        return []
    if 'annotator' not in prov or label_set.kind not in KINDS:
        return [f'a SYNTHETIC_GROUND_TRUTH {label_set.kind} label set that names no annotator / known kind cannot be '
                're-derived from its sessions']
    try:
        return _rederive(store, label_set, prov)
    except Exception as exc:                                         # noqa: BLE001 -- verify reports, never crashes
        return [f'the ground truth cannot be re-derived: {getattr(exc, "message", exc)}']


def _rederive(store, label_set, prov) -> list:
    stored = read_entries(store, label_set)
    session_ids = sorted({e['observation_id'].split('/')[0] for e in stored})
    if not stored:
        return []
    box_source = 'MASK_EXTENT' if prov.get('derivation', {}).get('rule') == MASK_EXTENT_RULE else 'ANNOTATOR'
    image_streams = {e['observation_id'].rsplit('/', 1)[1] for e in stored}
    mask_streams = {e['mask_observation_id'].rsplit('/', 1)[1] for e in stored if 'mask_observation_id' in e}
    if len(image_streams) != 1 or len(mask_streams) > 1:
        return [f'the entries span streams {sorted(image_streams | mask_streams)}: one derivation has one image and '
                'one mask stream']
    image_stream = image_streams.pop()
    mask_stream = mask_streams.pop() if mask_streams else None
    try:
        entries, provenance, _ = derive_entries(store, session_ids, label_set.kind, box_source=box_source,
                                                image_stream=image_stream,
                                                **({'mask_stream': mask_stream} if mask_stream else {}))
    except DataError as exc:
        return [f'the ground truth cannot be re-derived: {exc.message}']
    problems = []
    for key in ('generator', 'annotator', 'derivation'):
        if provenance.get(key) != prov.get(key):
            problems.append(f'provenance.{key} is {prov.get(key)!r}; the sessions give {provenance.get(key)!r}')
    derived = {e['observation_id']: e for e in entries}
    for entry in stored:
        if derived.get(entry['observation_id']) != entry:
            problems.append(f'{entry["observation_id"]}: the entry is not what the recorded ground truth gives')
    missing = sorted(set(derived) - {e['observation_id'] for e in stored})
    if missing:
        problems.append(f'{len(missing)} observation(s) of these sessions with ground truth are not labelled '
                        f'(first: {missing[0]})')
    return problems


__all__ = ['create_synthetic_label_set', 'derive_entries', 'rederive_problems', 'KINDS', 'BOX_SOURCES',
           'MASK_EXTENT_RULE']
