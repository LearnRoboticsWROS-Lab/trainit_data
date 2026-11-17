"""YOLO detection and segmentation: the exporter (D0.6; segmentation M4.5) and the detection label importer (PUBLIC).

EXPORT (``yolo`` 1.1) of one dataset version, task ``detect`` (BBOX_2D label sets) or ``segment`` (INSTANCE_MASK
label sets)::

    images/<split>/<stem>.png     the observation's canonical payload, byte for byte (rgb8 PNG)
    labels/<split>/<stem>.txt     detect:  one line per box: ``<class> <cx> <cy> <w> <h>``, normalized, ``%.6f``
                                  segment: one line per instance: ``<class> <x1> <y1> <x2> <y2> ...`` (>= 3
                                  vertices), normalized, ``%.6f``
    data.yaml                     the splits as RELATIVE directories + the class names (no ``path`` key: a
                                  consumer resolves them against the directory of data.yaml)
    trainit_export.json           the sidecar (``trainit_data.exports``)

``<stem>`` is the observation id with ``/`` replaced by ``__`` (``<session_id>__<seq>__<sensor_id>__<stream>``):
unique, stable and invertible, because no part of an observation id contains ``__``. The exported images are
exactly the LABELLED observations of the version's selection (an entry with no box / no instance is an explicit
negative and exports an empty label file); an unlabelled observation is not exported, never turned into a silent
negative. The split of an image is its capture's group's split in the version (``versions.capture_splits``: a
session, or a randomization run); the exporter assigns nothing and never looks at a source. Everything is sorted
and carries no timestamp: two exports of the same version and config are byte-identical.

SEGMENT (M4.5): each INSTANCE_MASK entry names the image and the instance mask of its capture (a 16-bit PNG of
compact ids) and gives every id its class. The mask payload is read (its sha256 checked) and every instance becomes
ONE polygon by the deterministic policy of ``formats._polygons`` (``POLICY``): the largest 8-connected component
of ``mask == id``, its outer boundary (holes ignored), Douglas-Peucker with ``epsilon_px`` (default 0.5), vertices
at the PIXEL CENTRES of boundary pixels in the corner-origin frame (``(col + 0.5) / W``, ``(row + 0.5) / H``). An
instance whose kept component has fewer than ``min_area_px`` pixels (default 16), or whose polygon is degenerate
(fewer than 3 vertices or zero area), is DROPPED and recorded; so are the pixels of the smaller components. An image
whose every instance is dropped exports an empty label file. ``summary.polygons`` records the policy, its parameters
and these statistics; ``epsilon_px`` / ``min_area_px`` are part of the config (and so of the export id). A mask
pixel id that the entry does not classify is refused (``EXPORT_REFUSED``), never guessed.

Refused (``EXPORT_REFUSED``), never approximated: no label set of the task's kind, label sets of different
ontologies, mixed label provenance unless ``allow_mixed_provenance`` (then recorded in the config), a labelled
observation that is not an ``rgb8`` PNG, and the segment options with task ``detect``.

IMPORT (:func:`import_yolo_labels`): a directory of ``<stem>.txt`` files (searched recursively; other files are
ignored, so an export's own directory imports as is) + the class names in index order -> ONE canonical BBOX_2D
label set (``IMPORTED`` or ``HUMAN``). Every line is validated strictly: exactly five fields, an integer class in
range, finite decimal numbers in [0, 1], a positive size, the box inside the image (``EDGE_TOLERANCE`` per edge
for the format's 6-decimal rounding). Any bad line refuses the WHOLE import naming ``file:line``
(``LABEL_INVALID``); nothing is clipped, nothing is skipped. Segmentation lines (more than five fields) are
refused: importing segmentation labels is not implemented. No Ultralytics import, anywhere.
"""
import math
import re
from pathlib import Path

import numpy as np

from .. import _strict as s
from .._sealed import SealedSessions, capture_of, observation_id as observation_id_check, observation_key
from ..errors import (CONTENT_KIND_UNSUPPORTED, DATASET_INCONSISTENT, EXPORT_REFUSED, INVALID_VALUE, LABEL_INVALID,
                      DataError)
from ..exports import ExportPlan, PlannedFile, export_id_of, prepare_destination, write_export
from ..labels import EDGE_TOLERANCE, create_label_set, label_set_id, load_mask
from ..splits import SPLITS
from ..versions import capture_splits, labelled_observations, load_split, load_version
from . import register_exporter
from ._polygons import POLICY as POLYGON_POLICY, instance_polygon

NAME = 'yolo'
VERSION = '1.1'                    # 1.1 (M4.5): task segment
TASKS = frozenset({'detect', 'segment'})
LABEL_KIND = {'detect': 'BBOX_2D', 'segment': 'INSTANCE_MASK'}
DEFAULT_EPSILON_PX = 0.5         # measured: IoU >= 0.95 on discs of radius >= 10 px (1.0: 0.92)
DEFAULT_MIN_AREA_PX = 16
SEGMENT_OPTIONS = ('epsilon_px', 'min_area_px')
IMPORT_SOURCES = frozenset({'IMPORTED', 'HUMAN'})
DATA_YAML = 'data.yaml'
_INT_RE = re.compile(r'^(?:0|[1-9][0-9]*)$')
_LINE_RE = re.compile(r'^[ \t]*[^ \t\r]+(?:[ \t]+[^ \t\r]+)*[ \t]*$')
_FIELD_SEP_RE = re.compile(r'[ \t]+')
_DEC_RE = re.compile(r'^[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?$')
_LABEL_PATH_RE = re.compile(r'^labels/(train|val|test)/([^/]+)\.txt$')
_IMAGE_PATH_RE = re.compile(r'^images/(train|val|test)/([^/]+)\.png$')


def stem_of(observation_id: str) -> str:
    return observation_id.replace('/', '__')


def observation_of(stem: str, where: str) -> str:
    parts = stem.split('__')
    if len(parts) != 4:
        s.fail(LABEL_INVALID, f'{stem!r} is not an exported image name (<session_id>__<seq>__<sensor_id>__<stream>)',
               where)
    return observation_id_check('/'.join(parts), where)


def _fmt(value: float) -> str:
    text = f'{value + 0.0:.6f}'
    return '0.000000' if text == '-0.000000' else text


def label_lines(boxes, classes, width: int, height: int) -> bytes:
    lines = []
    for box in boxes:
        x, y, w, h = box['bbox_xywh_px']
        lines.append(f'{classes.index(box["class"])} {_fmt((x + w / 2) / width)} {_fmt((y + h / 2) / height)} '
                     f'{_fmt(w / width)} {_fmt(h / height)}\n')
    return ''.join(lines).encode('ascii')


def segment_label_lines(mask, instances, classes, width: int, height: int, epsilon_px: float,
                        min_area_px: int, where: str):
    """``(bytes, polygons, records)`` of one INSTANCE_MASK entry: the label file (one polygon line per kept
    instance, in id order), the number of polygons, and one record per instance ``{id, class, components, pixels,
    dropped_pixels, vertices | dropped}``. ``instances``: the entry's ``[{id, class}]``; a mask id it does not
    classify is refused (``EXPORT_REFUSED``)."""
    if mask.ndim != 2 or mask.shape != (height, width):
        raise DataError(DATASET_INCONSISTENT, f'{where}: the instance mask is {mask.shape[::-1]}, the image is '
                        f'{width}x{height}', context={'where': where})
    by_id = {i['id']: i['class'] for i in instances}
    present = sorted(int(v) for v in np.unique(mask) if v != 0)
    unclassified = [i for i in present if i not in by_id]
    if unclassified:
        raise DataError(EXPORT_REFUSED, f'{where}: mask ids {unclassified} have no class in the label set (the '
                        f'entry classifies {sorted(by_id)}): an instance is never exported without its class',
                        context={'where': where, 'ids': unclassified})
    lines, records = [], []
    for iid in sorted(by_id):
        vertices, info = instance_polygon(mask == iid, epsilon_px)
        record = dict(info, id=iid, **{'class': by_id[iid]})
        if info['pixels'] == 0:
            record['dropped'] = 'EMPTY'
        elif info['pixels'] < min_area_px:
            record['dropped'] = 'BELOW_MIN_AREA'
        elif vertices is None:
            record['dropped'] = 'DEGENERATE'
        else:
            record['vertices'] = len(vertices)
            coords = ' '.join(f'{_fmt((c + 0.5) / width)} {_fmt((r + 0.5) / height)}' for c, r in vertices)
            lines.append(f'{classes.index(by_id[iid])} {coords}\n')
        records.append(record)
    return ''.join(lines).encode('ascii'), len(lines), records


def data_yaml(export_id: str, source: dict, classes, splits_present, rights: dict, task: str = 'detect') -> bytes:
    """Deterministic: generated from the sidecar's content only (``verify`` regenerates and compares it)."""
    kind = 'detection' if task == 'detect' else 'segmentation'
    lines = [f'# TrainIt YOLO {kind} export {export_id}: dataset {source["dataset_id"]} version '
             f'{source["version"]} ({source["content_hash"]}).',
             f'# Rights: {rights["classification"]}. Provenance and every file\'s sha256: trainit_export.json.',
             '# Paths are relative to the directory of this file.']
    lines += [f'{name}: images/{name}' for name in SPLITS if name in splits_present]
    lines.append(f'nc: {len(classes)}')
    lines.append('names:')
    lines += [f"  {i}: '{name}'" for i, name in enumerate(classes)]      # quoted: a class named `true` stays a name
    return ('\n'.join(lines) + '\n').encode('utf-8')


def _epsilon(value, where):
    value = s.real(value, where)
    if value < 0.0:
        s.fail(INVALID_VALUE, 'must be >= 0', where)
    return value


def _config(value, where='config'):
    """The canonical config. Task ``detect`` keeps the D0.6 form exactly (no segment option is accepted); task
    ``segment`` always records its options, defaults resolved."""
    f = s.Fields(value, where)
    out = {'task': f.opt('task', s.one_of(TASKS), 'detect'),
           'label_sets': list(f.opt('label_sets', s.list_of(label_set_id, unique=True), ())),
           'allow_mixed_provenance': f.opt('allow_mixed_provenance', s.boolean, False)}
    epsilon = f.opt('epsilon_px', _epsilon)
    min_area = f.opt('min_area_px', s.integer(1))
    f.done()
    if out['task'] == 'segment':
        out['epsilon_px'] = DEFAULT_EPSILON_PX if epsilon is None else epsilon
        out['min_area_px'] = DEFAULT_MIN_AREA_PX if min_area is None else min_area
    elif epsilon is not None or min_area is not None:
        raise DataError(EXPORT_REFUSED, f'{list(SEGMENT_OPTIONS)} are options of task segment, not detect',
                        context={'task': out['task']})
    return out


class YoloDetectionExporter:
    """The ``yolo`` exporter: task ``detect`` (BBOX_2D) and, since 1.1, task ``segment`` (INSTANCE_MASK). The class
    keeps its D0.6 name (public API)."""
    name = NAME
    version = VERSION
    content_kinds = frozenset({'OBSERVATIONS'})
    label_kinds = frozenset(LABEL_KIND.values())

    def export(self, store, dataset: str, version: int, destination, config: dict = None) -> dict:
        """Export ``dataset`` version ``version`` to ``destination``; returns the sidecar."""
        prepare_destination(destination, store)                      # refuse a bad destination before any work
        return write_export(store, destination, self.plan(store, dataset, version, config))

    def plan(self, store, dataset: str, version: int, config: dict = None) -> ExportPlan:
        """Everything the export writes, decided from the store alone (deterministic; nothing is read but the
        records and the label sets: payload bytes are copied, and checked, by the writer)."""
        record = load_version(store, dataset, version)
        if record.content_kind not in self.content_kinds:
            raise DataError(CONTENT_KIND_UNSUPPORTED, f'the {NAME} exporter reads {sorted(self.content_kinds)}, not '
                            f'{record.content_kind}', context={'content_kind': record.content_kind})
        cfg = _config(dict(config or {}))
        task = cfg['task']
        kind = LABEL_KIND[task]
        refs = {r['label_set_id']: r for r in record.content['label_sets']}
        chosen = sorted(cfg['label_sets']) or sorted(i for i, r in refs.items() if r['kind'] == kind)
        for ls_id in chosen:
            if ls_id not in refs or refs[ls_id]['kind'] != kind:
                raise DataError(EXPORT_REFUSED, f'{ls_id} is not a {kind} label set of {dataset} version {version} '
                                f'(task {task} exports {kind})', context={'label_set_id': ls_id})
        if not chosen:
            raise DataError(EXPORT_REFUSED, f'{dataset} version {version} has no {kind} label set to export '
                            f'(task {task})', context={'dataset_id': dataset, 'version': version, 'task': task})
        sources = sorted({refs[i]['source'] for i in chosen})
        if len(sources) > 1 and not cfg['allow_mixed_provenance']:
            raise DataError(EXPORT_REFUSED, f'label sets of mixed provenance {sources}: set allow_mixed_provenance to '
                            'export them together (it is then recorded)', context={'sources': sources})
        cfg['label_sets'] = chosen
        split = load_split(store, record)
        splits = capture_splits(store, record, split)
        labelled = labelled_observations(store, record, chosen)
        if not labelled:
            raise DataError(EXPORT_REFUSED, f'no observation of {dataset} version {version}\'s selection is labelled '
                            f'by {chosen}: there is nothing to export', context={'label_sets': chosen})
        classes = _classes_of(store, dataset, chosen)
        pins = {x['session_id']: x['index_sha256'] for x in record.content['sessions']}
        sessions = SealedSessions(store)
        counted = 'boxes' if task == 'detect' else 'instances'
        files, summary = [], {name: {'images': 0, counted: 0} for name in SPLITS}
        stats = _PolygonStats(cfg) if task == 'segment' else None
        for obs_id, items in labelled.items():
            if len(items) != 1:
                raise DataError(DATASET_INCONSISTENT, f'{obs_id} is labelled by {len(items)} {kind} label sets',
                                context={'observation_id': obs_id})
            entry = items[0][1]
            session_id = observation_key(obs_id)[0]
            session = sessions.get(session_id)
            capture_id = capture_of(obs_id)
            if session_id not in pins or session.header.index['sha256'] != pins[session_id] \
                    or capture_id not in splits:
                raise DataError(DATASET_INCONSISTENT, f'session {session_id} is not the one the version pins and '
                                'splits', context={'session_id': session_id})
            _, obs = session.observation(obs_id, f'{dataset}/{version}')
            image = getattr(obs.data, 'block', None) == 'image'
            if not image or obs.data.encoding != 'rgb8' or obs.payload.media_type != 'image/png':
                what = f'{obs.data.encoding} ({obs.payload.media_type})' if image else f'a {obs.data.block!r} block'
                raise DataError(EXPORT_REFUSED, f'{obs_id} is {what}: YOLO '
                                'exports rgb8 PNG images only', context={'observation_id': obs_id})
            name = splits[capture_id]
            stem = stem_of(obs_id)
            if task == 'detect':
                data, count = label_lines(entry['boxes'], classes, obs.data.width, obs.data.height), len(entry['boxes'])
            else:
                where = f'{dataset}/{version}: {obs_id}'
                _, mask_obs = session.observation(entry['mask_observation_id'], where)
                mask = load_mask(store, mask_obs, where)
                data, count, records = segment_label_lines(mask, entry['instances'], classes, obs.data.width,
                                                           obs.data.height, cfg['epsilon_px'], cfg['min_area_px'],
                                                           where)
                stats.add(obs_id, records)
            files.append(PlannedFile(f'images/{name}/{stem}.png', obs.payload.sha256, obs.payload.bytes,
                                     source=obs.payload.path, link={'observation_id': obs_id}))
            files.append(PlannedFile.generated(f'labels/{name}/{stem}.txt', data, {'label_of': obs_id}))
            summary[name]['images'] += 1
            summary[name][counted] += count
        source = {'dataset_id': dataset, 'version': version, 'content_kind': record.content_kind,
                  'content_hash': record.content_hash,
                  'label_sets': [{'label_set_id': i, 'sha256': refs[i]['sha256'], 'source': refs[i]['source']}
                                 for i in chosen],
                  'split': {'split_id': split.split_id, 'sha256': record.content['split']['sha256']}}
        exporter = {'name': NAME, 'version': VERSION, 'config': cfg}
        present = [n for n in SPLITS if summary[n]['images']]
        files.append(PlannedFile.generated(DATA_YAML, data_yaml(export_id_of(exporter, source), source, classes,
                                                                present, record.rights, task)))
        out = {'task': task, 'classes': classes, 'splits': summary}
        if stats is not None:
            out['polygons'] = stats.to_dict()
        return ExportPlan(exporter=exporter, source=source, rights=record.rights, summary=out, files=tuple(files))

    def verify(self, dest: Path, sidecar: dict) -> list:
        """The format's own rules, on the files as they are (hashes are checked by ``verify_export``)."""
        from ..integrity import Finding
        findings = []
        summary = sidecar['summary']
        try:
            classes = list(s.list_of(s.ident, min_length=1, unique=True)(summary.get('classes'), 'summary.classes'))
            counts = summary['splits']
            task = s.one_of(TASKS)(summary['task'], 'summary.task')
            if task != sidecar['exporter']['config'].get('task', 'detect'):
                s.fail(DATASET_INCONSISTENT, 'the summary and the config name different tasks', 'summary.task')
            polygons = _check_polygon_summary(summary, counts) if task == 'segment' else None
        except (DataError, KeyError, TypeError, AttributeError) as exc:
            return [Finding(DATASET_INCONSISTENT, 'ERROR', 'trainit_export.json',
                            f'the YOLO summary is invalid: {exc}')]
        counted, parse = ('boxes', parse_label_file) if task == 'detect' else ('instances', parse_segment_file)
        listed = {x['path']: x for x in sidecar['files']}
        images, labels = {}, {}
        for rel, entry in listed.items():
            if rel == DATA_YAML:
                continue
            m_img, m_lbl = _IMAGE_PATH_RE.match(rel), _LABEL_PATH_RE.match(rel)
            if m_img:
                images[(m_img.group(1), m_img.group(2))] = entry
            elif m_lbl:
                labels[(m_lbl.group(1), m_lbl.group(2))] = entry
            else:
                findings.append(Finding(DATASET_INCONSISTENT, 'ERROR', rel, 'not a YOLO export path'))
        for key in sorted(set(images) ^ set(labels)):
            findings.append(Finding(DATASET_INCONSISTENT, 'ERROR', f'{key[0]}/{key[1]}',
                                    'an image without its label file, or a label file without its image'))
        boxes = {name: 0 for name in SPLITS}                  # boxes or polygons
        for (name, stem), entry in sorted(labels.items()):
            rel = entry['path']
            path = dest.joinpath(*rel.split('/'))
            if not path.is_file():
                continue                                 # reported by verify_export
            try:
                parsed = parse(path.read_bytes(), len(classes), rel)
            except DataError as exc:
                findings.append(Finding(exc.code, 'ERROR', rel, exc.message))
                continue
            boxes[name] += len(parsed)
            obs = images.get((name, stem), {}).get('observation_id')
            if entry.get('label_of') is None or obs != entry['label_of'] or stem != stem_of(entry['label_of']):
                findings.append(Finding(DATASET_INCONSISTENT, 'ERROR', rel, 'the label file, its image and its '
                                        'observation do not name the same observation'))
        for name in SPLITS:
            n_images = sum(1 for (split, _) in images if split == name)
            if counts.get(name) != {'images': n_images, counted: boxes[name]}:
                findings.append(Finding(DATASET_INCONSISTENT, 'ERROR', 'trainit_export.json',
                                        f'summary.splits.{name} = {counts.get(name)}, the files hold {n_images} '
                                        f'images and {boxes[name]} {counted}'))
        if polygons is not None and polygons != sum(boxes.values()):
            findings.append(Finding(DATASET_INCONSISTENT, 'ERROR', 'trainit_export.json',
                                    f'summary.polygons.exported = {polygons}, the label files hold '
                                    f'{sum(boxes.values())} polygons'))
        present = [n for n in SPLITS if any(split == n for (split, _) in images)]
        expected = data_yaml(sidecar['export_id'], sidecar['source'], classes, present, sidecar['rights'], task)
        path = dest / DATA_YAML
        if DATA_YAML not in listed:
            findings.append(Finding(DATASET_INCONSISTENT, 'ERROR', DATA_YAML, 'data.yaml is not listed'))
        elif path.is_file() and path.read_bytes() != expected:
            findings.append(Finding(DATASET_INCONSISTENT, 'ERROR', DATA_YAML, 'data.yaml is not the one this '
                                    'export generates (splits, classes or source differ)'))
        return findings


class _PolygonStats:
    """``summary.polygons`` of a segment export: the policy and its parameters, and what the policy dropped."""

    def __init__(self, cfg):
        self.policy = dict(POLYGON_POLICY, epsilon_px=cfg['epsilon_px'], min_area_px=cfg['min_area_px'])
        self.instances = self.exported = self.vertices = 0
        self.multi_component = self.dropped_components = self.dropped_component_pixels = 0
        self.dropped = []

    def add(self, observation_id: str, records):
        for record in records:
            self.instances += 1
            if record['components'] > 1:
                self.multi_component += 1
                self.dropped_components += record['components'] - 1
                self.dropped_component_pixels += record['dropped_pixels']
            if 'dropped' in record:
                self.dropped.append({'observation_id': observation_id, 'id': record['id'], 'class': record['class'],
                                     'reason': record['dropped'], 'pixels': record['pixels']})
            else:
                self.exported += 1
                self.vertices += record['vertices']

    def to_dict(self) -> dict:
        return {'policy': self.policy, 'instances': self.instances, 'exported': self.exported,
                'vertices': self.vertices, 'multi_component_instances': self.multi_component,
                'dropped_components': self.dropped_components,
                'dropped_component_pixels': self.dropped_component_pixels,
                'dropped_instances': len(self.dropped), 'dropped': self.dropped}


def _check_polygon_summary(summary, counts) -> int:
    """``summary.polygons`` is self-consistent; returns the number of exported polygons."""
    poly = summary['polygons']
    numbers = {k: s.integer(0)(poly[k], f'summary.polygons.{k}') for k in
               ('instances', 'exported', 'vertices', 'multi_component_instances', 'dropped_components',
                'dropped_component_pixels', 'dropped_instances')}
    dropped = poly['dropped']
    if not isinstance(dropped, list) or len(dropped) != numbers['dropped_instances'] \
            or numbers['exported'] + numbers['dropped_instances'] != numbers['instances']:
        s.fail(DATASET_INCONSISTENT, 'instances = exported + dropped_instances = exported + len(dropped)',
               'summary.polygons')
    if {k: poly['policy'].get(k) for k in POLYGON_POLICY} != POLYGON_POLICY:
        s.fail(DATASET_INCONSISTENT, f'the policy is not {POLYGON_POLICY}', 'summary.polygons.policy')
    if numbers['exported'] != sum(c.get('instances', 0) for c in counts.values()):
        s.fail(DATASET_INCONSISTENT, 'exported is not the sum of summary.splits.*.instances', 'summary.polygons')
    return numbers['exported']


def _classes_of(store, dataset, chosen):
    from ..labels import load_label_set
    ontologies = {tuple(load_label_set(store, dataset, i).classes) for i in chosen}
    if len(ontologies) > 1:
        raise DataError(EXPORT_REFUSED, f'the label sets use different ontologies {sorted(ontologies)}',
                        context={'ontologies': [list(o) for o in sorted(ontologies)]})
    return list(ontologies.pop())


# ---- the label importer ------------------------------------------------------------------------------------

def _number(token: str, where: str) -> float:
    if not _DEC_RE.match(token):
        s.fail(LABEL_INVALID, f'{token!r} is not a decimal number', where)
    value = float(token)
    if not math.isfinite(value):
        s.fail(LABEL_INVALID, f'{token!r} is not finite', where)
    return value


def _lines(data: bytes, where: str, form: str):
    """``(file:line, fields)`` of every line of a YOLO label file (the strict text rules of both tasks)."""
    try:
        text = data.decode('ascii')
    except UnicodeDecodeError:
        s.fail(LABEL_INVALID, 'not an ASCII text file', where)
    if not text:
        return
    text = text.replace('\r\n', '\n')
    if text.endswith('\n'):
        text = text[:-1]
    for number, line in enumerate(text.split('\n'), start=1):
        at = f'{where}:{number}'
        if not _LINE_RE.match(line):
            s.fail(LABEL_INVALID, f'expected {form} separated by spaces or tabs, got '
                   f'{line[:80]!r}' if line.strip(' \t') else 'an empty line', at)
        yield at, _FIELD_SEP_RE.split(line.strip(' \t'))


def parse_segment_file(data: bytes, n_classes: int, where: str) -> list:
    """``[(class_index, [(x, y), ...])]`` of one YOLO segmentation label file, validated strictly (the text rules
    of :func:`parse_label_file`): an integer class in range, at least 3 vertices, finite decimal numbers in [0, 1],
    and a polygon of non-zero area. An EMPTY file is an image with no instance."""
    out = []
    for at, fields in _lines(data, where, '"<class> <x1> <y1> <x2> <y2> <x3> <y3> ..."'):
        if len(fields) < 7 or len(fields) % 2 == 0:
            s.fail(LABEL_INVALID, f'a polygon line is a class and at least 3 x y pairs, got {len(fields)} fields', at)
        if not _INT_RE.match(fields[0]) or int(fields[0]) >= n_classes:
            s.fail(LABEL_INVALID, f'class {fields[0]!r} is not an index below {n_classes}', at)
        values = [_number(token, at) for token in fields[1:]]
        if not all(0.0 <= v <= 1.0 for v in values):
            s.fail(LABEL_INVALID, 'normalized values lie in [0, 1]', at)
        points = list(zip(values[0::2], values[1::2]))
        n = len(points)
        if sum(points[i][0] * points[(i + 1) % n][1] - points[(i + 1) % n][0] * points[i][1] for i in range(n)) == 0:
            s.fail(LABEL_INVALID, 'the polygon has zero area', at)
        out.append((int(fields[0]), points))
    return out


def parse_label_file(data: bytes, n_classes: int, where: str) -> list:
    """``[(class_index, cx, cy, w, h)]`` of one YOLO detection label file, validated strictly. An EMPTY file (zero
    bytes) is an observation with no object. Lines end with ``\\n`` or ``\\r\\n`` (the last one may lack it); fields
    are separated by spaces or tabs only; an empty line, a lone ``\\r`` or any other character is refused."""
    out = []
    for at, fields in _lines(data, where, '"<class> <cx> <cy> <w> <h>"'):
        if len(fields) != 5:
            detail = ('a segmentation line (polygon): YOLO segmentation labels are not imported in D0.6'
                      if len(fields) > 5 else f'{len(fields)} fields')
            s.fail(LABEL_INVALID, f'expected "<class> <cx> <cy> <w> <h>", got {detail}', at)
        if not _INT_RE.match(fields[0]) or int(fields[0]) >= n_classes:
            s.fail(LABEL_INVALID, f'class {fields[0]!r} is not an index below {n_classes}', at)
        cx, cy, w, h = (_number(t, at) for t in fields[1:])
        if not all(0.0 <= v <= 1.0 for v in (cx, cy, w, h)):
            s.fail(LABEL_INVALID, 'normalized values lie in [0, 1]', at)
        if w <= 0.0 or h <= 0.0:
            s.fail(LABEL_INVALID, 'a box has a positive width and height', at)
        if (cx - w / 2 < -EDGE_TOLERANCE or cy - h / 2 < -EDGE_TOLERANCE or cx + w / 2 > 1 + EDGE_TOLERANCE
                or cy + h / 2 > 1 + EDGE_TOLERANCE):
            s.fail(LABEL_INVALID, f'the box ({cx}, {cy}, {w}, {h}) extends outside the image (refused, never '
                   'clipped)', at)
        out.append((int(fields[0]), cx, cy, w, h))
    return out


def import_yolo_labels(store, dataset: str, directory, classes, *, source: str = 'IMPORTED', tool: str = None,
                       author: str = None, notes: str = None, supersedes: str = None):
    """Import a directory of YOLO detection label files as ONE canonical BBOX_2D label set; returns it."""
    directory = Path(directory).expanduser().resolve()
    if not directory.is_dir():
        raise DataError(INVALID_VALUE, f'{directory} is not a directory', context={'directory': str(directory)})
    source = s.one_of(IMPORT_SOURCES)(source, 'source')
    classes = list(s.list_of(s.ident, min_length=1, unique=True)(list(classes), 'classes'))
    files = sorted(p for p in directory.rglob('*.txt') if p.is_file())
    if not files:
        raise DataError(LABEL_INVALID, f'{directory} holds no .txt label file', context={'directory': str(directory)})
    seen = {}
    sessions = SealedSessions(store)
    entries = []
    for path in files:
        rel = path.relative_to(directory).as_posix()
        obs_id = observation_of(path.stem, rel)
        if obs_id in seen:
            s.fail(LABEL_INVALID, f'{obs_id} is labelled by both {seen[obs_id]} and {rel}', rel)
        seen[obs_id] = rel
        _, obs = sessions.observation(obs_id, rel)
        if getattr(obs.data, 'block', None) != 'image':
            s.fail(LABEL_INVALID, f'{obs_id} has no image data block: a YOLO label file cannot label it', rel)
        width, height = obs.data.width, obs.data.height
        boxes = []
        for index, cx, cy, w, h in parse_label_file(path.read_bytes(), len(classes), rel):
            boxes.append({'class': classes[index],
                          'bbox_xywh_px': [(cx - w / 2) * width, (cy - h / 2) * height, w * width, h * height]})
        entries.append({'observation_id': obs_id, 'boxes': boxes})
    provenance = {'source': source, 'source_ref': 'yolo-detection-txt'}
    for name, value in (('tool', tool), ('author', author), ('notes', notes)):
        s.put(provenance, name, value)
    return create_label_set(store, dataset, kind='BBOX_2D', classes=classes, provenance=provenance, entries=entries,
                            supersedes=supersedes)


EXPORTER = register_exporter(YoloDetectionExporter())

__all__ = ['YoloDetectionExporter', 'import_yolo_labels', 'parse_label_file', 'parse_segment_file',
           'segment_label_lines', 'stem_of', 'EXPORTER']
