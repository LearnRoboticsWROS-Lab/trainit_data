"""Isaac Replicator -> canonical TrainIt records: the pure ingest adapter (PUBLIC, D0.7).

This module never imports Isaac (``omni``, ``isaacsim``, ``carb``, ``pxr``, Replicator): it consumes what a
generator script already pulled out of Replicator's annotators -- numpy arrays and JSON-compatible metadata -- and
writes ONE ordinary canonical session through the same :class:`~trainit_data.writer.SessionWriter` every source
uses. It is importable and testable anywhere the core is (Python >= 3.10, numpy, Pillow). The code that CALLS
Replicator lives in the generator (``examples/isaac_replicator_bin/generate.py`` publicly; a private cell's own
script privately).

What a generated frame becomes (one capture, fired by a ``SYNTHETIC_FRAME`` trigger, origin ``GENERATOR``):

=================  =================================================================================================
``color``          ``rgb`` annotator, RGBA uint8 -> rgb8 PNG (the alpha channel is dropped and recorded as a transform)
``depth``          ``distance_to_image_plane`` float32 metres -> 32FC1 npy (``inf`` = no hit, kept; never
                   ``distance_to_camera``, which is radial)
``instances``      ``instance_id_segmentation`` (+ ``bounding_box_2d_tight``) -> a 16-bit mask of COMPACT ids and its
                   ``instance_image`` block: Replicator's ids change from run to run, so each labelled object (the
                   generator's ``objects`` map: prim -> object key + class) gets id 1..N in object-key order; every
                   other prim (floor, bin) is 0; an object made of several prims is one instance
=================  =================================================================================================

The annotator box and the mask extent of each instance must agree within 1 px per edge, or the frame is refused
(``INVALID_VALUE``): a ground truth that contradicts itself is never written.

Calibration (``SYNTHETIC_RENDER`` intrinsics, ``SYNTHETIC_SCENE`` extrinsic) comes from the ``camera_params``
annotator: ``fx = f * W / aperture_h``, ``fy = f * H / aperture_v``, the principal point at ``(W / 2, H / 2)`` -- the
same pixel-edge convention as the Isaac ROS CameraInfo (D0.3) -- cross-checked against the projection matrix;
pinhole only, aperture offsets refused. The extrinsic is the world -> optical-frame pose (ROS optical axes: +z
forward, +y down) of the render product's camera, from its view transform. The three streams share one pixel grid
and one tick: sync ``SAME_RENDER_TICK`` / ``SYNCED``; depth ``registered_to: color``, ``same_render_product``.

Stamps are the simulator's timeline time read at the capture (``SIM_TIME`` / isaac, origin ``RENDER_TICK``), never
fabricated; the frame's order is its run / view index (context).
"""
import math

import numpy as np

from .. import _strict as s
from .._canonical import sha256_json
from ..calibration import CalibrationRecord
from ..errors import INVALID_VALUE, DataError
from ..observation import Quality
from ..session_record import Clock, SensorDecl, Source, StreamDecl
from ..synthetic import GENERATION_SCHEMA, NAMESPACE, TRIGGER_KIND, generation_block, run_id, run_seed, scene_manifest
from ..timestamps import Timestamp
from ..vocab import Vocabulary
from ..writer import ImageSample, SessionWriter

ADAPTER = 'trainit_data.adapters.isaac_replicator'
GENERATOR = 'isaac_replicator'
BACKEND = 'isaac'
MASK_ANNOTATOR = 'instance_id_segmentation'
BOX_ANNOTATOR = 'bounding_box_2d_tight'
DEPTH_ANNOTATOR = 'distance_to_image_plane'
STREAMS = (('color', 'rgb'), ('depth', 'depth'), ('instances', 'instance_segmentation'))
SYNC_POLICY = {'name': 'SAME_RENDER_TICK', 'version': 1}
PRINCIPAL_POINT_NOTE = ('Isaac render product: principal point at the image centre (W/2, H/2) in pixel-edge '
                        'coordinates, as the Isaac ROS CameraInfo publishes it; consumers must not compensate')
_PROJECTION_RTOL = 1e-4
_NS = 1_000_000_000


def _refuse(message, **context):
    raise DataError(INVALID_VALUE, message, context=context)


# ---- camera_params -> calibration ---------------------------------------------------------------------------

def _array(params, key, size):
    if key not in params:
        _refuse(f'camera_params has no {key!r}', key=key)
    value = np.asarray(params[key], dtype=np.float64).reshape(-1)
    if value.size != size or not np.all(np.isfinite(value)):
        _refuse(f'camera_params {key!r} is not {size} finite numbers', key=key)
    return value


def intrinsics_from_camera_params(params: dict) -> dict:
    """The canonical intrinsics block (CameraInfo form) of a Replicator ``camera_params`` sample."""
    model = str(params.get('cameraModel', 'pinhole'))
    if model != 'pinhole':
        _refuse(f'camera model {model!r}: only a pinhole render product is ingested', model=model)
    width, height = (int(v) for v in _array(params, 'renderProductResolution', 2))
    focal = float(_array(params, 'cameraFocalLength', 1)[0])
    aperture = _array(params, 'cameraAperture', 2)
    if 'cameraApertureOffset' in params and np.any(_array(params, 'cameraApertureOffset', 2) != 0.0):
        _refuse('a non-zero aperture offset (an off-centre principal point) is not ingested')
    if width < 1 or height < 1 or focal <= 0.0 or np.any(aperture <= 0.0):
        _refuse('camera_params has no positive resolution / focal length / aperture')
    fx, fy = focal * width / aperture[0], focal * height / aperture[1]
    if 'cameraProjection' in params:
        proj = _array(params, 'cameraProjection', 16)
        for name, value, from_projection in (('fx', fx, proj[0] * width / 2.0), ('fy', fy, proj[5] * height / 2.0)):
            if not math.isclose(value, from_projection, rel_tol=_PROJECTION_RTOL):
                _refuse(f'{name} {value} from focal length / aperture disagrees with the projection matrix '
                        f'({from_projection})', parameter=name)
    cx, cy = width / 2.0, height / 2.0
    return {'width': width, 'height': height, 'distortion_model': 'plumb_bob', 'D': [0.0] * 5,
            'K': [fx, 0.0, cx, 0.0, fy, cy, 0.0, 0.0, 1.0], 'R': [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
            'P': [fx, 0.0, cx, 0.0, 0.0, fy, cy, 0.0, 0.0, 0.0, 1.0, 0.0], 'binning': [0, 0],
            'roi': {'x': 0, 'y': 0, 'width': 0, 'height': 0, 'do_rectify': False}}


def _canonical_sign(q: list) -> list:
    """``q`` or ``-q`` (the same rotation): ``w > 0``, or, when ``w`` is 0, the first non-zero of x, y, z positive.
    Applied AFTER rounding, so floating-point noise around ``w = 0`` cannot flip the identity."""
    q = [v + 0.0 for v in q]                                   # no -0.0
    lead = q[3] if q[3] != 0.0 else next((v for v in q[:3] if v != 0.0), 1.0)
    return [-v + 0.0 for v in q] if lead < 0.0 else q


def _quaternion_xyzw(r) -> list:
    """A rotation matrix as a unit quaternion ``[x, y, z, w]`` (sign not yet canonical: see ``_canonical_sign``)."""
    trace = r[0, 0] + r[1, 1] + r[2, 2]
    if trace > 0.0:
        k = 2.0 * math.sqrt(trace + 1.0)
        q = [(r[2, 1] - r[1, 2]) / k, (r[0, 2] - r[2, 0]) / k, (r[1, 0] - r[0, 1]) / k, 0.25 * k]
    elif r[0, 0] > r[1, 1] and r[0, 0] > r[2, 2]:
        k = 2.0 * math.sqrt(1.0 + r[0, 0] - r[1, 1] - r[2, 2])
        q = [0.25 * k, (r[0, 1] + r[1, 0]) / k, (r[0, 2] + r[2, 0]) / k, (r[2, 1] - r[1, 2]) / k]
    elif r[1, 1] > r[2, 2]:
        k = 2.0 * math.sqrt(1.0 + r[1, 1] - r[0, 0] - r[2, 2])
        q = [(r[0, 1] + r[1, 0]) / k, 0.25 * k, (r[1, 2] + r[2, 1]) / k, (r[0, 2] - r[2, 0]) / k]
    else:
        k = 2.0 * math.sqrt(1.0 + r[2, 2] - r[0, 0] - r[1, 1])
        q = [(r[0, 2] + r[2, 0]) / k, (r[1, 2] + r[2, 1]) / k, 0.25 * k, (r[1, 0] - r[0, 1]) / k]
    norm = math.sqrt(sum(v * v for v in q))
    return [v / norm for v in q]


def optical_pose_from_camera_params(params: dict, decimals: int = 9) -> tuple:
    """``(translation_m, rotation_xyzw)``: the pose of the camera's OPTICAL frame (ROS axes) in the stage's world.

    Replicator's ``cameraViewTransform`` is world -> camera, row-major, row-vector convention (translation in the
    last row); a USD camera looks along -z with +y up, the optical frame along +z with +y down. Rounded to
    ``decimals`` so the same stage gives the same calibration id."""
    view = _array(params, 'cameraViewTransform', 16).reshape(4, 4).T          # column-vector world -> camera
    world_camera = np.linalg.inv(view)
    world_optical = world_camera @ np.diag([1.0, -1.0, -1.0, 1.0])
    rotation = world_optical[:3, :3]
    rigid = np.allclose(rotation @ rotation.T, np.eye(3), atol=1e-6)
    if not rigid or not math.isclose(np.linalg.det(rotation), 1.0, abs_tol=1e-6):
        _refuse('cameraViewTransform is not a rigid transform (scaled camera?)')
    meters = float(params.get('metersPerSceneUnit', 1.0))
    translation = [round(float(v) * meters, decimals) + 0.0 for v in world_optical[:3, 3]]
    return translation, _canonical_sign([round(v, decimals) for v in _quaternion_xyzw(rotation)])


def calibration_contents(*, sensor_id: str, frame_id: str, camera_params: dict, world_frame: str,
                         scene_sha256: str, camera_prim: str) -> dict:
    """``{stream: calibration content}`` for the three streams of one render product."""
    intrinsics = intrinsics_from_camera_params(camera_params)
    translation, rotation = optical_pose_from_camera_params(camera_params)
    ref = {'stage_sha256': scene_sha256, 'prim': camera_prim, 'source': 'camera_params.cameraViewTransform'}
    extrinsic = {'parent_frame': world_frame, 'child_frame': frame_id, 'translation': translation,
                 'rotation_xyzw': rotation, 'provenance': {'kind': 'SYNTHETIC_SCENE', 'ref': ref}}
    out = {}
    for stream, _ in STREAMS:
        content = {'sensor_id': sensor_id, 'stream': stream, 'optical_frame': frame_id, 'intrinsics': intrinsics,
                   'intrinsics_provenance': {'kind': 'SYNTHETIC_RENDER', 'ref': {'source': 'camera_params'},
                                             'notes': PRINCIPAL_POINT_NOTE},
                   'extrinsics': [extrinsic], 'transform_tree': {'source': 'synthetic_scene'}}
        if stream == 'depth':
            content['depth'] = {'invalid': 'nan_or_inf', 'registered_to': 'color',
                                'registration_basis': 'same_render_product'}
        out[stream] = content
    return out


# ---- annotator output -> one capture --------------------------------------------------------------------------

def _owner(prim: str, objects: dict):
    """The labelled object a rendered prim belongs to: the object whose prim is it or one of its ancestors."""
    best = None
    for root in objects:
        if prim == root or prim.startswith(root.rstrip('/') + '/'):
            if best is None or len(root) > len(best):
                best = root
    return best


def compact_instances(instance_ids, id_to_prim: dict, objects: dict, tight_boxes=()):
    """``(mask uint16, instances)``: Replicator's instance ids remapped to compact ids 1..N (object-key order) of the
    labelled objects; every other prim is 0. ``tight_boxes``: ``[{prim, x_min, y_min, x_max, y_max, occlusion?}]``
    (inclusive pixel bounds, as the annotator reports them)."""
    ids = np.asarray(instance_ids)
    if ids.ndim != 2 or ids.dtype.kind not in 'ui':
        _refuse(f'instance ids are a 2-D integer image, not {ids.dtype} {ids.shape}')
    for prim, obj in objects.items():
        s.ident(obj.get('object'), f'objects[{prim!r}].object')
        s.ident(obj.get('class'), f'objects[{prim!r}].class')
    by_object = {}
    for raw, prim in id_to_prim.items():
        root = _owner(str(prim), objects)
        if root is not None:
            by_object.setdefault(objects[root]['object'], (root, []))[1].append(int(raw))
    keys = sorted(by_object)
    if len(keys) > 65535:
        _refuse('more than 65535 labelled instances in one frame')
    mask = np.zeros(ids.shape, dtype=np.uint16)
    present = []
    for key in keys:
        root, raws = by_object[key]
        pixels = np.isin(ids, raws)
        if pixels.any():
            present.append((key, root, pixels))
    instances = []
    boxes = {}                                                   # by object key: an object of several prims, one box
    for box in tight_boxes:
        root = _owner(str(box['prim']), objects)
        if root is None:
            continue
        x0, y0, x1, y1 = (int(box[k]) for k in ('x_min', 'y_min', 'x_max', 'y_max'))
        key = objects[root]['object']
        prev = boxes.get(key)
        if prev is None:
            bounds = (x0, y0, x1, y1)
        else:
            bounds = (min(prev['b'][0], x0), min(prev['b'][1], y0), max(prev['b'][2], x1), max(prev['b'][3], y1))
        boxes[key] = {'b': bounds, 'occlusion': box.get('occlusion') if prev is None else None}
    for number, (key, root, pixels) in enumerate(present, start=1):
        mask[pixels] = number
        inst = {'id': number, 'object': key, 'class': objects[root]['class'], 'pixels': int(pixels.sum())}
        if key in boxes:
            x0, y0, x1, y1 = boxes[key]['b']
            inst['box_tight_xywh_px'] = [x0, y0, x1 - x0 + 1, y1 - y0 + 1]
            occlusion = boxes[key]['occlusion']
            if occlusion is not None and math.isfinite(float(occlusion)) and 0.0 <= float(occlusion) <= 1.0:
                inst['occlusion'] = round(float(occlusion), 6)
        instances.append(inst)
    return mask, instances


def _stamp(sim_time_s: float) -> Timestamp:
    if not isinstance(sim_time_s, (int, float)) or not math.isfinite(sim_time_s) or sim_time_s < 0.0:
        _refuse(f'the simulation time {sim_time_s!r} is not a finite time >= 0')
    ns = round(float(sim_time_s) * _NS)
    return Timestamp(ns // _NS, ns % _NS, 'SIM_TIME', 'RENDER_TICK', 'isaac')


class ReplicatorIngest:
    """One generation run -> one canonical session. Create it, :meth:`add_frame` per rendered view, :meth:`close`.

    ``config`` is the generator's full, resolved configuration (its hash is the session's
    ``recording_config.resolved_sha256`` and the generation block's ``config_sha256``). ``scene`` is the
    ``trainit-synthetic-scene/1.0`` manifest (``synthetic.scene_manifest``). ``environment`` (Isaac Sim, Replicator,
    numpy... versions) is where the reproducibility claim holds. ``camera_params``: the first frame's annotator
    sample; every frame must keep the same camera (a moving camera is a later seam)."""

    def __init__(self, store, *, sensor_id: str, frame_id: str, config: dict, master_seed: int, recipe: str,
                 runs: int, views_per_run: int, scene: dict, camera_params: dict, camera_prim: str,
                 environment: dict, software: dict = None, world_frame: str = 'world', rights: dict = None,
                 backend_details: dict = None, config_file_sha256: str = None):
        self.store = store
        self.sensor_id = s.ident(sensor_id, 'sensor_id')
        self.frame_id = s.token(frame_id, 'frame_id')
        self.recipe = s.token(recipe, 'recipe')
        self.scene = scene_manifest(scene, 'scene')
        self._camera_view = _array(camera_params, 'cameraViewTransform', 16).tolist()
        contents = calibration_contents(sensor_id=sensor_id, frame_id=frame_id, camera_params=camera_params,
                                        world_frame=world_frame, scene_sha256=self.scene['base_stage']['sha256'],
                                        camera_prim=s.string(camera_prim, 'camera_prim'))
        vocab = Vocabulary.default()
        self.calibrations = {}
        for stream, content in contents.items():
            record = CalibrationRecord.build(content, vocab)
            store.put_calibration(record)
            self.calibrations[stream] = record.calibration_id
        self.width = contents['color']['intrinsics']['width']
        self.height = contents['color']['intrinsics']['height']
        generation = generation_block({
            'schema': GENERATION_SCHEMA, 'generator': GENERATOR, 'recipe': self.recipe, 'master_seed': master_seed,
            'config_sha256': sha256_json(s.opaque(config, 'config')), 'runs': runs, 'views_per_run': views_per_run,
            'reproducibility': {'claim': 'SAME_DRAWS', 'render_bit_identical': False,
                                'environment': s.str_map(environment, 'environment')}})
        details = {'annotators': ['rgb', DEPTH_ANNOTATOR, MASK_ANNOTATOR, BOX_ANNOTATOR, 'camera_params'],
                   'camera_prim': camera_prim, 'render_product_resolution': [self.width, self.height]}
        details.update(backend_details or {})
        self.writer = SessionWriter.create(
            store,
            source=Source(transport='INGEST', backend=BACKEND, adapter=ADAPTER,
                          descriptor={'intrinsics_kind': 'SYNTHETIC_RENDER',
                                      'registration_basis': 'same_render_product', 'stamp_origin': 'RENDER_TICK',
                                      'supplied_by': 'generator'},
                          backend_details={BACKEND: details}),
            clock=Clock('SIM_TIME', 'isaac'),
            sensors=[SensorDecl(self.sensor_id, tuple(StreamDecl(st, mod) for st, mod in STREAMS))],
            rights=rights,
            provenance={'software': dict(software or {}), 'randomization': generation, 'scene': self.scene},
            recording_config=dict({'resolved': config}, **({'file_sha256': config_file_sha256}
                                                           if config_file_sha256 else {})),
            vocabulary=vocab)
        self.generation = generation
        self._frames = 0
        self._runs = {}

    @property
    def session_id(self) -> str:
        return self.writer.session_id

    def add_frame(self, *, run_index: int, view: int, draws: dict, view_draws: dict, sim_time_s: float, rgb,
                  depth, instance_ids, id_to_prim: dict, objects: dict, tight_boxes=(), camera_params: dict = None):
        """Persist one rendered view of run ``run_index``; returns the committed capture.

        ``draws``: the run's recorded randomization decisions (identical for every view of the run); ``view_draws``:
        this view's own. ``objects``: ``{prim path: {object, class}}`` of the labelled objects. The arrays are the
        annotators' output, unconverted."""
        s.integer(0)(run_index, 'run_index')
        s.integer(0)(view, 'view')
        if not 0 <= run_index < self.generation['runs'] or not 0 <= view < self.generation['views_per_run']:
            _refuse(f'run {run_index} / view {view} is outside the declared {self.generation["runs"]} runs x '
                    f'{self.generation["views_per_run"]} views')
        rid = run_id(self.recipe, draws)
        known = self._runs.get(run_index, {'run': rid, 'views': set()})
        if known['run'] != rid:
            _refuse(f'run {run_index} was recorded with other draws: one run, one set of draws')
        if view in known['views']:
            _refuse(f'view {view} of run {run_index} is already recorded')
        twin = next((i for i, r in self._runs.items() if r['run'] == rid and i != run_index), None)
        if twin is not None:
            _refuse(f'run {run_index} has the same draws as run {twin}: two runs of one generation are two scenes')
        if camera_params is not None and _array(camera_params, 'cameraViewTransform', 16).tolist() != self._camera_view:
            _refuse('the camera moved: this adapter ingests a fixed camera per session')
        unknown = sorted({o['class'] for o in objects.values()} - set(self.scene['semantics']['classes']))
        if unknown:
            _refuse(f'classes {unknown} are not in the scene manifest\'s semantic classes', classes=unknown)
        rgb = np.asarray(rgb)
        transforms = ()
        if rgb.ndim == 3 and rgb.shape[2] == 4:
            rgb, transforms = np.ascontiguousarray(rgb[:, :, :3]), ({'op': 'drop_alpha', 'params': {'from': 'rgba8'}},)
        depth = np.asarray(depth, dtype=np.float32) if depth is not None else None
        mask, instances = compact_instances(instance_ids, id_to_prim, objects, tight_boxes)
        for name, array in (('rgb', rgb), ('depth', depth), ('instance ids', mask)):
            if array is not None and array.shape[:2] != (self.height, self.width):
                _refuse(f'the {name} image is {array.shape[1]}x{array.shape[0]}, the render product is '
                        f'{self.width}x{self.height}')
        stamp = _stamp(sim_time_s)
        quality = None
        samples = [ImageSample(self.sensor_id, 'color', stamp, rgb, 'rgb8', self.frame_id,
                               calibration_id=self.calibrations['color'], transforms=transforms)]
        if depth is not None:
            invalid = float(np.count_nonzero(~np.isfinite(depth))) / depth.size
            quality = {'invalid_depth_fraction': round(invalid, 6)}
            samples.append(ImageSample(self.sensor_id, 'depth', stamp, depth, '32FC1', self.frame_id, unit_scale_m=1.0,
                                       calibration_id=self.calibrations['depth'], quality=Quality.from_dict(quality)))
        block = {'instances': instances, 'annotator': MASK_ANNOTATOR}
        if any('box_tight_xywh_px' in i for i in instances):
            block['box_annotator'] = BOX_ANNOTATOR
        samples.append(ImageSample(self.sensor_id, 'instances', stamp, mask, '16UC1', self.frame_id,
                                   calibration_id=self.calibrations['instances'], block_fields=block))
        trigger = {'trigger_id': 'replicator', 'kind': TRIGGER_KIND, 'firing_index': self._frames,
                   'origin': 'GENERATOR', 'issued': {'stamp': stamp.to_dict()},
                   'reason': {'recipe': self.recipe, 'run_seed': run_seed(self.generation['master_seed'], run_index),
                              'draws': draws, 'view_draws': view_draws},
                   'context': {NAMESPACE: {'run': rid, 'run_index': run_index, 'view': view}}}
        sync = {'policy': dict(SYNC_POLICY), 'reference': f'{self.sensor_id}/color', 'status': 'SYNCED',
                'max_skew_ns': 0}
        capture = self.writer.append_capture(trigger=trigger, outcome='COMPLETE', samples=samples, sync=sync)
        self._runs.setdefault(run_index, known)['views'].add(view)        # registered only once committed
        self._frames += 1
        return capture

    def close(self, warnings=()):
        """Seal the session (CLOSED); returns its header."""
        return self.writer.close(warnings=warnings)

    def abandon(self):
        """Stop without sealing (an aborted generation): the session stays ACTIVE for ``trainit-data recover``."""
        self.writer.release()


__all__ = ['ReplicatorIngest', 'intrinsics_from_camera_params', 'optical_pose_from_camera_params',
           'calibration_contents', 'compact_instances', 'ADAPTER', 'GENERATOR', 'MASK_ANNOTATOR', 'BOX_ANNOTATOR',
           'DEPTH_ANNOTATOR']
