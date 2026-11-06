"""Synthetic, customer-neutral fixtures for the D0.7 tests: Replicator-SHAPED annotator output, in plain numpy.

No Isaac anywhere: a "render" is a tiny image where each object is an axis-aligned rectangle, the instance-id image
uses ARBITRARY raw ids (as Replicator does: they change from run to run), the floor is an unlabelled prim, and the
2-D tight boxes are the inclusive pixel bounds the annotator reports. The draws come from a seeded numpy generator,
so the same seed gives the same recorded generation.
"""
import numpy as np

from trainit_data.adapters.isaac_replicator import ReplicatorIngest
from trainit_data.synthetic import SCENE_SCHEMA, run_seed, semantics_of

W, H = 32, 24
SENSOR = 'bin_camera'
FRAME = 'bin_camera_optical_frame'
CLASSES = ['cube', 'cylinder', 'sphere']
RECIPE = 'example_bin_primitives/1'
STAGE_SHA = 'sha256:' + '5a' * 32
GENERATOR_SHA = 'sha256:' + '6b' * 32


def camera_params(width=W, height=H, z=1.0, **over):
    focal, aperture = 18.0, [20.955, 20.955 * height / width]
    params = {'cameraModel': 'pinhole', 'renderProductResolution': [width, height], 'cameraFocalLength': focal,
              'cameraAperture': aperture, 'cameraApertureOffset': [0.0, 0.0],
              'cameraProjection': [2 * focal / aperture[0], 0, 0, 0, 0, 2 * focal / aperture[1], 0, 0, 0, 0, 1e-6, -1,
                                   0, 0, 1.000001, 0],
              'cameraViewTransform': [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, -z, 1], 'metersPerSceneUnit': 1.0}
    params.update(over)
    return params


def scene(**over):
    out = {'schema': SCENE_SCHEMA, 'kind': 'PROCEDURAL',
           'generator': {'name': 'example_bin_generator', 'sha256': GENERATOR_SHA},
           'base_stage': {'sha256': STAGE_SHA, 'completeness': 'RESOLVED_LAYERS', 'layers': [], 'unresolved': []},
           'assets': [], 'semantics': semantics_of(CLASSES, 'prim semantic label "class"'),
           'limitations': ['procedural primitives: the scene is its generator + config + draws']}
    out.update(over)
    return out


CONFIG = {'generator': 'example_bin_generator', 'resolution': [W, H], 'objects_per_run': [2, 3], 'classes': CLASSES}


def draws_for(master_seed: int, run_index: int) -> dict:
    """The run's draws: object rectangles (no overlap) with their classes, and a light."""
    rng = np.random.default_rng(run_seed(master_seed, run_index))
    count = int(rng.integers(2, 4))
    objects = []
    for k in range(count):
        x = 1 + 10 * k + int(rng.integers(0, 3))
        y = int(rng.integers(1, 12))
        objects.append({'object': f'o{k:02d}', 'class': CLASSES[int(rng.integers(0, len(CLASSES)))],
                        'rect_px': [x, y, int(rng.integers(3, 8)), int(rng.integers(3, 10))]})
    return {'objects': objects, 'light_intensity': round(float(rng.uniform(500, 3000)), 3)}


def render(draws: dict, view: int, raw_offset: int):
    """A fake render of the run's draws: rgb, depth, raw instance ids, idToLabels-like map, tight boxes."""
    rgb = np.zeros((H, W, 4), dtype=np.uint8)
    rgb[..., 3] = 255
    rgb[..., 0] = 40 + 10 * view
    depth = np.full((H, W), 1.0, dtype=np.float32)
    depth[0, 0] = np.inf                                         # a pixel with no hit
    ids = np.full((H, W), raw_offset, dtype=np.uint32)           # the floor
    id_to_prim = {raw_offset: '/World/floor'}
    boxes = []
    for k, obj in enumerate(draws['objects']):
        x, y, w, h = obj['rect_px']
        raw = raw_offset + 7 * (k + 1)
        ids[y:y + h, x:x + w] = raw
        id_to_prim[raw] = f'/World/parts/{obj["object"]}/mesh'
        rgb[y:y + h, x:x + w, 1] = 200
        depth[y:y + h, x:x + w] = 0.9
        boxes.append({'prim': f'/World/parts/{obj["object"]}/mesh', 'x_min': x, 'y_min': y, 'x_max': x + w - 1,
                      'y_max': y + h - 1, 'occlusion': 0.0})
    objects = {f'/World/parts/{o["object"]}': {'object': o['object'], 'class': o['class']} for o in draws['objects']}
    return rgb, depth, ids, id_to_prim, objects, boxes


def make_synthetic_session(store, *, master_seed=7, runs=4, views=2, raw_offset=1, rights=None, close=True,
                           software=None, record_runs=None):
    """A synthetic session of ``runs`` x ``views`` frames (only the first ``record_runs`` runs recorded, if given)."""
    ingest = ReplicatorIngest(
        store, sensor_id=SENSOR, frame_id=FRAME, config=dict(CONFIG, master_seed=master_seed, runs=runs, views=views),
        master_seed=master_seed, recipe=RECIPE, runs=runs, views_per_run=views, scene=scene(),
        camera_params=camera_params(), camera_prim='/World/Camera',
        environment={'isaac_sim': 'example-5.1', 'replicator': 'example-1.12', 'numpy': np.__version__},
        software=software or {'example_generator': '1.0'}, rights=rights)
    for run_index in range(runs if record_runs is None else record_runs):
        draws = draws_for(master_seed, run_index)
        for view in range(views):
            rgb, depth, ids, id_to_prim, objects, boxes = render(draws, view, raw_offset + run_index)
            ingest.add_frame(run_index=run_index, view=view, draws=draws, view_draws={'light_jitter': 0.1 * view},
                             sim_time_s=round(1.0 / 60 * (run_index * views + view + 1), 9), rgb=rgb, depth=depth,
                             instance_ids=ids, id_to_prim=id_to_prim, objects=objects, tight_boxes=boxes)
    if not close:
        return ingest
    return ingest.close()


__all__ = ['make_synthetic_session', 'camera_params', 'scene', 'draws_for', 'render', 'CLASSES', 'SENSOR', 'FRAME',
           'RECIPE', 'W', 'H', 'CONFIG']
