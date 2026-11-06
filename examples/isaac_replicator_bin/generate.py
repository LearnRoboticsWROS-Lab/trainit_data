"""Customer-neutral Isaac Replicator generator: primitives in a tray -> ONE canonical TrainIt session (D0.7 example).

The ONLY code of this example that touches Isaac. It builds a neutral scene (a tray, cubes / cylinders / spheres,
a dome and a key light, a fixed overhead camera), randomizes it with its OWN seeded generator, renders with
Replicator's annotators, and hands the plain numpy arrays to ``trainit_data.adapters.isaac_replicator`` -- which
writes the canonical session. Replicator never writes a dataset: labels, versions and the YOLO export come later
from the canonical records (``trainit-data labels from-synthetic``, ``version create``, ``export yolo``).

Mode A (ISAAC_REPLICATOR.md §4): nothing is simulated; every object is PLACED by the recorded draws (rejection
sampling on the tray floor, no overlap), so the scene is exactly its draws. Per run: the objects (count, class,
size, position, yaw, colour); per view: the lights (photometric). The camera is fixed.

Every scene or lighting change is followed by ``settle_steps`` rendered frames before the captured one: the real-time
renderer accumulates over frames, and without them the previous run lingers as faint ghosts in the RGB (never in the
masks or depth) -- unlabelled pixels.

Two checks speak about the WORLD, not the code, and stop the run loudly: every drawn object stands where it was
drawn on the stage, and every drawn object is in the rendered instance mask (from overhead, with no overlap, each
one is visible). Each run's objects live under a prim path of their own (``/World/Parts/r<run>``): re-defining a
path as another primitive type makes the renderer skip that mesh (measured in Isaac Sim 5.1), which only the
second check would catch.

Reproducibility (the SAME_DRAWS claim): the same config + seed, in the same supported environment, records the same
runs, ordering, draws, semantic assignments and calibrations (``trainit-data compare-generation A B``). Rendered
pixels are not promised bit-identical.

This is the REFERENCE SYNTHETIC PROVIDER (D0.8): ``provider.json`` beside it declares it
(``trainit-synthetic-provider/1.0``, ``trainit_data.synthetic_provider``). The configuration and the seed / runs /
views are validated against that descriptor BEFORE Isaac starts, and progress is reported on stdout with the
``trainit-provider-progress/1`` lines a launching tool (TSA) reads. It is installed with ``trainit_data`` under
``share/trainit_data/synthetic_providers/isaac_replicator_bin/``.

Run with Isaac Sim's Python (Kit), from this repository's root (or from the installed share directory)::

    env -i HOME=$HOME PATH=/usr/bin:/bin TRAINIT_DATA_ROOT=$HOME/TrainItData \\
        $HOME/isaacsim/python.sh examples/isaac_replicator_bin/generate.py --seed 20261102 --runs 24 --views 2

The last protocol line is ``TRAINIT_SESSION <session_id>``, or ``TRAINIT_FAILED`` (exit 2 for a configuration
refused before Isaac started; exit 1 after a failure in Isaac, the session then left ACTIVE for
``trainit-data recover``; Kit is closed either way).
"""
import argparse
import hashlib
import json
import math
import sys
import traceback
from pathlib import Path

sys.stdout.reconfigure(line_buffering=True)          # Kit's app.close() does not flush redirected stdout

HERE = Path(__file__).resolve().parent
SENSOR_ID = 'bin_camera'
FRAME_ID = 'bin_camera_optical_frame'
CAMERA_PRIM = '/World/Camera'
PARTS = '/World/Parts'
SEMANTIC_RULE = 'prim semantic label "class" (isaacsim.core.utils.semantics.add_labels)'


def core_path() -> Path:
    """Where this provider's ``trainit_data`` core lives: the repository beside it (a source checkout), else the
    installed prefix it was installed with (``<prefix>/share/trainit_data/synthetic_providers/<id>/`` -> the
    prefix's ``lib/python3.*/site-packages``). Kit's Python has no TrainIt install of its own, and a directory named
    trainit_data in the working directory is not the package."""
    source = HERE.parents[1] / 'trainit_data'
    if (source / 'trainit_data' / '__init__.py').is_file():
        return source
    prefix = HERE.parents[3] if len(HERE.parents) > 3 else HERE
    for site in sorted(prefix.glob('lib/python3*/site-packages')):
        if (site / 'trainit_data' / '__init__.py').is_file():
            return site
    raise RuntimeError(f'the trainit_data core is neither in a repository beside {HERE} nor in an install prefix')


def store_root(args, resolve_store_root):
    """``--store`` WINS over ``TRAINIT_DATA_ROOT`` (a preview's scratch store is never the user's)."""
    return args.store or resolve_store_root()


def progress(phase, **facts):
    """One ``trainit-provider-progress/1`` line."""
    print('TRAINIT_PROGRESS ' + json.dumps(dict(facts, phase=phase), sort_keys=True))


def world_problems(cfg) -> list:
    """Rules of THIS generator the descriptor's per-field bounds cannot state: the whole tray stays in the fixed
    camera's view (else a drawn object may be out of frame and the run stops), and the largest object fits."""
    problems = []
    width, height = cfg['resolution']
    cam = cfg['camera']
    half_w = cam['height_m'] * cam['horizontal_aperture'] / 2 / cam['focal_length']
    half_h = half_w * height / width
    tray_x, tray_y = cfg['tray']['inner_m']
    if tray_x / 2 > half_w or tray_y / 2 > half_h:
        problems.append(f'tray.inner_m {cfg["tray"]["inner_m"]}: the tray does not fit in the camera view at '
                        f'camera.height_m {cam["height_m"]} (visible {2 * half_w:.3f} x {2 * half_h:.3f} m)')
    largest = cfg['object_size_m'][1] * math.sqrt(2) / 2
    if largest > min(tray_x, tray_y) / 2:
        problems.append(f'object_size_m {cfg["object_size_m"]}: the largest object does not fit in the tray')
    return problems


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('--config', default=str(HERE / 'bin_config.json'))
    parser.add_argument('--seed', type=int, required=True, help='the master seed')
    parser.add_argument('--runs', type=int, required=True, help='randomization runs (one scene each)')
    parser.add_argument('--views', type=int, default=2, help='rendered views per run (lighting varies)')
    parser.add_argument('--store', help='the data store (default: $TRAINIT_DATA_ROOT, else ~/TrainItData)')
    parser.add_argument('--classification', default='INTERNAL', choices=['INTERNAL', 'PUBLIC_APPROVED'],
                        help='rights of the generated session (neutral synthetic data: INTERNAL unless the owner '
                        'states otherwise)')
    parser.add_argument('--gui', action='store_true', help='open the Isaac window (default: headless)')
    return parser.parse_args(argv)


def sha256_file(path: Path) -> str:
    return 'sha256:' + hashlib.sha256(path.read_bytes()).hexdigest()


# ---- the recorded randomization (pure: numpy only) --------------------------------------------------------------

def _r(value, decimals):
    return round(float(value), decimals) + 0.0


def draw_run(rng, cfg) -> dict:
    """One randomized scene. Every value is rounded BEFORE it is applied, so the recorded draws are exactly the
    scene."""
    d = cfg['decimals']
    half_x, half_y = cfg['tray']['inner_m'][0] / 2, cfg['tray']['inner_m'][1] / 2
    count = int(rng.integers(cfg['objects_per_run'][0], cfg['objects_per_run'][1] + 1))
    placed = []
    for k in range(count):
        cls = cfg['classes'][int(rng.integers(0, len(cfg['classes'])))]
        size = _r(rng.uniform(*cfg['object_size_m']), d)
        radius = size * (math.sqrt(2) / 2 if cls == 'cube' else 0.5)
        for _ in range(cfg['placement']['attempts']):
            x = _r(rng.uniform(-half_x + radius, half_x - radius), d)
            y = _r(rng.uniform(-half_y + radius, half_y - radius), d)
            if all(math.hypot(x - o['position_m'][0], y - o['position_m'][1]) >= radius + o['_radius']
                   + cfg['placement']['margin_m'] for o in placed):
                break
        else:
            continue                                             # no free spot: the object is not drawn
        placed.append({'object': f'o{k:02d}', 'class': cls, 'size_m': size, '_radius': radius,
                       'position_m': [x, y, _r(size / 2, d)], 'yaw_deg': _r(rng.uniform(0.0, 360.0), d),
                       'color_rgb': [_r(v, d) for v in rng.uniform(*cfg['color_range'], size=3)]})
    return {'objects': [{k: v for k, v in o.items() if k != '_radius'} for o in placed]}


def draw_view(rng, cfg) -> dict:
    d, lights = cfg['decimals'], cfg['lights']
    return {'dome_intensity': _r(rng.uniform(*lights['dome_intensity']), d),
            'key_intensity': _r(rng.uniform(*lights['key_intensity']), d),
            'key_elevation_deg': _r(rng.uniform(*lights['key_elevation_deg']), d),
            'key_azimuth_deg': _r(rng.uniform(*lights['key_azimuth_deg']), d)}


# ---- Isaac ------------------------------------------------------------------------------------------------------

def main(argv=None):
    args = parse_args(argv)
    progress('starting')
    config_path = Path(args.config).resolve()
    try:
        core = core_path()
        # the repository's core goes FIRST (a trainit_data directory in the working directory is not the
        # package); an install prefix's site-packages goes LAST (it may hold other packages: never shadow Kit's)
        if core.name == 'site-packages':
            sys.path.append(str(core))
        else:
            sys.path.insert(0, str(core))
        from trainit_data.synthetic_provider import config_problems, general_problems, load_descriptor
        descriptor = load_descriptor(HERE)
        cfg = json.loads(config_path.read_text())
        problems = (general_problems(descriptor, seed=args.seed, runs=args.runs, views=args.views)
                    + config_problems(descriptor, cfg))
        problems = problems or world_problems(cfg)
    except Exception as exc:                                    # noqa: BLE001 -- reported on the protocol
        problems = [f'{type(exc).__name__}: {exc}']
    if problems:                                                # refused BEFORE Isaac starts: nothing written
        progress('invalid_config', problems=problems)
        print('TRAINIT_FAILED')
        sys.exit(2)
    import numpy as np
    from trainit_data.adapters.isaac_replicator import ReplicatorIngest
    from trainit_data.store import Store, resolve_store_root
    from trainit_data.synthetic import SCENE_SCHEMA, run_seed, semantics_of

    from isaacsim import SimulationApp
    app = SimulationApp({'headless': not args.gui})
    import omni.kit.app
    import omni.replicator.core as rep
    import omni.timeline
    import omni.usd
    from isaacsim.core.utils.semantics import add_labels
    from pxr import Gf, UsdGeom, UsdLux

    def ext_version(name):
        mgr = omni.kit.app.get_app().get_extension_manager()
        ext_id = mgr.get_enabled_extension_id(name) or ''
        return ext_id.split('-', 1)[1] if '-' in ext_id else 'unknown'

    try:
        from isaacsim.core.version import get_version
        isaac_version = '.'.join(str(v) for v in get_version()[2:5])
    except Exception:                                           # noqa: BLE001 -- recorded as unknown, never guessed
        isaac_version = 'unknown'

    # -- the base stage (static, before any randomization) --
    omni.usd.get_context().new_stage()
    stage = omni.usd.get_context().get_stage()
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.Xform.Define(stage, '/World')
    floor = UsdGeom.Cube.Define(stage, '/World/Floor')
    floor.CreateSizeAttr(1.0)
    UsdGeom.Xformable(floor).AddTranslateOp().Set(Gf.Vec3d(0, 0, -0.005))
    UsdGeom.Xformable(floor).AddScaleOp().Set(Gf.Vec3f(2.0, 2.0, 0.01))
    floor.CreateDisplayColorAttr([Gf.Vec3f(0.35, 0.35, 0.35)])
    tray = cfg['tray']
    ix, iy, wh, wt = tray['inner_m'][0], tray['inner_m'][1], tray['wall_height_m'], tray['wall_thickness_m']
    for name, (cx, cy, sx, sy) in {'north': (0, iy / 2 + wt / 2, ix + 2 * wt, wt),
                                   'south': (0, -iy / 2 - wt / 2, ix + 2 * wt, wt),
                                   'east': (ix / 2 + wt / 2, 0, wt, iy), 'west': (-ix / 2 - wt / 2, 0, wt, iy)}.items():
        wall = UsdGeom.Cube.Define(stage, f'/World/Tray/{name}')
        wall.CreateSizeAttr(1.0)
        UsdGeom.Xformable(wall).AddTranslateOp().Set(Gf.Vec3d(cx, cy, wh / 2))
        UsdGeom.Xformable(wall).AddScaleOp().Set(Gf.Vec3f(sx, sy, wh))
        wall.CreateDisplayColorAttr([Gf.Vec3f(0.2, 0.3, 0.6)])
    dome = UsdLux.DomeLight.Define(stage, '/World/Lights/Dome')
    dome.CreateIntensityAttr(800.0)
    key = UsdLux.DistantLight.Define(stage, '/World/Lights/Key')
    key.CreateIntensityAttr(2000.0)
    key_rotate = UsdGeom.Xformable(key).AddRotateXYZOp()
    key_rotate.Set(Gf.Vec3f(0.0, 0.0, 0.0))
    cam_cfg = cfg['camera']
    camera = UsdGeom.Camera.Define(stage, CAMERA_PRIM)
    camera.CreateProjectionAttr(UsdGeom.Tokens.perspective)
    camera.CreateFocalLengthAttr(cam_cfg['focal_length'])
    camera.CreateHorizontalApertureAttr(cam_cfg['horizontal_aperture'])
    width, height = cfg['resolution']
    camera.CreateVerticalApertureAttr(cam_cfg['horizontal_aperture'] * height / width)
    camera.CreateClippingRangeAttr(Gf.Vec2f(*cam_cfg['clipping_m']))     # the default near plane is 1 m
    UsdGeom.Xformable(camera).AddTranslateOp().Set(Gf.Vec3d(0, 0, cam_cfg['height_m']))
    UsdGeom.Xform.Define(stage, PARTS)
    base_layer = stage.GetRootLayer().ExportToString()
    base_sha = 'sha256:' + hashlib.sha256(base_layer.encode('utf-8')).hexdigest()

    generator_sha = sha256_file(Path(__file__))
    scene = {'schema': SCENE_SCHEMA, 'kind': 'PROCEDURAL',
             'generator': {'name': 'isaac_replicator_bin', 'sha256': generator_sha},
             'base_stage': {'sha256': base_sha, 'completeness': 'RESOLVED_LAYERS', 'layers': [], 'unresolved': []},
             'assets': [], 'semantics': semantics_of(cfg['classes'], SEMANTIC_RULE),
             'limitations': ['procedural scene: built in memory by the generator (no stage file, no external layer '
                             'or asset); its identity is the generator hash + the base root layer as exported '
                             'before randomization + the config + the recorded draws',
                             'rendered pixels are not promised bit-identical across GPUs, drivers or Isaac '
                             'versions (the claim is SAME_DRAWS)']}

    # -- Replicator: a render product and its annotators --
    rp = rep.create.render_product(CAMERA_PRIM, (width, height))
    annotators = {
        'rgb': rep.AnnotatorRegistry.get_annotator('rgb'),
        'depth': rep.AnnotatorRegistry.get_annotator('distance_to_image_plane'),
        'instances': rep.AnnotatorRegistry.get_annotator('instance_id_segmentation', init_params={'colorize': False}),
        'boxes': rep.AnnotatorRegistry.get_annotator('bounding_box_2d_tight'),
        'camera': rep.AnnotatorRegistry.get_annotator('camera_params'),
    }
    for annotator in annotators.values():
        annotator.attach([rp])
    for _ in range(cfg['warmup_steps']):                         # the first frames are stale
        rep.orchestrator.step(rt_subframes=cfg['rt_subframes'])
    timeline = omni.timeline.get_timeline_interface()

    resolved = dict(cfg, runs=args.runs, views_per_run=args.views)
    environment = {'isaac_sim': isaac_version, 'replicator': ext_version('omni.replicator.core'),
                   'numpy': np.__version__, 'python': sys.version.split()[0], 'renderer': 'RTX real-time'}
    store = Store.open(store_root(args, resolve_store_root), create=True)
    ingest = ReplicatorIngest(
        store, sensor_id=SENSOR_ID, frame_id=FRAME_ID, config=resolved, master_seed=args.seed, recipe=cfg['recipe'],
        runs=args.runs, views_per_run=args.views, scene=scene, camera_params=annotators['camera'].get_data(),
        camera_prim=CAMERA_PRIM, environment=environment,
        software={'generator': 'examples/isaac_replicator_bin/generate.py', 'generator_sha256': generator_sha,
                  'isaac_sim': isaac_version, 'replicator': environment['replicator']},
        rights={'classification': args.classification},
        backend_details={'rt_subframes': cfg['rt_subframes'], 'mode': 'A (placed, no physics)'},
        config_file_sha256=sha256_file(config_path))
    print(f'session {ingest.session_id}: {args.runs} runs x {args.views} views, seed {args.seed}')
    progress('session', session_id=ingest.session_id, runs=args.runs, views=args.views)

    header = None
    try:
        for run_index in range(args.runs):
            rng = np.random.default_rng(run_seed(args.seed, run_index))
            draws = draw_run(rng, cfg)
            if run_index:
                stage.RemovePrim(f'{PARTS}/r{run_index - 1:04d}')
            group = f'{PARTS}/r{run_index:04d}'
            UsdGeom.Xform.Define(stage, group)
            objects = {}
            for obj in draws['objects']:
                path = f'{group}/{obj["object"]}'
                size = obj['size_m']
                if obj['class'] == 'cube':
                    prim = UsdGeom.Cube.Define(stage, path)
                    prim.CreateSizeAttr(size)
                elif obj['class'] == 'cylinder':
                    prim = UsdGeom.Cylinder.Define(stage, path)
                    prim.CreateRadiusAttr(size / 2)
                    prim.CreateHeightAttr(size)
                    prim.CreateAxisAttr('Z')
                else:
                    prim = UsdGeom.Sphere.Define(stage, path)
                    prim.CreateRadiusAttr(size / 2)
                xform = UsdGeom.Xformable(prim)
                xform.AddTranslateOp().Set(Gf.Vec3d(*obj['position_m']))
                xform.AddRotateZOp().Set(obj['yaw_deg'])
                prim.CreateDisplayColorAttr([Gf.Vec3f(*obj['color_rgb'])])
                add_labels(prim.GetPrim(), labels=[obj['class']], instance_name='class')
                objects[path] = {'object': obj['object'], 'class': obj['class']}
            # scene validation: the stage holds exactly the drawn objects where they were drawn (fail loudly)
            for obj in draws['objects']:
                placed = UsdGeom.Xformable(stage.GetPrimAtPath(f'{group}/{obj["object"]}'))
                at = placed.ComputeLocalToWorldTransform(0).ExtractTranslation()
                if max(abs(a - b) for a, b in zip(at, obj['position_m'])) > 1e-6:
                    raise RuntimeError(f'run {run_index}: {obj["object"]} is not where it was drawn')
            for view in range(args.views):
                view_draws = draw_view(rng, cfg)
                dome.GetIntensityAttr().Set(view_draws['dome_intensity'])
                key.GetIntensityAttr().Set(view_draws['key_intensity'])
                key_rotate.Set(Gf.Vec3f(0.0, 90.0 - view_draws['key_elevation_deg'], view_draws['key_azimuth_deg']))
                # The real-time renderer accumulates over frames (temporal AA / upscaling): with a fixed camera the
                # previous scene lingers as faint ghosts that no label explains. Settle before the captured frame.
                for _ in range(cfg['settle_steps']):
                    rep.orchestrator.step(rt_subframes=cfg['rt_subframes'])
                rep.orchestrator.step(rt_subframes=cfg['rt_subframes'])
                seg = annotators['instances'].get_data()
                boxes = annotators['boxes'].get_data()
                id_to_prim = {int(k): str(v) for k, v in seg['info']['idToLabels'].items()}
                seen = {objects[p]['object'] for p in objects
                        if any(q == p and np.any(seg['data'] == k) for k, q in id_to_prim.items())}
                missing = sorted({o['object'] for o in draws['objects']} - seen)
                if missing:
                    raise RuntimeError(f'run {run_index} view {view}: drawn objects {missing} are not in the rendered '
                                       'instance mask (the renderer skipped them, or the scene is not the draws)')
                tight = [{'prim': str(p), 'x_min': int(r['x_min']), 'y_min': int(r['y_min']),
                          'x_max': int(r['x_max']), 'y_max': int(r['y_max']),
                          'occlusion': float(r['occlusionRatio'])}
                         for p, r in zip(boxes['info']['primPaths'], boxes['data'])]
                ingest.add_frame(run_index=run_index, view=view, draws=draws, view_draws=view_draws,
                                 sim_time_s=float(timeline.get_current_time()), rgb=annotators['rgb'].get_data(),
                                 depth=annotators['depth'].get_data(), instance_ids=seg['data'],
                                 id_to_prim=id_to_prim, objects=objects, tight_boxes=tight,
                                 camera_params=annotators['camera'].get_data())
            print(f'run {run_index + 1}/{args.runs}: {len(draws["objects"])} objects')
            progress('run', run=run_index + 1, runs=args.runs, objects=len(draws['objects']))
        progress('sealing')
        header = ingest.close()
    except BaseException:                                        # noqa: BLE001 -- reported, then Kit is closed
        ingest.abandon()                                         # left ACTIVE: `trainit-data recover` seals it
        traceback.print_exc(file=sys.stdout)
    finally:
        for annotator in annotators.values():
            annotator.detach([rp.path])
        rp.destroy()
    if header is not None:
        print(json.dumps({'session_id': header.session_id, 'captures': header.counters['captures'],
                          'calibrations': list(header.calibration_ids)}))
        print(f'TRAINIT_SESSION {header.session_id}')
    else:
        print('TRAINIT_FAILED')
    app.close()
    sys.exit(0 if header is not None else 1)


if __name__ == '__main__':
    main()
