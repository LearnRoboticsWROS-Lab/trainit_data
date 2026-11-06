# Isaac Replicator → canonical TrainIt data (customer-neutral example, D0.7)

A neutral scene (a tray with cubes, cylinders and spheres, a dome and a key light, a fixed overhead camera),
randomized by a seeded generator and rendered with Isaac Replicator. The result is ONE ordinary canonical TrainIt
session. Labels, dataset versions and the YOLO export come afterwards from the canonical records, never from
Replicator.

```
generate.py (Isaac Sim's Python: the only code here that imports omni / isaacsim / pxr)
   | numpy arrays + JSON metadata
trainit_data.adapters.isaac_replicator  (pure: no Isaac import)
   |
session (INGEST / isaac, SIM_TIME; color rgb8, depth 32FC1 m, instances 16-bit mask; SYNTHETIC_RENDER +
         SYNTHETIC_SCENE calibration; the generation block and the scene manifest)
   | trainit-data labels from-synthetic
LabelSet (SYNTHETIC_GROUND_TRUTH: INSTANCE_MASK, BBOX_2D)
   | trainit-data version create --group-by randomization_run
DatasetVersion  ->  trainit-data export yolo  (the one YOLO path)
```

It needs Isaac Sim 5.1 (tested: 5.1.0, Replicator 1.12.27, headless) and nothing else: the generator puts this
repository's `trainit_data` core on its path.

## Run it

From the repository root (`env -i` keeps a sourced ROS environment out of Kit):

```bash
env -i HOME=$HOME PATH=/usr/bin:/bin TRAINIT_DATA_ROOT=$HOME/TrainItData \
    $HOME/isaacsim/python.sh examples/isaac_replicator_bin/generate.py --seed 20261102 --runs 24 --views 2
# ... TRAINIT_SESSION ses-<...>
```

Then, with the core installed (ROS workspace or `pip install ./trainit_data`):

```bash
S=ses-<...>
trainit-data verify $S
trainit-data dataset create example_bin --name "Example bin primitives"
trainit-data labels from-synthetic example_bin --session $S --kind BBOX_2D               # the annotator's boxes
trainit-data labels from-synthetic example_bin --session $S --kind INSTANCE_MASK
trainit-data version create example_bin --session $S --label-set <ls-...> --group-by randomization_run
trainit-data export yolo example_bin 1 /tmp/example_bin_yolo
```

## As a synthetic provider (D0.8)

`provider.json` declares this example as the REFERENCE synthetic provider (`trainit-synthetic-provider/1.0`, read by
`trainit_data.synthetic_provider`): its runtime, entry script, base configuration, the seed / runs / views bounds,
every parameter a tool may expose (each a real key of `bin_config.json`, with type, bounds, unit, label, group and
level; its default is the base configuration's value), its outputs, and the progress protocol. A launching tool (the
TSA Data panel) renders its form from the descriptor, writes a resolved configuration and runs this script as an
external process:

```bash
<isaac python.sh> generate.py --config <resolved.json> --seed N --runs N --views N [--store DIR]
# TRAINIT_PROGRESS {"phase": "starting"}                ... "session", "run" (n / N), "sealing"
# TRAINIT_SESSION ses-<...>     or     TRAINIT_FAILED
```

The script validates the configuration and the seed / runs / views against the descriptor, and checks that the tray
stays in the camera's view, BEFORE Isaac starts; a refused configuration prints
`TRAINIT_PROGRESS {"phase": "invalid_config", "problems": [...]}`, then `TRAINIT_FAILED`, and exits 2 with nothing
written. It is installed with `trainit_data` under `share/trainit_data/synthetic_providers/isaac_replicator_bin/` and
finds its core in the same install prefix. Parameters the generator does not have (orientation ranges, camera pose,
light colour, backgrounds, distractors, noise) are not declared, so no tool can offer them.

## What is recorded

- **Per run** (one randomized scene; a split group): the objects (count, class, size, position, yaw, colour), in
  the capture's `trigger.reason.draws`; the run id `rr-…` is the hash of the recipe + those draws.
- **Per view**: the lights (`trigger.reason.view_draws`).
- **Per session**: the resolved config and its file hash, the master seed, the recipe, the run / view counts, the
  generator's hash, the Isaac Sim / Replicator / numpy versions, the base stage's hash (exported BEFORE
  randomization), the semantic classes, and the stated limitations.

## Reproducibility

The same config + seed in the same supported environment records the same runs, ordering, draws, semantic
assignments and calibrations. Check it with:

```bash
trainit-data compare-generation <session A> <session B>
```

Rendered pixels are NOT promised bit-identical (GPU, driver and Isaac version all play a part). The comparison
reports whether they happened to be, without requiring it.

## Checks that speak about the world

The run stops loudly if a drawn object is not where it was drawn on the stage, or is missing from the rendered
instance mask.

The real-time renderer accumulates over frames (temporal antialiasing / upscaling). With a fixed camera, the
previous run's objects linger in the RGB as faint ghosts that no label explains, and the masks and depth do not
show them. Found on the first real-store acceptance (2026-11-02). The generator therefore renders `settle_steps`
frames after every scene or lighting change before the captured one.

Isaac Sim 5.1 skips a mesh whose prim path is re-defined as another primitive type, so each run's objects get a
path of their own. The adapter refuses a frame whose annotator box and mask extent disagree by more than 1 px.

## Not here

No customer asset, geometry or name. No physics: objects are placed, not poured (Mode A in
`ISAAC_REPLICATOR.md`). No YOLO writer: the canonical exporter is the only one. No training.
