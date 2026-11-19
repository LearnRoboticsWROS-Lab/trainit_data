# trainit_data — TrainIt Data Foundation

TrainIt's canonical acquisition, dataset and provenance layer (D0). Live RGB-D acquisition, simulation and
pre-recorded data all become the SAME canonical records:
- calibration identity;
- explicit clock domains;
- sync quality;
- trigger provenance;
- immutable dataset versions;
- export to training formats.

**Status: version 0.10.2 (slices D0.1 - D0.9: the complete D0 Data Foundation; M4.5: YOLO segmentation export;
P0.a / P0.b: the robot task server's job runs and its machine-service requests).**
- D0.1, the pure-Python core: the record contracts, the vocabulary, the local store, the crash-safe session
  writer, `verify` / `recover`, and the `trainit-data` CLI.
- D0.2 - D0.5, `trainit_data_ros`: the live recorder over the ROS camera contract (mock, Isaac, the physical
  D455; MANUAL, INTERVAL, EVENT and APPLICATION_STATE triggers) and the replay publisher.
- D0.6: datasets, immutable versions, label sets, group-aware splits, the YOLO detection exporter and label import.
- D0.7: synthetic data. `trainit_data.adapters.isaac_replicator` (pure numpy) turns Isaac Replicator annotator
  output into a canonical session. `trainit-data labels from-synthetic` derives SYNTHETIC_GROUND_TRUTH label sets
  (INSTANCE_MASK; BBOX_2D, direct or mask-derived). Splits can group by randomization run. The neutral example is
  `examples/isaac_replicator_bin/`.
- D0.8: the TSA Data panel drives the recorder and this CLI (TSA never imports this repository); a selectable
  frame-pairing policy, `FIRST_ALIGNED_AT_OR_AFTER_BOUNDARY` v1, beside the default `FIRST_AT_OR_AFTER_BOUNDARY` v1.
- D0.9: the failure suite (D0-I) and the contract proof (D0-J): a backend, a modality and a sync policy added as
  vocabulary rows only, including a low-rate non-image reading (the `vector` data block, e.g. one IMU sample).
  The shipped vocabulary has no `vector` / `imu` row: a deployment that records one adds the rows (IMU recording is
  not available out of the box). A new calibration must hold a calibrated camera matrix (an all-zero or REP-104
  uncalibrated K is refused); a stored one stays readable and `verify` flags it.
- M4.5 (0.10.0): `trainit-data export yolo --task segment`, the YOLO segmentation export of INSTANCE_MASK label
  sets (exporter `yolo` 1.1; see "Exporting a dataset version" below).
- P0.a (0.10.1): `trainit_data_ros` follows `trainit_interfaces` 0.4.0 (ExecutionEvent `job_id` / `result_code`): every
  job run of a robot task server is adopted from its own RUN_STARTED.
- P0.b (0.10.2): a machine-service `request` PROCESS_EVENT is admitted and fires an `edge: any` EVENT trigger on its
  channel (`rising` / `falling` never: a request is not a process switching on or off).
- Not yet: EPISODES and high-rate series (D1), LeRobot (D3), YOLO segmentation label import, COCO.

## Install

In a ROS 2 Humble colcon workspace, next to `trainit_hardware_layer` and `trainit_interfaces` (which
`trainit_data_ros` needs):

```bash
cd ~/trainit_ws/src
git clone https://github.com/LearnRoboticsWROS-Lab/trainit_data.git
cd ~/trainit_ws && rosdep install --from-paths src --ignore-src -y
colcon build --packages-select trainit_data trainit_data_ros
source install/setup.bash && python3 -m trainit_data --version
```

The core alone, without ROS (a training environment, Isaac's own Python):

```bash
cd trainit_data                                  # this repository
pip install ./trainit_data                       # numpy + Pillow; no ROS, no simulator, no training framework
python3 -m trainit_data --version
```

## Two packages, one version line

| Package | Role | Depends on |
|---|---|---|
| `trainit_data` | the canonical core: persisted data contracts, store, versions, labels, splits, exporters, non-ROS ingest adapters (`trainit_data.adapters`), the `trainit-data` CLI | Python ≥ 3.10, numpy, Pillow. **No ROS, no simulator, no training framework, no other TrainIt package** |
| `trainit_data_ros` | live acquisition: the recorder over the ROS camera contract (mock / Isaac / real alike), triggers, the replay publisher | `trainit_data`, `trainit_hardware_layer`, rclpy and standard ROS messages |

`trainit_data` installs with colcon (ament_python) and with plain `pip install ./trainit_data`, so a training or
provider environment and Isaac's own Python can read and write TrainIt datasets without the ROS tree.

## Boundaries

**What this repository does not own:**
- **External formats.** YOLO, COCO, LeRobotDataset, DP3 datasets, rosbag2 / MCAP and Isaac Replicator output are
  exporters, transports or ingest adapters. They never define TrainIt data.
- **Perception.** Interpreting observations (detection, segmentation, pose) is `trainit_perception`'s; this
  repository owns no perception.
- **Configuration and execution.** TSA configures and launches the recorder. TMR and TP never import this
  repository, and nor does TSA.
- **D1 / D2 / D3.**
  - D0 records observations only.
  - Robot / task / action timelines (D1), demonstration sources (D2) and learning-provider exports such as
    LeRobotDataset (D3) come later. They extend these records and never redesign them.
  - No training framework is ever imported here.

## The records

| Record | Schema | What it is |
|---|---|---|
| vocabulary | `trainit-vocab/1.0` | extensible registry DATA: backends, modalities, encodings, transports, trigger kinds / origins, context namespaces, sync policies, stamp origins, missing-stream reasons, provenance kinds. A session pins it by content hash |
| `Timestamp` | (embedded) | `sec`, `nanosec`, `clock_domain`, `origin` (+ `sim_source`); never compared across clock domains (`INCOMPATIBLE_CLOCKS`) |
| `Observation` | (embedded in a capture) | a common header + a modality-typed data block: `image` (rgb / depth, a payload file), `instance_image` (synthetic instance masks), `vector` (one low-rate non-image reading held in the record, no payload) |
| `Capture` | `trainit-capture/1.0` | one trigger firing: trigger record, sync result (policy name + version), outcome `COMPLETE` / `PARTIAL` / `FAILED` with a reason per missing stream |
| `CalibrationRecord` | `trainit-calibration/1.0` | intrinsics, depth semantics, extrinsics + provenance; content-addressed id `cal-<16 hex>` |
| session header | `trainit-session/1.0` | one acquisition / ingest run: `ACTIVE` → `CLOSED` (writer) or `INTERRUPTED` (recover). Not a dataset |

**Public API:** the modules listed in `trainit_data/__init__.py`. Modules starting with `_` are internal.

**On disk** (`$TRAINIT_DATA_ROOT`, default `~/TrainItData`):

```
store.json  vocab/vocab-<h>.json  calibrations/cal-<h>.json
sessions/ses-<utc>-<hex>/session.json  captures.jsonl  payload/<seq>/<sensor>__<stream>.png|.npy  orphans/
```

**Payloads (lossless, never converted):**
- rgb8 → PNG;
- 16UC1 depth (mm) → 16-bit PNG;
- 32FC1 depth (m) → float32 `.npy` (NaN / inf preserved).

**Crash safety:** payload → `.tmp` → fsync → rename, and only then the fsynced index line. `recover` truncates a torn
index tail (keeping a copy), MOVES unreferenced files to `orphans/` and seals the session `INTERRUPTED`. Nothing is
ever deleted.

**CLI:**

```bash
trainit-data inspect [SESSION]            # ros2 run trainit_data trainit-data ..., or python3 -m trainit_data ...
trainit-data verify [SESSION]             # read-only; exit 1 on any ERROR finding
trainit-data recover SESSION              # an interrupted ACTIVE session -> INTERRUPTED
trainit-data --json ...                   # one trainit-cli/1 object (the machine contract)
```

## Recording live data (D0.2)

```bash
export TRAINIT_DATA_ROOT=~/TrainItData                       # the default
ros2 run trainit_hardware_layer mock_camera                  # any backend publishing the camera contract
ros2 launch trainit_data_ros recorder.launch.py namespace:=r1 \
    config_file:=$(ros2 pkg prefix trainit_data_ros)/share/trainit_data_ros/config/mock_rgbd.recording.yaml
ros2 service call /r1/trainit_data_recorder/start_session std_srvs/srv/Trigger
ros2 service call /r1/trainit_data_recorder/capture std_srvs/srv/Trigger      # MANUAL
ros2 service call /r1/trainit_data_recorder/close_session std_srvs/srv/Trigger
trainit-data verify                                          # or: python3 -m trainit_data verify
ros2 run trainit_data_ros replay --ros-args -p session_id:=<id>   # a sealed session back onto the contract
```

The recording configuration (`trainit-recording/1.0`) holds only what the user chose; the recorder resolves every
default (topics and frames from the camera contract and the hardware profile, the sync tolerance from the frame
rate) and stores the effective configuration in the session.

## Exporting a dataset version

```bash
trainit-data export yolo DATASET VERSION DEST                      # detection: BBOX_2D label sets
trainit-data export yolo DATASET VERSION DEST --task segment \
    [--epsilon-px 0.5] [--min-area-px 16] [--label-set ls-...]      # segmentation: INSTANCE_MASK label sets
trainit-data verify-export DEST                                    # hashes, counts, label validity, data.yaml;
                                                                   # with the store: regenerated and compared
```

`DEST` is a missing or empty directory outside every repository and the store. The layout is Ultralytics':
`images/<split>/<stem>.png` (the canonical payload, byte for byte), `labels/<split>/<stem>.txt`, `data.yaml`, and
the sidecar `trainit_export.json` (exporter name, version and config, the source version, label sets and split
pinned by hash, every file's sha256). Two exports of the same version and config are byte-identical.

**Segmentation** (`--task segment`, exporter `yolo` 1.1): one line `<class> x1 y1 x2 y2 ...` per instance,
normalized, 6 decimals. The polygon policy (`largest_component_outer_boundary` v1, pure numpy, no cv2) is
deterministic:
- per instance id of the mask, the LARGEST 8-connected component is kept (a tie goes to the first in raster order);
  the other components are dropped and counted;
- its OUTER boundary is traced (holes are ignored: a YOLO polygon has none, a ring exports filled);
- Douglas-Peucker with `epsilon_px` (default 0.5) simplifies it, with exact integer arithmetic;
- vertices are the PIXEL CENTRES of boundary pixels: `(col + 0.5) / W`, `(row + 0.5) / H`, so a rasterizer that
  truncates to integers and fills the polygon (Ultralytics' `polygon2mask`) gets the boundary pixels back;
- an instance whose kept component has fewer than `min_area_px` pixels (default 16), or whose polygon is
  degenerate (a one-pixel line), is dropped; an image whose every instance is dropped exports an empty label file;
- a mask id without a class in the label set is refused (`EXPORT_REFUSED`).

The default epsilon keeps a round-trip IoU >= 0.95 on discs of radius >= 10 px (1.0 would drop to ~0.92).
`epsilon_px` and `min_area_px` are part of the config (and of the export id). The sidecar's `summary.polygons`
records the policy with its parameters, the instances seen and exported, the vertices, the multi-component
instances with the dropped components and their pixels, and every dropped instance with its reason. An export made by
`yolo` 1.0 still verifies on its own files; against the store `verify-export` warns that this core (1.1) cannot
regenerate it.

## Data never lives in this repository

**Where data goes:** captured datasets go to the data store. The default is `~/TrainItData`; override it with
`TRAINIT_DATA_ROOT`. The store is never inside a software repository.

| What | Path |
|---|---|
| software | `~/fr5_ws/src/trainit_data` |
| data | `~/TrainItData` |

**Rights:**
- Customer data, customer assets, model weights and private training material are never committed here.
- The Community licence of this SOFTWARE does not make any dataset public.

## Design

The design lives in the TrainIt engineering overlay:
- `docs/adr/0029-trainit-data-foundation.md`;
- `docs/engineering/D0_DESIGN.md`;
- `docs/engineering/D0_SCHEMA_PROPOSAL.md`.

## Tests

From this repository, in a sourced workspace (the core suite also runs in a plain Python):

```bash
(cd trainit_data && PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest -q -p no:cacheprovider test)
(cd trainit_data_ros && PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest -q -p no:cacheprovider test)
```

## Licence

Apache-2.0 (`LICENSE`). TrainIt Community (ADR-0029 Decision 2), subject to the publication / licence
verification before any public release.
