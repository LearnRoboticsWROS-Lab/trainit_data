# CLAUDE.md — trainit_data (TrainIt Data Foundation)

**Repository-local instructions.** The product-level bootstrap is the overlay's `CLAUDE.md` (one directory up, in
`fr5_ws/src`): read it first. Rules that bind here:

- **Language.** Talk to the user in Italian; write everything in this repository in English.
- **Design authority.**
  - ADR-0029 and `docs/engineering/D0_DESIGN.md` / `D0_SCHEMA_PROPOSAL.md` in the overlay.
  - Implement only the slice the owner has given a GO for (`D0_DESIGN.md` §21). Progress is recorded in
    `docs/engineering/D0_CHECKPOINT.md`.
- **The core `trainit_data` stays ROS-free.**
  - Forbidden: rclpy, ROS messages, omni / Isaac, pydantic, cv2 on the core path, torch / LeRobot / ultralytics,
    and any other TrainIt package.
  - Non-ROS ingest code lives in `trainit_data.adapters.*`, and the core never imports it.
  - `test/test_core_purity.py` enforces this. Never weaken it to make a change pass.
- **Internal core modules:** only `trainit_data_ros` may import `trainit_data._canonical` / `_strict` (same
  repository, same release; `test_package_boundary.py` checks the versions match). Nothing else may.
- **`trainit_data_ros` keeps its logic ROS-free:** `recording_config`, `coordinator`, `conversions` and
  `writer_worker` import no rclpy and are unit-tested without a graph; `recorder_node` / `replay_node` only wire
  ROS to them. Its ROS tests run on a private domain (87) against an install that has `trainit_hardware_layer`
  >= 0.2.2. A callback must never raise (a multi-threaded executor can stop on it); store I/O happens only in
  the writer thread.
- **`trainit_data_ros` dependencies.** It depends only on `trainit_data` and `trainit_hardware_layer`
  (+ `trainit_interfaces` from slice D0.5). Never TSA, TMR, TP, Pro or a cell. TSA / TMR / TP never import this
  repository: the overlay's `tools/architecture_guards/check_data_boundary.py` checks it.
- **No data in the repository.**
  - Captured data goes to `~/TrainItData` (`TRAINIT_DATA_ROOT`), never here.
  - Test fixtures are small and synthetic, with customer-neutral names and values.
  - Never copy customer code, assets, measurements or data (e.g. from private program repositories) into this
    public repository.
  - The pre-push guard refuses payload files that are not in its reviewed allow-list.
- **One version line.** Both packages carry the same version, in `package.xml`, `setup.py` and `__init__.py`.
- **Git.** No push, no GitHub repository creation, no tag: those are owner actions. Never run destructive git.
- **Testing.** Never declare a test passed unless it ran. Tests use
  `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest -p no:cacheprovider`.
  - Run the core suite in BOTH interpreters: the ROS Python 3.10 and Isaac Lab's Python 3.11 (its `env_isaaclab`
    venv, without `PYTHONPATH`). `test/test_canonical.py` pins literal hashes, so passing in both proves the
    canonical form is identical.
  - A literal hash in a test changes only with an intended canonical-form change, shown to the owner.
- **One canonical mechanism.** Every hash, content-addressed id and index line goes through
  `trainit_data/_canonical.py`. Never add a second JSON / hash routine.
- **Record rules live in the record's `from_dict`.** Code that builds a record serialises it and parses it back.
  Rules that need the session (declared streams, session clock) live in the writer and are re-checked by `verify`.
