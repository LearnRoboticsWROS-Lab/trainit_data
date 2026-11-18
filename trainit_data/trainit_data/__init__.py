"""TrainIt Data Foundation: the canonical core (ADR-0029).

This package owns TrainIt's PERSISTED data contracts (``docs/engineering/D0_SCHEMA_PROPOSAL.md`` in the
TrainIt overlay). It is pure Python by contract: no rclpy or ROS message type, no simulator (omni /
Isaac), no pydantic, no cv2 on the core path, no training framework, and no other TrainIt package. That
is what lets a training environment and Isaac's own Python use it.

PUBLIC API (slices D0.1, D0.6 and D0.7). Import from these modules; everything else is internal.

=====================  =================================================================================
``errors``             ``DataError`` and the stable error codes
``vocab``              ``Vocabulary`` (``trainit-vocab/1.0``, extensible registry data)
``timestamps``         ``Timestamp``, ``compare_timestamps``, ``skew_ns`` (``INCOMPATIBLE_CLOCKS``)
``observation``        ``Observation`` = header + a data block (``ImageBlock``; ``InstanceImageBlock``, D0.7)
``capture``            ``Capture``, ``TriggerRecord``, ``SyncResult``, ``Missing`` (``trainit-capture/1.0``)
``calibration``        ``CalibrationRecord`` (``trainit-calibration/1.0``, content-addressed)
``session_record``     ``SessionHeader`` (``trainit-session/1.0``), ``Source``, ``Clock``, ``SensorDecl``
``store``              ``Store``, ``resolve_store_root`` (``~/TrainItData`` / ``TRAINIT_DATA_ROOT``)
``writer``             ``SessionWriter`` (via ``Store.create_session``), ``ImageSample``, ``VectorSample``
``reader``             ``SessionReader`` (via ``Store.open_session``)
``integrity``          ``verify_session``, ``verify_store``, ``recover_session``
``dataset``            ``DatasetIdentity`` (``trainit-dataset-identity/1.0``), ``create_dataset``, ``combine_rights``
``labels``             ``LabelSet`` (``trainit-labels/1.0``, 1.1: INSTANCE_MASK), ``create_label_set``, ``read_entries``
``splits``             ``SplitAssignment`` (``trainit-split/1.0``, 1.1: ``group_by``), ``build_split`` (GROUP_HASH)
``versions``           ``DatasetVersion`` (``trainit-dataset/1.0``, 1.1), ``create_version``, ``capture_splits`` ...
``dataset_integrity``  ``verify_dataset``
``synthetic``          D0.7: randomization runs (``run_id``, ``group_of``), the generation block
                       (``trainit-randomization/1.0``), the scene manifest (``trainit-synthetic-scene/1.0``),
                       ``compare_generations`` (D0-G)
``ground_truth``       D0.7: ``create_synthetic_label_set`` (recorded annotator output -> SYNTHETIC_GROUND_TRUTH)
``exports``            the export sidecar (``trainit-export/1.0``), ``verify_export``
``formats``            the exporter registry; ``formats.yolo``: YOLO detection and (M4.5) segmentation export +
                       ``import_yolo_labels``
``cli``                the ``trainit-data`` command (``--json``: ``trainit-cli/1``)
=====================  =================================================================================

INTERNAL (may change without notice): ``_canonical`` (the one canonical-JSON / sha256 mechanism),
``_strict``, ``_fsio``, ``_codecs``, and any name starting with ``_``. The one sanctioned exception is
``trainit_data_ros`` (same repository, same version line, checked by its tests), which uses ``_canonical`` and
``_strict`` so its configuration parses and hashes exactly like the core's records. Inside the public modules, helpers
that take vocabulary rows or ``_strict`` callables (``Vocabulary.checker``, the ``from_dict`` ``where``
argument, ``ImageBlock.from_dict``) are internal plumbing too: construct records by keyword.

ALSO PUBLIC, and the longest-lived contract: the persisted formats -- the record schemas above, the CLI's
``trainit-cli/1`` JSON, ``trainit-recovery/1.0`` and the store layout (``store.json``,
``vocab/vocab-<16 hex>.json``, ``calibrations/cal-<16 hex>.json``, ``sessions/<id>/{session.json,
captures.jsonl, payload/<seq:06d>/<sensor>__<stream>.<ext>, .active, orphans/, recovery.json}``,
``datasets/<id>/{dataset.json, labels/<ls-id>/{labelset.json, entries.jsonl}, versions/<n>/{manifest.json,
selection.jsonl, splits.json}}``) and the export sidecar ``trainit_export.json``. A session is not self-contained:
its pinned vocabulary and its calibrations live in the store beside it.

NOT in the core (by scope): ingest adapters (``trainit_data.adapters``; D0.7: ``isaac_replicator``, pure numpy, no
Isaac import; the core never imports them),
EPISODES content (D1; reserved, refused until its reader exists), external-format adapters beyond YOLO detection /
segmentation (COCO, LeRobot: later slices), and any training framework. Live ROS acquisition is the sibling
package ``trainit_data_ros`` (D0.2).
"""

__version__ = '0.10.1'
