"""Non-ROS ingest adapters (ADR-0029 Decision 3).

An adapter turns already-extracted arrays and metadata from a producer that is not a ROS source --
Isaac Replicator annotators, files, a future Capture Tool log -- into the SAME canonical records the
live recorder writes. Backend-specific concepts stay here and in the opaque ``backend_details`` block.

The dependency direction is one-way: adapters may import the core, and the core (contracts, store,
exporters) never imports ``trainit_data.adapters`` (``test/test_core_purity.py``).

``isaac_replicator`` (D0.7): Replicator annotator arrays -> one canonical synthetic session.
"""
