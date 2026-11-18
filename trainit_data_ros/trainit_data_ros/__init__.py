"""TrainIt Data Foundation: live ROS acquisition (ADR-0029).

The recorder (the CaptureCoordinator) reads the ROS camera contract of ``trainit_hardware_layer`` and
writes ``trainit_data``'s canonical records:
- it is the only place where ROS messages become canonical records;
- it runs in the robot's slot (ADR-0023);
- TSA configures, launches and observes it, and never hosts it (ADR-0027 as narrowly amended by
  ADR-0029 Decision 7).

Slice D0.2: ``recorder`` (the CaptureCoordinator node: MANUAL + INTERVAL triggers, the
FIRST_AT_OR_AFTER_BOUNDARY v1 policy, CameraInfo / TF -> CalibrationRecord, a bounded writer queue,
diagnostics) and ``replay`` (a sealed session back onto the camera contract). The ROS-free parts --
``recording_config``, ``coordinator``, ``conversions``, ``writer_worker`` -- are unit-tested without a graph.
"""

__version__ = '0.10.1'
