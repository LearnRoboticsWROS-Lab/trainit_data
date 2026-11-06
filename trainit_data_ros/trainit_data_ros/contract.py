"""The ROS camera contract as the recorder reads it (``trainit_hardware_layer/contracts/camera.md``).

The recorder introduces NO topic convention of its own: a stream name of the contract maps to the
contract's relative topic names under the camera namespace, and to its modality. Depth is the contract's
REGISTERED depth (camera.md, "Registered depth", D0.2a): it carries the colour optical frame and the
colour pixel grid.
"""

# stream -> (image topic, camera_info topic, modality, optical-frame key in the profile, registered_to)
CAMERA_STREAMS = {
    'color': ('color/image_raw', 'color/camera_info', 'rgb', 'color_optical', None),
    'depth': ('depth/image_rect_raw', 'depth/camera_info', 'depth', 'color_optical', 'color'),
}

# The OPTIONAL latched source-information document of a camera (camera.md, "Source information"):
# std_msgs/String carrying trainit-camera-source/1.0 JSON, RELIABLE + TRANSIENT_LOCAL.
SOURCE_INFO = 'source_info'

# The runner's canonical execution events (trainit_interfaces/ExecutionEvent, ADR-0030): RELATIVE, resolved in the
# recorder's robot slot (ADR-0023) exactly as the runner publishes it; RELIABLE, VOLATILE, keep-last 100.
EXECUTION_EVENTS = 'trainit/execution_events'

# REP-118: the image encoding states the unit of a depth image. Nothing else is accepted for depth.
DEPTH_UNIT_SCALE_M = {'16UC1': 0.001, '32FC1': 1.0}
DEPTH_INVALID = {'16UC1': 'zero', '32FC1': 'nan_or_inf'}


def join(namespace: str, relative: str) -> str:
    namespace = namespace.rstrip('/')
    return f'{namespace}/{relative}' if namespace else relative
