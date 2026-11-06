"""Neutral fixtures for the trainit_data_ros tests (no ROS import here)."""
import copy
from types import SimpleNamespace

import numpy as np

FRAME = 'overhead_camera_color_optical_frame'

PROFILE = {
    'topics': {'namespace': '/overhead_camera'},
    'frames': {'color_optical': FRAME, 'depth_optical': 'overhead_camera_depth_optical_frame'},
    'sensor': {'color': {'width': 8, 'height': 6, 'fps': 30}, 'depth': {'width': 8, 'height': 6, 'fps': 30}},
    'backends': {'mock': {'source_descriptor': {'backend': 'mock', 'intrinsics_kind': 'NOMINAL_PROFILE',
                                                'stamp_origin': 'HEADER', 'registration_basis': 'same_pixel_grid'}}},
}


def load_profile(model):
    assert model == 'example_rgbd'
    return copy.deepcopy(PROFILE)


def config(**overrides):
    doc = {'schema': 'trainit-recording/1.0', 'source': {'backend': 'mock'},
           'sensors': [{'sensor_id': 'overhead_camera', 'profile': 'example_rgbd',
                        'streams': {'color': {}, 'depth': {}}}],
           'triggers': [{'id': 'manual', 'kind': 'MANUAL'}],
           'sync': {'policy': 'FIRST_AT_OR_AFTER_BOUNDARY', 'policy_version': 1}}
    doc.update(overrides)
    return doc


def info_msg(frame=FRAME, width=8, height=6, fx=4.0):
    roi = SimpleNamespace(x_offset=0, y_offset=0, width=0, height=0, do_rectify=False)
    return SimpleNamespace(width=width, height=height, distortion_model='plumb_bob', d=[0.0] * 5,
                           k=[fx, 0.0, 4.0, 0.0, fx, 3.0, 0.0, 0.0, 1.0], r=[1.0, 0, 0, 0, 1.0, 0, 0, 0, 1.0],
                           p=[fx, 0.0, 4.0, 0.0, 0.0, fx, 3.0, 0.0, 0.0, 0.0, 1.0, 0.0], binning_x=0, binning_y=0,
                           roi=roi, header=SimpleNamespace(frame_id=frame))


def image_msg(array, encoding, frame=FRAME, step_pad=0, bigendian=False):
    channels = 1 if array.ndim == 2 else array.shape[2]
    row = array.shape[1] * channels * array.dtype.itemsize
    body = array.astype(array.dtype.newbyteorder('>' if bigendian else '<')).tobytes()
    data = b''.join(body[i * row:(i + 1) * row] + b'\0' * step_pad for i in range(array.shape[0]))
    return SimpleNamespace(height=array.shape[0], width=array.shape[1], encoding=encoding,
                           is_bigendian=int(bigendian), step=row + step_pad, data=data,
                           header=SimpleNamespace(frame_id=frame))


def rgb(seed=0):
    return np.random.default_rng(seed).integers(0, 256, (6, 8, 3), dtype=np.uint8)


def depth_mm(seed=0):
    return np.random.default_rng(seed).integers(0, 65536, (6, 8), dtype=np.uint16)
