"""ROS message -> canonical facts (duck-typed: no ROS import, unit-testable without a graph).

This is the ONLY place where the contract's ``sensor_msgs`` shapes become canonical values (ADR-0029
Decision 4). Nothing is converted silently: an encoding outside the modality's vocabulary row, a wrong byte
order or a short buffer is a refusal, never a reinterpretation.
"""
import numpy as np

from .contract import DEPTH_INVALID

# encoding -> (numpy dtype, channels) for the encodings the contract allows (the vocabulary decides which).
_LAYOUT = {'rgb8': ('u1', 3), 'mono8': ('u1', 1), '16UC1': ('u2', 1), 'mono16': ('u2', 1), '32FC1': ('f4', 1)}


class ImageRefused(Exception):
    """The image cannot be stored as declared (the coordinator records ENCODING_REFUSED)."""


def image_to_array(msg) -> np.ndarray:
    """A COPY of an Image's pixels as an HxW(xC) little-endian-native array (step padding removed)."""
    if msg.encoding not in _LAYOUT:
        raise ImageRefused(f'encoding {msg.encoding!r} is not stored by this recorder')
    kind, channels = _LAYOUT[msg.encoding]
    dtype = np.dtype(('>' if msg.is_bigendian else '<') + kind)
    row = msg.width * channels * dtype.itemsize
    data = bytes(msg.data)
    if msg.height == 0 or msg.width == 0 or msg.step < row or len(data) < msg.step * msg.height:
        raise ImageRefused(f'{msg.width}x{msg.height} {msg.encoding}: step {msg.step}, {len(data)} bytes')
    rows = np.frombuffer(data, np.uint8, count=msg.step * msg.height).reshape(msg.height, msg.step)[:, :row]
    array = rows.copy().view(dtype).reshape((msg.height, msg.width) if channels == 1
                                            else (msg.height, msg.width, channels))
    return array.astype(dtype.newbyteorder('='), copy=False)


def intrinsics(info) -> dict:
    """A CameraInfo, verbatim, as the canonical intrinsics block."""
    return {'width': int(info.width), 'height': int(info.height), 'distortion_model': str(info.distortion_model),
            'K': [float(v) for v in info.k], 'D': [float(v) for v in info.d], 'R': [float(v) for v in info.r],
            'P': [float(v) for v in info.p], 'binning': [int(info.binning_x), int(info.binning_y)],
            'roi': {'x': int(info.roi.x_offset), 'y': int(info.roi.y_offset), 'width': int(info.roi.width),
                    'height': int(info.roi.height), 'do_rectify': bool(info.roi.do_rectify)}}


def calibration_content(*, sensor_id, stream, optical_frame, info_block, intrinsics_kind, intrinsics_notes=None,
                        encoding=None, registered_to=None, registration_basis=None, extrinsic=None,
                        moving=False, description_sha256=None) -> dict:
    """The content of a ``trainit-calibration/1.0`` record (its identity is computed by trainit_data).

    ``extrinsic``: ``(parent, translation, rotation_xyzw, provenance_kind)`` as the runtime's TF held it.
    ``transform_tree.source`` is DECLARED from the configuration (``tf`` for a moving camera, else
    ``tf_static``), not observed. ``description_sha256`` (the configured robot / camera description files,
    hashed at session start) answers "which description did this nominal extrinsic come from".
    """
    content = {'sensor_id': sensor_id, 'stream': stream, 'optical_frame': optical_frame, 'intrinsics': info_block,
               # The CameraInfo's CONTENT is the identity; the topic it arrived on is not (it is in the session's
               # recording configuration): the same calibration on two topic names is one record.
               'intrinsics_provenance': {'kind': intrinsics_kind, 'ref': {'source': 'sensor_msgs/CameraInfo'}},
               'extrinsics': [], 'transform_tree': {'source': 'tf' if moving else 'tf_static'}}
    if intrinsics_notes is not None:
        content['intrinsics_provenance']['notes'] = intrinsics_notes
    if description_sha256 is not None:
        content['transform_tree']['description_sha256'] = description_sha256
    if registered_to is not None or encoding in DEPTH_INVALID:
        depth = {'invalid': DEPTH_INVALID[encoding]}
        if registered_to is not None:
            depth.update(registered_to=registered_to, registration_basis=registration_basis)
        content['depth'] = depth
    if extrinsic is not None:
        parent, translation, rotation, kind = extrinsic
        content['extrinsics'] = [{'parent_frame': parent, 'child_frame': optical_frame,
                                  'translation': [float(v) for v in translation],
                                  'rotation_xyzw': [float(v) for v in rotation],
                                  'provenance': {'kind': kind}}]
        if description_sha256 is not None and kind == 'NOMINAL_DESCRIPTION':
            # A nominal extrinsic comes FROM the description; a measured one does not (it carries its own ref).
            content['extrinsics'][0]['provenance']['ref'] = {'description_sha256': description_sha256}
    return content


def same_geometry(a: dict, b: dict) -> bool:
    """Two intrinsics blocks describe the same pixel grid and projection (registered depth, camera.md)."""
    return a == b
