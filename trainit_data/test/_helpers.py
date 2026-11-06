"""Synthetic, customer-neutral fixtures for the D0.1 tests (D0_DESIGN.md §21.4: neutral names and values).

Every value here is invented: a neutral ``overhead_camera`` with an 8x6 image, illustrative intrinsics,
``base_link`` / ``overhead_camera_optical_frame`` frames. ``test_fixture_scrub.py`` keeps it that way.
"""
import numpy as np

from trainit_data.calibration import CalibrationRecord
from trainit_data.timestamps import Timestamp
from trainit_data.writer import ImageSample

SENSOR = 'overhead_camera'
FRAME = 'overhead_camera_optical_frame'
WIDTH, HEIGHT = 8, 6


def calibration_content(stream='color', **overrides) -> dict:
    content = {
        'sensor_id': SENSOR, 'stream': stream, 'optical_frame': FRAME,
        'intrinsics': {'width': WIDTH, 'height': HEIGHT, 'distortion_model': 'plumb_bob',
                       'K': [4.0, 0, 4.0, 0, 4.0, 3.0, 0, 0, 1], 'D': [0, 0, 0, 0, 0],
                       'R': [1, 0, 0, 0, 1, 0, 0, 0, 1], 'P': [4.0, 0, 4.0, 0, 0, 4.0, 3.0, 0, 0, 0, 1, 0]},
        'intrinsics_provenance': {'kind': 'SYNTHETIC_RENDER', 'notes': 'synthetic test camera'},
        'extrinsics': [{'parent_frame': 'base_link', 'child_frame': FRAME, 'translation': [0.5, 0.0, 0.5],
                        'rotation_xyzw': [0.7071068, -0.7071068, 0.0, 0.0],
                        'provenance': {'kind': 'NOMINAL_DESCRIPTION', 'ref': {'description_package': 'example_cell'}}}],
        'transform_tree': {'source': 'tf_static'},
    }
    if stream == 'depth':
        content['depth'] = {'invalid': 'zero', 'registered_to': 'color', 'registration_basis': 'same_render_product'}
    content.update(overrides)
    return content


def calibration(vocab, stream='color', **overrides) -> CalibrationRecord:
    return CalibrationRecord.build(calibration_content(stream, **overrides), vocab)


def stamp(sec=100, nanosec=0, domain='ROS_TIME', origin='HEADER', sim_source=None) -> Timestamp:
    return Timestamp(sec, nanosec, domain, origin, sim_source)


def trigger(at: Timestamp, index=0, kind='MANUAL', origin='OPERATOR') -> dict:
    return {'trigger_id': 'manual', 'kind': kind, 'firing_index': index, 'origin': origin,
            'issued': {'stamp': at.to_dict(), 'host_monotonic_ns': 1000 + index},
            'reason': {'note': 'synthetic', 'caller': 'test'}}


def sync(status='SYNCED', skew=0, tolerance=None, reference=f'{SENSOR}/color') -> dict:
    out = {'policy': {'name': 'FIRST_AT_OR_AFTER_BOUNDARY', 'version': 1, 'params': {'timeout_ms': 1000}},
           'reference': reference, 'status': status}
    if skew is not None:
        out['max_skew_ns'] = skew
    if tolerance is not None:
        out['tolerance_ns'] = tolerance
    return out


def rgb(seed=0) -> np.ndarray:
    return np.random.default_rng(seed).integers(0, 256, (HEIGHT, WIDTH, 3), dtype=np.uint8)


def depth_mm(seed=0) -> np.ndarray:
    array = np.random.default_rng(seed).integers(0, 65536, (HEIGHT, WIDTH), dtype=np.uint16)
    array[0, 0], array[0, 1] = 0, 65535                   # invalid (zero) and the full range survive
    return array


def depth_m(seed=0) -> np.ndarray:
    array = np.random.default_rng(seed).random((HEIGHT, WIDTH), dtype=np.float32) * 3.0
    array[0, 0], array[0, 1], array[0, 2] = np.nan, np.inf, -np.inf
    return array


SENSORS = [{'sensor_id': SENSOR, 'profile': {'model': 'example_rgbd'},
            'streams': [{'stream': 'color', 'modality': 'rgb', 'topic': '/camera/color/image_raw'},
                        {'stream': 'depth', 'modality': 'depth', 'topic': '/camera/depth/image_raw'}]}]


def open_session(store, *, clock=None, sensors=None, source=None, **kwargs):
    return store.create_session(
        source=source or {'transport': 'LIVE_ROS', 'backend': 'mock',
                          'descriptor': {'stamp_origin': 'HEADER', 'supplied_by': 'test'}},
        clock=clock or {'session_domain': 'ROS_TIME', 'use_sim_time': False},
        sensors=sensors or SENSORS, **kwargs)


def rgbd(at: Timestamp, seed=0, *, color_cal=None, depth_cal=None, depth='16UC1', stream_stamps=None):
    stream_stamps = stream_stamps or {}
    color = ImageSample(SENSOR, 'color', stream_stamps.get('color', at), rgb(seed), 'rgb8', FRAME,
                        calibration_id=color_cal)
    if depth == '16UC1':
        d = ImageSample(SENSOR, 'depth', stream_stamps.get('depth', at), depth_mm(seed), '16UC1', FRAME,
                        unit_scale_m=0.001, calibration_id=depth_cal)
    else:
        d = ImageSample(SENSOR, 'depth', stream_stamps.get('depth', at), depth_m(seed), '32FC1', FRAME,
                        unit_scale_m=1.0, calibration_id=depth_cal)
    return [color, d]


def snapshot(root) -> dict:
    """{relative path: bytes} of every file under ``root`` (to prove an operation changed / deleted nothing)."""
    return {p.relative_to(root).as_posix(): p.read_bytes() for p in sorted(root.rglob('*')) if p.is_file()}
