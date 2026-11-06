"""Synthetic, customer-neutral fixtures for the D0.6 tests (datasets, versions, labels, splits, exports).

Two sealed sessions from two DIFFERENT sources -- one simulated (``isaac``, SIM_TIME), one physical (``real``,
ROS_TIME), with different sensor ids -- so every dataset rule is exercised on mixed sources, as D0-H requires.
Every value is invented (neutral names, 8x6 images, illustrative intrinsics).
"""
import numpy as np

from _helpers import FRAME, HEIGHT, WIDTH, calibration, depth_mm, rgb, stamp, sync
from trainit_data.writer import ImageSample

SIM_SENSOR = 'overhead_camera'
REAL_SENSOR = 'bench_camera'
CLASSES = ['part', 'tray']


def _sensors(sensor_id):
    return [{'sensor_id': sensor_id, 'profile': {'model': 'example_rgbd'},
             'streams': [{'stream': 'color', 'modality': 'rgb'}, {'stream': 'depth', 'modality': 'depth'}]}]


def _trigger(at, index):
    return {'trigger_id': 'manual', 'kind': 'MANUAL', 'firing_index': index, 'origin': 'OPERATOR',
            'issued': {'stamp': at.to_dict()}, 'reason': {'note': 'synthetic'}}


def make_session(store, vocab, *, simulated: bool, complete=3, partial=1, rights=None, seed=0):
    """A sealed session with ``complete`` COMPLETE captures then ``partial`` PARTIAL ones (depth missing)."""
    sensor = SIM_SENSOR if simulated else REAL_SENSOR
    if simulated:
        source = {'transport': 'LIVE_ROS', 'backend': 'isaac', 'descriptor': {'stamp_origin': 'HEADER'}}
        clock = {'session_domain': 'SIM_TIME', 'sim_source': 'isaac', 'use_sim_time': True}
        kw = {'domain': 'SIM_TIME', 'sim_source': 'isaac'}
    else:
        source = {'transport': 'LIVE_ROS', 'backend': 'real', 'descriptor': {'stamp_origin': 'HEADER'}}
        clock = {'session_domain': 'ROS_TIME', 'use_sim_time': False}
        kw = {}
    color_cal = store.put_calibration(calibration(vocab, 'color', sensor_id=sensor))
    depth_cal = store.put_calibration(calibration(vocab, 'depth', sensor_id=sensor))
    spec = dict(source=source, clock=clock, sensors=_sensors(sensor))
    if rights is not None:
        spec['rights'] = rights
    writer = store.create_session(**spec)
    for i in range(complete + partial):
        at = stamp(100 + i, **kw)
        color = ImageSample(sensor, 'color', at, rgb(seed + i), 'rgb8', FRAME, calibration_id=color_cal)
        if i < complete:
            depth = ImageSample(sensor, 'depth', at, depth_mm(seed + i), '16UC1', FRAME, unit_scale_m=0.001,
                                calibration_id=depth_cal)
            writer.append_capture(trigger=_trigger(at, i), outcome='COMPLETE',
                                  sync=sync(reference=f'{sensor}/color'), samples=[color, depth])
        else:
            writer.append_capture(trigger=_trigger(at, i), outcome='PARTIAL',
                                  sync=sync('SINGLE_STREAM', None, reference=f'{sensor}/color'), samples=[color],
                                  missing=[{'stream': f'{sensor}/depth', 'reason': 'TIMEOUT'}])
    return writer.close()


def color_obs(header, seq):
    sensor = header.sensors[0].sensor_id
    return f'{header.session_id}/{seq:06d}/{sensor}/color'


def box(cls='part', x=1.0, y=1.0, w=4.0, h=3.0):
    return {'class': cls, 'bbox_xywh_px': [x, y, w, h]}


def yolo_line(cls_index, x, y, w, h):
    return f'{cls_index} {(x + w / 2) / WIDTH:.6f} {(y + h / 2) / HEIGHT:.6f} {w / WIDTH:.6f} {h / HEIGHT:.6f}\n'


__all__ = ['make_session', 'color_obs', 'box', 'yolo_line', 'CLASSES', 'WIDTH', 'HEIGHT', 'np']
