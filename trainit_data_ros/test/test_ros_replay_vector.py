"""D0.9 (independent review): replay re-publishes the CAMERA contract only. A session that also holds a non-image
stream (a ``vector`` IMU reading, D0-J) replays its images; the reading is skipped by name, never an error per tick.
"""
import copy

import numpy as np
import pytest

rclpy = pytest.importorskip('rclpy', reason='the replay test needs a sourced ROS 2 Humble')

from ros_graph import Graph, wait_for  # noqa: E402
from trainit_data.store import Store  # noqa: E402
from trainit_data.timestamps import Timestamp  # noqa: E402
from trainit_data.vocab import Vocabulary  # noqa: E402
from trainit_data.writer import ImageSample, VectorSample  # noqa: E402

TOOL = 'handheld_tool'


@pytest.fixture
def graph():
    g = Graph()
    yield g
    g.close()


def tool_session(root):
    store = Store.open(root, create=True)
    doc = copy.deepcopy(Vocabulary.default().to_dict())
    doc['backends'].append('capture_tool')
    doc['data_blocks']['vector'] = {'version': '1.0'}
    doc['modalities']['imu'] = {'data_block': 'vector', 'encodings': [], 'payloads': []}
    writer = store.create_session(
        source={'transport': 'LIVE_ROS', 'backend': 'capture_tool'}, clock={'session_domain': 'ROS_TIME'},
        sensors=[{'sensor_id': TOOL, 'streams': [{'stream': 'color', 'modality': 'rgb'},
                                                 {'stream': 'imu', 'modality': 'imu'}]}],
        vocabulary=Vocabulary.from_dict(doc))
    for i in range(2):
        at = Timestamp(100 + i, 0, 'ROS_TIME', 'HEADER')
        writer.append_capture(
            trigger={'trigger_id': 'manual', 'kind': 'MANUAL', 'firing_index': i, 'origin': 'OPERATOR',
                     'issued': {'stamp': at.to_dict()}, 'reason': {}},
            outcome='COMPLETE', sync={'policy': {'name': 'FIRST_AT_OR_AFTER_BOUNDARY', 'version': 1, 'params': {}},
                                      'reference': f'{TOOL}/color', 'status': 'SYNCED', 'max_skew_ns': 0},
            samples=[VectorSample(TOOL, 'imu', at, 'handheld_tool_imu_frame', [0.0, 9.81], ['m/s^2', 'm/s^2']),
                     ImageSample(TOOL, 'color', at, np.full((6, 8, 3), 40 + i, np.uint8), 'rgb8',
                                 'handheld_tool_optical_frame')])
    return writer.close().session_id


def test_a_session_with_a_vector_stream_replays_its_images_only(graph, tmp_path):
    from sensor_msgs.msg import Image
    from trainit_data_ros.replay_node import SENSOR_QOS, Replay
    root = tmp_path / 'TrainItData'
    session_id = tool_session(root)
    received = []
    helper = graph.helper('replay_listener')
    helper.create_subscription(Image, '/replay_ns/color/image_raw', received.append, SENSOR_QOS)
    replay = graph.add(Replay(context=graph.ctx, parameter_overrides=graph.params(
        {'session_id': session_id, 'store_root': str(root), 'camera_namespace': '/replay_ns', 'rate_hz': 20.0})))
    assert replay.skipped == [f'{TOOL}/imu (imu)']
    assert wait_for(lambda: len(received) >= 4, 10.0), f'{len(received)} image(s) received'
    assert replay.errors == 0                       # the vector reading never reaches load_payload
    assert {bytes(m.data[:1]) for m in received} == {bytes([40]), bytes([41])}     # both captures' images
    assert set(replay.pubs) == {(TOOL, 'color')}
