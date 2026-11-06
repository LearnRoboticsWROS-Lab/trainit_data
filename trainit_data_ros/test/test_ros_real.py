"""D0.4: a REAL-camera-shaped source through the SAME recorder, and its source topology (no RealSense code path).

The camera contract is served by the hardware layer's mock (registered 16UC1 depth in millimetres, the colour optical
frame, both CameraInfos: what the real backend's launch exposes), and the camera's latched ``<ns>/source_info`` is a
``trainit-camera-source/1.0`` document of the real backend. The recording configuration is the SHIPPED D0-C file
(``config/realsense_d455_standalone.recording.yaml``). The physical D455 run is the owner-guided D0-C acceptance;
these tests pin the behaviour without a device:

- the session is canonical and verifies, with the real backend's descriptor and the camera's document recorded;
- LOCAL vs REMOTE comes from the boot-id evidence, and a remote source changes NOTHING in Observation / Capture
  (the Jetson topology needs no canonical-schema change);
- a required document that is missing, or one that contradicts the descriptor, refuses the session;
- a replay keeps the original's device provenance.
"""
import json
import time
from pathlib import Path

import pytest
import yaml

rclpy = pytest.importorskip('rclpy', reason='the acquisition tests need a sourced ROS 2 Humble')

from ros_graph import Graph, static_tf, wait_for  # noqa: E402
from test_ros_acquisition import SERVICE, SLOT, cli_verify, mock_camera  # noqa: E402
from trainit_data._canonical import sha256_json  # noqa: E402
from trainit_data.integrity import verify_session  # noqa: E402
from trainit_data.store import Store  # noqa: E402
from trainit_hardware_layer.camera import source_info as source_doc  # noqa: E402

CONFIG = Path(__file__).resolve().parents[1] / 'config' / 'realsense_d455_standalone.recording.yaml'
FRAME = 'camera_color_optical_frame'
SERIAL = '000000000455'


@pytest.fixture
def graph():
    g = Graph()
    yield g
    g.close()


@pytest.fixture
def store_root(tmp_path):
    root = tmp_path / 'TrainItData'
    Store.open(root, create=True)
    return root


def d0c_config(**overrides):
    doc = yaml.safe_load(CONFIG.read_text())
    doc['triggers'] = [{'id': 'manual', 'kind': 'MANUAL'}]
    doc.update(overrides)
    return doc


def camera_document(*, boot=None, clock_domain='global_time', serial=SERIAL):
    host = source_doc.host_facts()
    if boot is not None:
        host['boot_id_sha256'] = boot
    domains = {'color': clock_domain, 'depth': clock_domain}
    timestamping = {'origin': 'OBSERVED', 'clock_domains': domains, 'frames_observed': {'color': 30, 'depth': 30}}
    if source_doc.stamp_origin(domains):
        timestamping['stamp_origin'] = source_doc.stamp_origin(domains)
    return source_doc.build(
        backend='real', published_wall_utc_ns=time.time_ns(),
        device={'origin': 'DISCOVERED', 'name': 'realsense_d455', 'serial_number': serial,
                'firmware_version': '0.0.0.0'},
        attachment={'origin': 'DISCOVERED', 'via': 'sysfs', 'kind': 'USB', 'port_id': '1-1', 'speed_mbps': 5000,
                    'hub': None},
        host=host, deployment={'origin': 'DECLARED', 'namespace': '/camera'}, timestamping=timestamping)


def publish_source_info(graph, document, name='test_source_info'):
    from rclpy.qos import QoSDurabilityPolicy, QoSHistoryPolicy, QoSProfile, QoSReliabilityPolicy
    from std_msgs.msg import String
    node = graph.helper(name)
    qos = QoSProfile(reliability=QoSReliabilityPolicy.RELIABLE, durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
                     history=QoSHistoryPolicy.KEEP_LAST, depth=1)
    node._pub = node.create_publisher(String, '/camera/source_info', qos)
    node.send = lambda doc: node._pub.publish(String(data=json.dumps(doc, sort_keys=True)))
    node.send(document)
    return node


def start(graph, tmp_path, store_root, doc, name='real'):
    from trainit_data_ros.recorder_node import Recorder
    path = tmp_path / f'{name}.recording.yaml'
    path.write_text(yaml.safe_dump(doc))
    return graph.add(Recorder(context=graph.ctx, namespace=SLOT, parameter_overrides=graph.params(
        {'config_file': str(path), 'store_root': str(store_root)})))


def ready(rec, documents=True):
    return wait_for(lambda: all(v > 3 for v in rec.coordinator.received.values())
                    and (not documents or 'bench_camera' in rec._source_info), 10.0)


def session(graph, store_root, captures=3):
    client = graph.helper('test_client')
    started = graph.call(client, f'{SERVICE}/start_session')
    assert started.success, started.message
    answers = [graph.call(client, f'{SERVICE}/capture') for _ in range(captures)]
    closed = graph.call(client, f'{SERVICE}/close_session')
    assert closed.success, closed.message
    return Store.open(store_root).open_session(started.message), answers, json.loads(closed.message)


def real_session(graph, tmp_path, store_root, document, captures=3):
    mock_camera(graph)
    static_tf(graph, parent='camera_mount')
    publish_source_info(graph, document)
    rec = start(graph, tmp_path, store_root, d0c_config())
    assert ready(rec)
    return session(graph, store_root, captures)


def _shape(value):
    """The key structure of a record (values dropped): what a topology must NOT change."""
    if isinstance(value, dict):
        return {k: _shape(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_shape(v) for v in value[:1]]
    return type(value).__name__


def test_d0c_shape_a_real_camera_session_records_its_device_and_topology(graph, tmp_path, store_root):
    document = camera_document()
    reader, answers, closed = real_session(graph, tmp_path, store_root, document)
    assert all(a.success and a.message.split()[1] == 'COMPLETE' for a in answers), [a.message for a in answers]
    header = reader.header
    assert header.source.backend == 'real'
    assert header.source.descriptor == {'intrinsics_kind': 'DEVICE_REPORTED', 'stamp_origin': 'DRIVER_GLOBAL_TIME',
                                        'registration_basis': 'driver_align_depth', 'supplied_by':
                                        'trainit_hardware_layer profile realsense_d455 backends.real'}
    block = header.provenance['acquisition']
    assert block['schema'] == 'trainit-acquisition/1.0'
    assert block['transport']['kind'] == 'ROS2_DDS' and block['transport']['ros_domain_id'] >= 0
    assert block['recorder']['host']['origin'] == 'OBSERVED'
    entry = block['sensors']['bench_camera']
    assert entry['relation'] == 'LOCAL_SENSOR_LOCAL_RECORDER' and entry['stamp_origin_verified'] is True
    assert entry['source_info'] == document and entry['source_info_sha256'] == sha256_json(document)
    assert entry['source_info_topic'] == '/camera/source_info'
    description = header.provenance['robot']['description']
    assert description['root'] == {'package': 'trainit_hardware_layer', 'file': 'urdf/d455_standalone.urdf.xacro'}
    store = Store.open(store_root)
    for capture in reader.captures():
        assert capture.sync.status == 'SYNCED'
        colour, depth = capture.observations
        for obs in capture.observations:
            assert obs.stamp.clock_domain == 'ROS_TIME' and obs.stamp.origin == 'DRIVER_GLOBAL_TIME'
            assert obs.data.frame_id == FRAME
            cal = store.get_calibration(obs.data.calibration_id, reader.vocab)
            assert cal.intrinsics_provenance.kind == 'DEVICE_REPORTED'
            assert cal.extrinsics[0].parent_frame == 'camera_mount'
            assert cal.extrinsics[0].provenance.kind == 'NOMINAL_DESCRIPTION'          # never MEASURED_HAND_EYE
            assert cal.extrinsics[0].provenance.ref == {'description_sha256': description['sha256']}
        assert (depth.data.encoding, depth.data.unit_scale_m) == ('16UC1', 0.001)
        depth_cal = store.get_calibration(depth.data.calibration_id, reader.vocab)
        assert depth_cal.depth.registered_to == 'color' and depth_cal.depth.invalid == 'zero'
        assert depth_cal.depth.registration_basis == 'driver_align_depth'
        assert 'serial' not in json.dumps(capture.to_dict())          # device facts live in provenance only
    assert closed['warnings'] == []
    assert verify_session(store, reader.session_id).ok
    code, _ = cli_verify(store_root, reader.session_id)
    assert code == 0


def ahead_camera(graph, ahead_ms):
    """The contract mock, its stamps running ``ahead_ms`` ahead of this host's clock (a device-clock fit gone wrong)."""
    import rclpy.time
    camera = mock_camera(graph)
    real_clock = camera.get_clock()

    class Ahead:
        def now(self):
            return rclpy.time.Time(nanoseconds=real_clock.now().nanoseconds + int(ahead_ms * 1e6),
                                   clock_type=real_clock.clock_type)
    camera.get_clock = Ahead                               # _tick stamps with get_clock().now()
    return camera


@pytest.mark.parametrize('boot, expected', [(None, 'FAILED'), ('e' * 64, 'COMPLETE')])
def test_a_same_host_source_stamping_after_receipt_is_never_complete(graph, tmp_path, store_root, boot, expected):
    """Local (same boot): a stamp after its own receipt is not this host's clock -> CLOCK_MISMATCH, a warning.
    Remote: two hosts' clocks are never compared this way (the capture is judged as before)."""
    ahead_camera(graph, 300)
    static_tf(graph, parent='camera_mount')
    publish_source_info(graph, camera_document(boot=boot))
    rec = start(graph, tmp_path, store_root, d0c_config())
    assert ready(rec)
    reader, answers, closed = session(graph, store_root, captures=2)
    assert [a.message.split()[1] for a in answers] == [expected, expected], [a.message for a in answers]
    if expected == 'FAILED':
        assert [w['code'] for w in closed['warnings']] == ['STAMPS_AFTER_RECEIPT']
        assert {m.reason for c in reader.captures() for m in c.missing} == {'CLOCK_MISMATCH'}
    else:
        assert closed['warnings'] == []
    assert verify_session(Store.open(store_root), reader.session_id).ok


def test_a_remote_source_changes_the_relation_and_nothing_in_the_captures(graph, tmp_path, store_root):
    local, _, _ = real_session(graph, tmp_path, store_root, camera_document())
    graph.reset()
    remote, _, _ = real_session(graph, tmp_path, store_root, camera_document(boot='e' * 64))
    acq = remote.header.provenance['acquisition']['sensors']['bench_camera']
    assert acq['relation'] == 'REMOTE_SENSOR_SOURCE'
    assert local.header.provenance['acquisition']['sensors']['bench_camera']['relation'] == \
        'LOCAL_SENSOR_LOCAL_RECORDER'
    local_captures, remote_captures = list(local.captures()), list(remote.captures())
    assert [_shape(c.to_dict()) for c in local_captures] == [_shape(c.to_dict()) for c in remote_captures]
    assert local.header.schema == remote.header.schema
    assert list(local.header.calibration_ids) == list(remote.header.calibration_ids)    # same device, same id
    assert verify_session(Store.open(store_root), remote.session_id).ok


def test_a_required_source_document_that_never_arrived_refuses_the_session(graph, tmp_path, store_root):
    mock_camera(graph)
    static_tf(graph, parent='camera_mount')
    rec = start(graph, tmp_path, store_root, d0c_config())
    assert ready(rec, documents=False)
    client = graph.helper('test_client')
    started = graph.call(client, f'{SERVICE}/start_session')
    assert not started.success and 'SOURCE_DESCRIPTOR_INVALID' in started.message
    assert '/camera/source_info' in started.message
    assert Store.open(store_root).list_sessions() == []                   # nothing written


@pytest.mark.parametrize('document, needle', [
    (lambda: camera_document(clock_domain='system_time'), 'OBSERVED stamp origin'),     # RECEIPT, not global time
    (lambda: dict(camera_document(), backend='isaac'), 'backend'),
])
def test_a_source_document_that_contradicts_the_session_refuses_it(graph, tmp_path, store_root, document, needle):
    mock_camera(graph)
    static_tf(graph, parent='camera_mount')
    publish_source_info(graph, document())
    rec = start(graph, tmp_path, store_root, d0c_config())
    assert ready(rec)
    started = graph.call(graph.helper('test_client'), f'{SERVICE}/start_session')
    assert not started.success and needle in started.message, started.message


def test_a_malformed_source_document_refuses_the_session_and_writes_nothing(graph, tmp_path, store_root):
    from rclpy.qos import QoSDurabilityPolicy, QoSHistoryPolicy, QoSProfile, QoSReliabilityPolicy
    from std_msgs.msg import String
    mock_camera(graph)
    static_tf(graph, parent='camera_mount')
    node = graph.helper('test_bad_source_info')
    node._pub = node.create_publisher(String, '/camera/source_info', QoSProfile(
        reliability=QoSReliabilityPolicy.RELIABLE, durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
        history=QoSHistoryPolicy.KEEP_LAST, depth=1))
    node._pub.publish(String(data='not json'))
    rec = start(graph, tmp_path, store_root, d0c_config())
    assert ready(rec)
    started = graph.call(graph.helper('test_client'), f'{SERVICE}/start_session')
    assert not started.success and 'INVALID_VALUE' in started.message, started.message
    assert Store.open(store_root).list_sessions() == []
    assert rec.state == 'IDLE' and rec._session_source_info is None


def test_a_document_whose_publisher_is_gone_is_stale_never_current_evidence(graph, tmp_path, store_root):
    mock_camera(graph)
    static_tf(graph, parent='camera_mount')
    publisher = publish_source_info(graph, camera_document())
    rec = start(graph, tmp_path, store_root, d0c_config())
    assert ready(rec)                                        # received while the camera's node was alive
    graph.remove(publisher)                                  # the camera node exits (e.g. the device unplugged)
    assert wait_for(lambda: rec.count_publishers('/camera/source_info') == 0, 5.0)
    started = graph.call(graph.helper('test_client'), f'{SERVICE}/start_session')
    assert not started.success and 'no source document' in started.message, started.message
    assert 'bench_camera' not in rec._source_info


def test_a_source_document_changed_during_a_session_is_reported(graph, tmp_path, store_root):
    mock_camera(graph)
    static_tf(graph, parent='camera_mount')
    publisher = publish_source_info(graph, camera_document())
    rec = start(graph, tmp_path, store_root, d0c_config())
    assert ready(rec)
    client = graph.helper('test_client')
    started = graph.call(client, f'{SERVICE}/start_session')
    assert started.success, started.message
    publisher.send(camera_document(serial='000000000999'))                   # e.g. the driver restarted on another
    assert wait_for(lambda: rec.source_info_changes == 1, 5.0)
    closed = json.loads(graph.call(client, f'{SERVICE}/close_session').message)
    assert [w['code'] for w in closed['warnings']] == ['SOURCE_INFO_CHANGED']
    header = Store.open(store_root).open_session(started.message).header
    assert header.provenance['acquisition']['sensors']['bench_camera']['source_info']['device']['serial_number'] \
        == SERIAL                                                             # the one valid at the start
    assert [w['code'] for w in header.warnings] == ['SOURCE_INFO_CHANGED']


def test_a_real_session_replays_with_its_device_provenance_retained(graph, tmp_path, store_root):
    from trainit_data_ros.replay_node import Replay
    original, _, _ = real_session(graph, tmp_path, store_root, camera_document())
    graph.reset()                                          # the camera, its document and the recorder are gone
    graph.add(Replay(context=graph.ctx, parameter_overrides=graph.params(
        {'session_id': original.session_id, 'store_root': str(store_root), 'rate_hz': 10.0})))
    doc = d0c_config(source={'backend': 'replay', 'replay_of': original.session_id})
    doc.pop('provenance')                                  # a replay inherits the description provenance
    doc['sensors'][0]['source_info'] = {'required': False}  # the replay publisher is no device
    rec = start(graph, tmp_path, store_root, doc, name='replay')
    assert ready(rec, documents=False)
    replayed, answers, _ = session(graph, store_root)
    assert all(a.success for a in answers), [a.message for a in answers]
    header = replayed.header
    assert header.source.backend == 'replay' and header.source.replay_of == original.session_id
    assert header.source.descriptor['intrinsics_kind'] == 'DEVICE_REPORTED'
    block = header.provenance['acquisition']
    assert block['sensors']['bench_camera']['relation'] == 'UNKNOWN'
    assert block['original'] == {'session_id': original.session_id,
                                 'acquisition': original.header.provenance['acquisition']}
    assert block['original']['acquisition']['sensors']['bench_camera']['source_info']['device']['serial_number'] \
        == SERIAL
    assert list(header.calibration_ids) == list(original.header.calibration_ids)
    originals = {o.stream: original.load_payload(o).tobytes() for c in original.captures() for o in c.observations}
    for capture in replayed.captures():
        for obs in capture.observations:
            assert replayed.load_payload(obs).tobytes() == originals[obs.stream]
    assert verify_session(Store.open(store_root), replayed.session_id).ok
