"""The session's ``acquisition`` block (D0.4): source topology from evidence, never from topic names (no ROS)."""
import copy
import json

import pytest

from _ros_helpers import config, load_profile
from trainit_data._canonical import sha256_json
from trainit_data.errors import INVALID_VALUE, SOURCE_DESCRIPTOR_INVALID, DataError
from trainit_data.vocab import Vocabulary
from trainit_data_ros import acquisition
from trainit_data_ros.recording_config import resolve
from trainit_hardware_layer.camera import source_info as source_doc

VOCAB = Vocabulary.default()
HERE, ELSEWHERE = 'a' * 64, 'b' * 64
TRANSPORT = {'origin': 'OBSERVED', 'kind': 'ROS2_DDS', 'rmw': 'rmw_fastrtps_cpp', 'ros_domain_id': 87}
REAL = {'backend': 'real', 'intrinsics_kind': 'DEVICE_REPORTED', 'stamp_origin': 'DRIVER_GLOBAL_TIME',
        'registration_basis': 'driver_align_depth'}


def real_profile(model):
    profile = load_profile(model)
    profile['backends']['real'] = {'source_descriptor': dict(REAL)}
    return profile


def resolved(required=False, backend='real', **sensor):
    doc = config(source={'backend': backend})
    doc['sensors'][0]['source_info'] = dict({'required': required}, **sensor)
    return resolve(doc, VOCAB, load_profile=real_profile, use_sim_time=False)


def camera_doc(boot=HERE, backend='real', clock_domain='global_time', **extra):
    timestamping = {'origin': 'OBSERVED', 'clock_domains': {'color': clock_domain, 'depth': clock_domain}}
    origin = source_doc.stamp_origin(timestamping['clock_domains'])
    if origin:
        timestamping['stamp_origin'] = origin
    host = {'origin': 'DISCOVERED', 'hostname': 'edge-01', 'machine': 'aarch64', 'kernel': '5.15.0-tegra'}
    if boot:
        host['boot_id_sha256'] = boot
    blocks = dict(device={'origin': 'DISCOVERED', 'name': 'example_rgbd', 'serial_number': '000000000001'},
                  host=host, timestamping=timestamping)
    blocks.update(extra)
    return json.dumps(source_doc.build(backend=backend, published_wall_utc_ns=1, **blocks))


def recorder(boot=HERE, host_id=None):
    out = {'host': {'origin': 'OBSERVED', 'hostname': 'workstation', 'machine': 'x86_64', 'kernel': '6.8.0'}}
    if boot:
        out['host']['boot_id_sha256'] = boot
    if host_id:
        out['declared'] = {'origin': 'DECLARED', 'host_id': host_id}
    return out


def build(cfg, raw, rec=None, **kw):
    received = {} if raw is None else {'overhead_camera': raw}
    return acquisition.build(cfg, received, transport=TRANSPORT, recorder=rec or recorder(), **kw)


def test_the_same_boot_is_a_local_sensor_and_local_recorder():
    block = build(resolved(), camera_doc())
    entry = block['sensors']['overhead_camera']
    assert block['schema'] == 'trainit-acquisition/1.0' and block['transport'] == TRANSPORT
    assert entry['relation'] == 'LOCAL_SENSOR_LOCAL_RECORDER'
    assert entry['stamp_origin_verified'] is True
    assert entry['source_info_topic'] == '/overhead_camera/source_info'               # the contract name
    assert entry['source_info']['device']['serial_number'] == '000000000001'
    assert entry['source_info_sha256'] == sha256_json(entry['source_info'])


def test_another_boot_is_a_remote_source_and_nothing_else_changes():
    local = build(resolved(), camera_doc(boot=HERE))
    remote = build(resolved(), camera_doc(boot=ELSEWHERE))
    assert remote['sensors']['overhead_camera']['relation'] == 'REMOTE_SENSOR_SOURCE'
    for block in (local, remote):                                    # only the relation and its evidence differ
        for key in ('relation', 'relation_basis', 'source_info', 'source_info_sha256'):
            block['sensors']['overhead_camera'].pop(key)
    assert local == remote


@pytest.mark.parametrize('raw, rec', [
    (None, recorder()),                                 # no source document
    (camera_doc(boot=None), recorder()),                # the source could not read its boot id
    (camera_doc(), recorder(boot=None)),                # the recorder could not read its own
])
def test_without_evidence_the_relation_is_unknown(raw, rec):
    entry = build(resolved(), raw, rec)['sensors']['overhead_camera']
    assert entry['relation'] == 'UNKNOWN'


def test_the_relation_is_never_read_from_a_topic_name():
    cfg = resolved(topic='/remote_jetson/camera/source_info')
    assert build(cfg, camera_doc())['sensors']['overhead_camera']['relation'] == 'LOCAL_SENSOR_LOCAL_RECORDER'


def test_a_required_document_that_never_arrived_refuses_the_session():
    with pytest.raises(DataError) as err:
        build(resolved(required=True), None)
    assert err.value.code == SOURCE_DESCRIPTOR_INVALID and 'source_info' in err.value.message
    assert build(resolved(required=False), None)['sensors']['overhead_camera'] == {
        'source_info_topic': '/overhead_camera/source_info', 'relation': 'UNKNOWN',
        'relation_basis': 'no source document'}


@pytest.mark.parametrize('raw', ['not json', '[]', json.dumps({'schema': 'trainit-camera-source/1.0'}),
                                 json.dumps({'schema': 'something-else/1.0', 'backend': 'real',
                                             'published_wall_utc_ns': 1})])
def test_a_malformed_document_refuses_the_session(raw):
    with pytest.raises(DataError) as err:
        build(resolved(), raw)
    assert err.value.code == INVALID_VALUE


def test_a_document_of_another_backend_refuses_the_session():
    with pytest.raises(DataError) as err:
        build(resolved(), camera_doc(backend='isaac'))
    assert err.value.code == SOURCE_DESCRIPTOR_INVALID


def test_an_observed_stamp_origin_that_contradicts_the_descriptor_refuses_the_session():
    with pytest.raises(DataError) as err:                  # system_time headers are RECEIPT, not global time
        build(resolved(), camera_doc(clock_domain='system_time'))
    assert err.value.code == SOURCE_DESCRIPTOR_INVALID
    assert err.value.context == {'sensor_id': 'overhead_camera', 'observed': 'RECEIPT',
                                 'declared': 'DRIVER_GLOBAL_TIME'}


def test_an_unestablished_stamp_origin_is_recorded_as_unverified_never_as_verified():
    entry = build(resolved(), camera_doc(clock_domain='hardware_clock'))['sensors']['overhead_camera']
    assert entry['stamp_origin_verified'] is False
    assert 'stamp_origin' not in entry['source_info']['timestamping']


def test_the_recorder_host_is_observed_and_its_declared_id_kept_apart(tmp_path):
    proc = tmp_path / 'proc'
    (proc / 'sys' / 'kernel' / 'random').mkdir(parents=True)
    (proc / 'sys' / 'kernel' / 'random' / 'boot_id').write_text('00000000-0000-4000-8000-000000000000\n')
    rec = acquisition.recorder_host('bench-ws', proc_root=proc)
    assert rec['host']['origin'] == 'OBSERVED' and len(rec['host']['boot_id_sha256']) == 64
    assert rec['declared'] == {'origin': 'DECLARED', 'host_id': 'bench-ws'}
    assert 'declared' not in acquisition.recorder_host(None, proc_root=proc)


def test_a_replay_carries_its_originals_block_and_is_not_cross_checked_against_the_device():
    original_block = build(resolved(), camera_doc())
    cfg = copy.deepcopy(resolved())
    cfg['source'] = {'backend': 'replay', 'transport': 'LIVE_ROS', 'replay_of': 'ses-20261029T000000Z-00000000',
                     'descriptor': dict(REAL, stamp_origin='HEADER')}
    block = build(cfg, None, original={'session_id': 'ses-20261029T000000Z-00000000',
                                       'acquisition': original_block})
    assert block['original']['acquisition']['sensors']['overhead_camera']['source_info'] == \
        original_block['sensors']['overhead_camera']['source_info']             # the device provenance is kept
    assert block['sensors']['overhead_camera']['relation'] == 'UNKNOWN'           # the replay is no device


def _replay_cfg():
    cfg = copy.deepcopy(resolved())
    cfg['source'] = {'backend': 'replay', 'transport': 'LIVE_ROS', 'replay_of': 'ses-20261029T000000Z-00000000',
                     'descriptor': dict(REAL, stamp_origin='HEADER')}
    return cfg


def test_a_replay_receiving_a_live_cameras_document_is_refused():
    with pytest.raises(DataError) as err:          # a live camera publishes into the replay's namespace
        build(_replay_cfg(), camera_doc(), original={'session_id': 'ses-20261029T000000Z-00000000'})
    assert err.value.code == SOURCE_DESCRIPTOR_INVALID and 'replay' in err.value.message


def test_a_replay_of_a_replay_carries_one_level_of_original():
    first = build(_replay_cfg(), None, original={'session_id': 'ses-20261029T000000Z-00000000',
                                                 'acquisition': build(resolved(), camera_doc())})
    second = build(_replay_cfg(), None, original={'session_id': 'ses-20261029T000001Z-00000001',
                                                  'acquisition': first})
    assert 'original' in first and 'original' not in second['original']['acquisition']
    assert second['original']['session_id'] == 'ses-20261029T000001Z-00000001'


def test_the_block_is_json_plain_and_deterministic():
    block = build(resolved(), camera_doc())
    assert json.loads(json.dumps(block)) == block
    assert sha256_json(block) == sha256_json(build(resolved(), camera_doc()))


def test_the_source_info_default_and_override_are_resolved():
    cfg = resolve(config(), VOCAB, load_profile=load_profile, use_sim_time=False)
    assert cfg['sensors'][0]['source_info'] == {'topic': '/overhead_camera/source_info', 'required': False}
    assert 'deployment' not in cfg
    doc = config(deployment={'host_id': 'bench_ws'})
    doc['sensors'][0]['source_info'] = {'topic': '/elsewhere/source_info', 'required': True}
    cfg = resolve(doc, VOCAB, load_profile=load_profile, use_sim_time=False)
    assert cfg['sensors'][0]['source_info'] == {'topic': '/elsewhere/source_info', 'required': True}
    assert cfg['deployment'] == {'host_id': 'bench_ws'}


@pytest.mark.parametrize('mutate', [
    lambda d: d['sensors'][0].update(source_info={'topic': 'relative/source_info'}),     # cell-global names
    lambda d: d['sensors'][0].update(source_info={'required': 'yes'}),
    lambda d: d['sensors'][0].update(source_info={'serial': '1'}),
    lambda d: d.update(deployment={'host_ip': '10.0.0.2'}),                                # no addresses
])
def test_bad_source_info_and_deployment_settings_are_refused(mutate):
    doc = config()
    mutate(doc)
    with pytest.raises(DataError):
        resolve(doc, VOCAB, load_profile=load_profile, use_sim_time=False)
