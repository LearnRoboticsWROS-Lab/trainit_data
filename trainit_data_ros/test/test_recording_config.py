"""trainit-recording/1.0: strict parsing, explicit default resolution, the effective configuration."""
import copy

import pytest

from _ros_helpers import FRAME, config, load_profile
from trainit_data._canonical import canonical_json
from trainit_data.errors import (INVALID_VALUE, MISSING_FIELD, SCHEMA_MAJOR_UNSUPPORTED, SOURCE_DESCRIPTOR_INVALID,
                                 UNKNOWN_FIELD, UNKNOWN_VOCAB, DataError)
from trainit_data.vocab import Vocabulary
from trainit_data_ros.recording_config import resolve

VOCAB = Vocabulary.default()


def _resolve(doc, **kw):
    kw.setdefault('use_sim_time', False)
    return resolve(doc, VOCAB, load_profile=load_profile, **kw)


def test_every_default_is_resolved_and_recorded():
    cfg = _resolve(config())
    color, depth = cfg['sensors'][0]['streams']['color'], cfg['sensors'][0]['streams']['depth']
    assert color == {'topic': '/overhead_camera/color/image_raw', 'info': '/overhead_camera/color/camera_info',
                     'modality': 'rgb', 'expected_frame': FRAME, 'fps': 30.0}
    assert depth['topic'] == '/overhead_camera/depth/image_rect_raw' and depth['expected_frame'] == FRAME
    assert depth['registered_to'] == 'color'
    assert cfg['sync'] == {'policy': 'FIRST_AT_OR_AFTER_BOUNDARY', 'policy_version': 1,
                           'reference': 'overhead_camera/color', 'tolerance_ms': pytest.approx(1000 / 30 / 2),
                           'tolerance_source': 'half the frame period of the slowest stream (30 fps, hardware profile)',
                           'timeout_ms': 1000, 'clock_mismatch_ms': 1000}
    assert cfg['writer'] == {'queue_size': 8, 'max_pending': 4, 'buffer_frames': 8}
    assert cfg['clock'] == {'session_domain': 'ROS_TIME'}
    assert cfg['source'] == {'backend': 'mock', 'transport': 'LIVE_ROS', 'descriptor': {
        'intrinsics_kind': 'NOMINAL_PROFILE', 'stamp_origin': 'HEADER', 'registration_basis': 'same_pixel_grid',
        'supplied_by': 'trainit_hardware_layer profile example_rgbd backends.mock'}}
    assert cfg['rights'] == {'classification': 'INTERNAL'}
    assert canonical_json(cfg) == canonical_json(_resolve(config()))        # deterministic


def test_explicit_values_win_and_are_kept():
    doc = config(sync={'policy': 'FIRST_AT_OR_AFTER_BOUNDARY', 'policy_version': 1, 'tolerance_ms': 5.0,
                       'timeout_ms': 250, 'reference': 'overhead_camera/depth'},
                 writer={'queue_size': 2})
    doc['sensors'][0]['streams']['color'] = {'topic': '/elsewhere/rgb'}
    cfg = _resolve(doc)
    assert cfg['sync']['tolerance_ms'] == 5.0 and cfg['sync']['tolerance_source'] == 'configured'
    assert cfg['sync']['timeout_ms'] == 250 and cfg['sync']['reference'] == 'overhead_camera/depth'
    assert cfg['writer']['queue_size'] == 2 and cfg['sensors'][0]['streams']['color']['topic'] == '/elsewhere/rgb'


@pytest.mark.parametrize('mutate, code', [
    (lambda d: d.update(surprise=1), UNKNOWN_FIELD),
    (lambda d: d['sensors'][0].update(colour_space='srgb'), UNKNOWN_FIELD),
    (lambda d: d['sync'].update(policy_variant='x'), UNKNOWN_FIELD),
    (lambda d: d.update(schema='trainit-recording/2.0'), SCHEMA_MAJOR_UNSUPPORTED),
    (lambda d: d['source'].update(backend='holodeck'), UNKNOWN_VOCAB),
    (lambda d: d['triggers'].append({'id': 'x', 'kind': 'TELEPATHY'}), UNKNOWN_VOCAB),
    (lambda d: d['triggers'].append({'id': 'x', 'kind': 'EVENT'}), MISSING_FIELD),               # no channel / edge
    (lambda d: d['triggers'].append({'id': 'x', 'kind': 'APPLICATION_STATE'}), MISSING_FIELD),   # no task / transition
    (lambda d: d['triggers'].append({'id': 'x', 'kind': 'INTERVAL'}), INVALID_VALUE),          # no period
    (lambda d: d['triggers'].append({'id': 'manual', 'kind': 'INTERVAL', 'period_ms': 5}), INVALID_VALUE),
    (lambda d: d['triggers'].append({'id': 'manual_2', 'kind': 'MANUAL'}), INVALID_VALUE),
    (lambda d: d['sync'].update(policy='NEAREST_WITHIN_WINDOW'), UNKNOWN_VOCAB),
    (lambda d: d['sync'].update(reference='overhead_camera/thermal'), INVALID_VALUE),
    (lambda d: d['sensors'][0]['streams'].update(thermal={}), INVALID_VALUE),
    (lambda d: d['sensors'][0]['streams'].pop('color'), INVALID_VALUE),          # depth is registered to colour
    (lambda d: d['sensors'].append(copy.deepcopy(d['sensors'][0])), INVALID_VALUE),
    (lambda d: d['source'].update(replay_of='ses-20261028T000000Z-00000000'), INVALID_VALUE),
    (lambda d: d.update(clock={'sim_source': 'isaac'}), INVALID_VALUE),           # not on use_sim_time
])
def test_bad_configurations_are_refused(mutate, code):
    doc = config()
    mutate(doc)
    with pytest.raises(DataError) as err:
        _resolve(doc)
    assert err.value.code == code


def test_a_policy_in_the_vocabulary_but_not_implemented_is_refused():
    doc = Vocabulary.default().to_dict()
    doc['sync_policies']['NEAREST_WITHIN_WINDOW'] = {'versions': [1]}
    extended = Vocabulary.from_dict(doc)
    cfg = config(sync={'policy': 'NEAREST_WITHIN_WINDOW', 'policy_version': 1})
    with pytest.raises(DataError) as err:
        resolve(cfg, extended, load_profile=load_profile, use_sim_time=False)
    assert err.value.code == INVALID_VALUE and 'no implementation' in err.value.message


def test_sim_time_needs_the_simulator_named():
    with pytest.raises(DataError):
        _resolve(config(), use_sim_time=True)
    cfg = _resolve(config(clock={'sim_source': 'isaac'}), use_sim_time=True)
    assert cfg['clock'] == {'session_domain': 'SIM_TIME', 'sim_source': 'isaac'}


def test_the_source_descriptor_is_required_and_validated(monkeypatch):
    import _ros_helpers
    bad = copy.deepcopy(_ros_helpers.PROFILE)
    del bad['backends']['mock']['source_descriptor']
    with pytest.raises(DataError) as err:
        resolve(config(), VOCAB, load_profile=lambda m: copy.deepcopy(bad), use_sim_time=False)
    assert err.value.code == SOURCE_DESCRIPTOR_INVALID
    bad['backends']['mock']['source_descriptor'] = {'backend': 'mock', 'intrinsics_kind': 'GUESSED',
                                                    'stamp_origin': 'HEADER', 'registration_basis': 'x'}
    with pytest.raises(DataError) as err:
        resolve(config(), VOCAB, load_profile=lambda m: copy.deepcopy(bad), use_sim_time=False)
    assert err.value.code == SOURCE_DESCRIPTOR_INVALID


def test_a_replay_inherits_its_descriptor():
    doc = config(source={'backend': 'replay', 'replay_of': 'ses-20261028T000000Z-00000000'})
    with pytest.raises(DataError) as err:
        _resolve(doc)
    assert err.value.code == SOURCE_DESCRIPTOR_INVALID
    inherited = {'intrinsics_kind': 'NOMINAL_PROFILE', 'stamp_origin': 'HEADER',
                 'registration_basis': 'same_pixel_grid', 'supplied_by': 'replay'}
    cfg = _resolve(doc, replay_descriptor=inherited)
    assert cfg['source']['replay_of'] == 'ses-20261028T000000Z-00000000'
    assert cfg['source']['descriptor'] == inherited


@pytest.mark.parametrize('mutate', [
    lambda d: d['sensors'][0].update(namespace='camera'),                            # relative: lands in the slot
    lambda d: d['sensors'][0]['streams']['color'].update(topic='color/image_raw'),
    lambda d: d['sensors'][0]['streams']['depth'].update(expected_frame='overhead_camera_depth_optical_frame'),
    lambda d: d.update(writer={'max_pending': 999}),
])
def test_cell_global_names_registration_frames_and_limits(mutate):
    doc = config()
    mutate(doc)
    with pytest.raises(DataError) as err:
        _resolve(doc)
    assert err.value.code == INVALID_VALUE


def test_the_real_backend_reads_the_profiles_real_entry():
    import _ros_helpers
    profile = copy.deepcopy(_ros_helpers.PROFILE)
    profile['backends']['real'] = {'source_descriptor': {'backend': 'real', 'intrinsics_kind': 'DEVICE_REPORTED',
                                                         'stamp_origin': 'DRIVER_GLOBAL_TIME',
                                                         'registration_basis': 'driver_align_depth'}}
    cfg = resolve(config(source={'backend': 'real'}), VOCAB, load_profile=lambda m: copy.deepcopy(profile),
                  use_sim_time=False)
    assert cfg['source']['descriptor']['intrinsics_kind'] == 'DEVICE_REPORTED'


def test_a_replay_inherits_the_original_extrinsic_provenance():
    doc = config(source={'backend': 'replay', 'replay_of': 'ses-20261028T000000Z-00000000'})
    descriptor = {'intrinsics_kind': 'SYNTHETIC_RENDER', 'stamp_origin': 'HEADER',
                  'registration_basis': 'same_render_product', 'supplied_by': 'replay'}
    cfg = _resolve(doc, replay_descriptor=descriptor,
                   replay_extrinsic_provenance={'overhead_camera': 'SYNTHETIC_SCENE'})
    assert cfg['sensors'][0]['extrinsic_provenance'] == 'SYNTHETIC_SCENE'


def _isaac_profile(sim_source='isaac'):
    import _ros_helpers
    profile = copy.deepcopy(_ros_helpers.PROFILE)
    profile['backends']['isaac'] = {'source_descriptor': {
        'backend': 'isaac', 'intrinsics_kind': 'SYNTHETIC_RENDER', 'stamp_origin': 'HEADER',
        'registration_basis': 'same_render_product', 'sim_source': sim_source}}
    return lambda m: copy.deepcopy(profile)


def test_the_simulator_comes_from_the_backend_descriptor():
    cfg = resolve(config(source={'backend': 'isaac'}), VOCAB, load_profile=_isaac_profile(), use_sim_time=True)
    assert cfg['clock'] == {'session_domain': 'SIM_TIME', 'sim_source': 'isaac'}
    assert 'sim_source' not in cfg['source']['descriptor'] and '_sim_source' not in cfg['source']['descriptor']
    with pytest.raises(DataError) as err:          # a configured simulator must agree with the descriptor
        resolve(config(source={'backend': 'isaac'}, clock={'sim_source': 'gazebo'}), VOCAB,
                load_profile=_isaac_profile(), use_sim_time=True)
    assert err.value.code == INVALID_VALUE


def test_description_and_scene_are_part_of_the_effective_configuration():
    provenance = {'description': {'package': 'example_description', 'file': 'urdf/cell.urdf.xacro',
                                  'args': {'use_sim': 'true'}},
                  'scene': {'package': 'example_scenes', 'file': 'cell.usd', 'prim': '/World/Camera'}}
    cfg = _resolve(config(provenance=provenance))
    assert cfg['provenance'] == provenance
    for bad in ({'description': {'package': 'x', 'file': '/abs/path'}}, {'description_files': []},
                {'description': {'package': 'x', 'file': 'a.xacro', 'args': {'n': 1}}}, {'other': 1}):
        with pytest.raises(DataError):
            _resolve(config(provenance=bad))


def test_a_replay_declares_no_provenance_of_its_own():
    doc = config(source={'backend': 'replay', 'replay_of': 'ses-20261028T000000Z-00000000'},
                 provenance={'description': {'package': 'example_description', 'file': 'urdf/cell.urdf.xacro'}})
    inherited = {'intrinsics_kind': 'NOMINAL_PROFILE', 'stamp_origin': 'HEADER',
                 'registration_basis': 'same_pixel_grid', 'supplied_by': 'replay'}
    with pytest.raises(DataError) as err:
        _resolve(doc, replay_descriptor=inherited)
    assert err.value.code == INVALID_VALUE and 'inherits' in err.value.message


def test_descriptor_extras_never_reach_the_session_descriptor():
    profile = _isaac_profile()('example_rgbd')
    profile['backends']['isaac']['source_descriptor']['intrinsics_notes'] = 'pixel-edge principal point'
    cfg = resolve(config(source={'backend': 'isaac'}), VOCAB, load_profile=lambda m: copy.deepcopy(profile),
                  use_sim_time=True)
    assert set(cfg['source']['descriptor']) == {'intrinsics_kind', 'stamp_origin', 'registration_basis', 'supplied_by'}
    assert cfg['source']['intrinsics_notes'] == 'pixel-edge principal point'
    assert cfg['clock']['sim_source'] == 'isaac'


def test_the_aligned_sync_policy_is_selectable_and_the_default_stays_v1():
    """Owner decision 2026-11-05: a second, selectable policy; an authoring tool's default is declared, not implied."""
    from trainit_data_ros.recording_config import DEFAULT_POLICY, IMPLEMENTED_POLICIES
    from trainit_data_ros.recording_vocabulary import vocabulary
    cfg = resolve(config(sync={'policy': 'FIRST_ALIGNED_AT_OR_AFTER_BOUNDARY', 'policy_version': 1}), VOCAB,
                  load_profile=load_profile, use_sim_time=False)
    assert (cfg['sync']['policy'], cfg['sync']['policy_version']) == ('FIRST_ALIGNED_AT_OR_AFTER_BOUNDARY', 1)
    assert DEFAULT_POLICY == ('FIRST_AT_OR_AFTER_BOUNDARY', 1) and DEFAULT_POLICY in IMPLEMENTED_POLICIES
    assert vocabulary()['sync']['defaults']['policy'] == {'name': 'FIRST_AT_OR_AFTER_BOUNDARY', 'version': 1}
