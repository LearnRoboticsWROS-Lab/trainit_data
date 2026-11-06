"""ROS message -> canonical facts: exact pixels, verbatim intrinsics, the calibration record content."""
import numpy as np
import pytest

from _ros_helpers import FRAME, depth_mm, image_msg, info_msg, rgb
from trainit_data.calibration import CalibrationRecord
from trainit_data.vocab import Vocabulary
from trainit_data_ros import conversions


@pytest.mark.parametrize('array, encoding', [
    (rgb(1), 'rgb8'), (depth_mm(2), '16UC1'),
    (np.array([[np.nan, np.inf, -np.inf, 0.5]] * 3, dtype=np.float32), '32FC1')])
@pytest.mark.parametrize('pad, big', [(0, False), (3, False), (0, True)])
def test_images_convert_bit_exactly(array, encoding, pad, big):
    out = conversions.image_to_array(image_msg(array, encoding, step_pad=pad, bigendian=big))
    assert out.shape == array.shape and out.dtype.kind == array.dtype.kind
    assert out.astype(out.dtype.newbyteorder('<')).tobytes() == array.astype(array.dtype.newbyteorder('<')).tobytes()


@pytest.mark.parametrize('msg', [
    image_msg(rgb(), 'bgr8'),                                          # no silent channel reorder
    image_msg(rgb(), 'rgb8'),
])
def test_unknown_or_short_images_are_refused(msg):
    if msg.encoding == 'rgb8':
        msg.data = msg.data[:-1]
    with pytest.raises(conversions.ImageRefused):
        conversions.image_to_array(msg)


def test_camera_info_is_carried_verbatim():
    block = conversions.intrinsics(info_msg(fx=4.25))
    assert block['K'][0] == 4.25 and block['width'] == 8 and block['roi']['do_rectify'] is False


def test_the_calibration_content_is_a_valid_canonical_record():
    vocab = Vocabulary.default()
    common = dict(sensor_id='overhead_camera', optical_frame=FRAME, info_block=conversions.intrinsics(info_msg()),
                  intrinsics_kind='NOMINAL_PROFILE')
    color = CalibrationRecord.build(conversions.calibration_content(stream='color', encoding='rgb8', **common), vocab)
    assert color.depth is None and color.extrinsics == ()
    depth = CalibrationRecord.build(conversions.calibration_content(
        stream='depth', encoding='16UC1', registered_to='color', registration_basis='same_pixel_grid',
        extrinsic=('base_link', (0.5, 0.0, 0.5), (0.0, 0.0, 0.0, 1.0), 'NOMINAL_DESCRIPTION'), **common), vocab)
    assert depth.depth.invalid == 'zero' and depth.depth.registered_to == 'color'
    assert depth.extrinsics[0].parent_frame == 'base_link' and depth.extrinsics[0].child_frame == FRAME
    float_depth = conversions.calibration_content(stream='depth', encoding='32FC1', registered_to='color',
                                                  registration_basis='same_pixel_grid', **common)
    assert float_depth['depth']['invalid'] == 'nan_or_inf'


def test_the_description_hash_is_recorded_with_the_extrinsic():
    vocab = Vocabulary.default()
    content = conversions.calibration_content(
        sensor_id='overhead_camera', stream='color', optical_frame=FRAME, info_block=conversions.intrinsics(info_msg()),
        intrinsics_kind='SYNTHETIC_RENDER', encoding='rgb8',
        extrinsic=('base_link', (0.5, 0.0, 0.5), (0.0, 0.0, 0.0, 1.0), 'NOMINAL_DESCRIPTION'),
        description_sha256='sha256:' + 'a' * 64)
    record = CalibrationRecord.build(content, vocab)
    assert record.extrinsics[0].provenance.ref == {'description_sha256': 'sha256:' + 'a' * 64}
    assert record.transform_tree['description_sha256'] == 'sha256:' + 'a' * 64


def test_the_topic_name_is_not_part_of_the_calibration_identity():
    """The owner rule: the same calibration content on two topic names is ONE record (the topic is in the
    session's recording configuration)."""
    content = conversions.calibration_content(
        sensor_id='overhead_camera', stream='depth', optical_frame=FRAME, info_block=conversions.intrinsics(info_msg()),
        intrinsics_kind='SYNTHETIC_RENDER', encoding='32FC1', registered_to='color', registration_basis='x')
    assert content['intrinsics_provenance']['ref'] == {'source': 'sensor_msgs/CameraInfo'}


def test_a_measured_extrinsic_does_not_claim_the_description():
    vocab = Vocabulary.default()
    content = conversions.calibration_content(
        sensor_id='overhead_camera', stream='color', optical_frame=FRAME, info_block=conversions.intrinsics(info_msg()),
        intrinsics_kind='DEVICE_REPORTED', encoding='rgb8', intrinsics_notes='a note',
        extrinsic=('base_link', (0.5, 0.0, 0.5), (0.0, 0.0, 0.0, 1.0), 'MEASURED_HAND_EYE'),
        description_sha256='sha256:' + 'a' * 64)
    record = CalibrationRecord.build(content, vocab)
    assert record.extrinsics[0].provenance.ref == {}
    assert record.intrinsics_provenance.notes == 'a note'
