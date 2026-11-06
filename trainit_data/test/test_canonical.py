"""The one canonical serialization / hashing mechanism (acceptance 2: deterministic across Python 3.10 and 3.11).

The literal hashes below were computed once and are asserted by BOTH interpreters (the ROS Python 3.10 and
Isaac's Python 3.11): the same test file passing in each is the cross-version proof.
"""
import pytest

from _helpers import calibration, stamp, sync, trigger
from trainit_data._canonical import canonical_json, parse_json, sha256_json
from trainit_data.capture import Capture
from trainit_data.errors import INVALID_VALUE, MALFORMED_RECORD, DataError

DOC = {'z': [1, 2.5, -0.0, 1e-7, 123456789012345678901234567890], 'a': {'ü': 'ñ', 'b': True, 'n': None},
       'f': 0.1 + 0.2}

GOLDEN_DOC = 'sha256:c7cf5503f1aafb1498d9e383a0b88f3ee25829bdd362fa697cf07e305b353ad3'
# The SHIPPED vocabulary's content hash. D0.7 added rows (modality instance_segmentation, data block instance_image,
# trigger kind SYNTHETIC_FRAME, origin GENERATOR, context trainit.randomization, sync policy SAME_RENDER_TICK): a
# content change, not a canonical-form change (D0.1-D0.6: sha256:44bd4c404e8c511946a84c75b0d5c0ec94c07b861c83b65b9fe2d4
# 9829046955). D0.8 (owner decision 2026-11-05) added ONE row: sync policy FIRST_ALIGNED_AT_OR_AFTER_BOUNDARY v1, again a
# content change (D0.7: sha256:40bab2004e490d4853b8fde9842ff6717c6efb73c63a42d46b5bf64e16b2f6cd). Sessions keep the
# vocabulary they pinned.
GOLDEN_VOCAB = 'sha256:8e8429b44e3ae437f9c9b5fd04f0f31b67a1aa46054077a1f49833eaf2cdd576'
GOLDEN_CALIBRATION = 'sha256:e14d4b471e5eccbf01abe1059d1930fceadbcc7c9216053458f4c42c1b8dd2da'
GOLDEN_CAPTURE = 'sha256:b6a25ba28feebc0a40ca773f3a0de969f97a62a82b982a2e838b27dd38615adc'


def golden_capture_dict():
    at = stamp(1794321012, 500000000)
    return {'schema': 'trainit-capture/1.0', 'capture_id': 'ses-20261110T143000Z-9e8d7c6b/000042', 'seq': 42,
            'trigger': trigger(at, 42), 'outcome': 'PARTIAL', 'sync': sync('SINGLE_STREAM', skew=None),
            'observations': [{
                'observation_id': 'ses-20261110T143000Z-9e8d7c6b/000042/overhead_camera/color',
                'sensor_id': 'overhead_camera', 'stream': 'color', 'modality': 'rgb', 'stamp': at.to_dict(),
                'receipt': {'host_monotonic_ns': 912345678901, 'wall_utc_ns': 1794321012507000000},
                'payload': {'path': 'sessions/ses-20261110T143000Z-9e8d7c6b/payload/000042/overhead_camera__color.png',
                            'media_type': 'image/png', 'sha256': 'sha256:' + '4b' * 32, 'bytes': 412377},
                'data': {'block': 'image', 'version': '1.0', 'width': 848, 'height': 480, 'encoding': 'rgb8',
                         'frame_id': 'overhead_camera_optical_frame', 'transforms': []}}],
            'missing': [{'stream': 'overhead_camera/depth', 'reason': 'TIMEOUT', 'detail': 'no sample in 1000 ms'}],
            'extensions': {'trainit.test.note': 'ü', 'trainit.test.values': [1.5, 2]}}


def test_canonical_form_is_sorted_compact_utf8():
    assert canonical_json(DOC) == ('{"a":{"b":true,"n":null,"ü":"ñ"},"f":0.30000000000000004,'
                                   '"z":[1,2.5,-0.0,1e-07,123456789012345678901234567890]}').encode('utf-8')


def test_literal_hashes_are_identical_in_every_interpreter(vocab):
    assert sha256_json(DOC) == GOLDEN_DOC
    assert vocab.sha256 == GOLDEN_VOCAB
    assert calibration(vocab).sha256 == GOLDEN_CALIBRATION
    assert sha256_json(Capture.from_dict(golden_capture_dict(), vocab).to_dict()) == GOLDEN_CAPTURE


def test_key_order_and_whitespace_never_change_a_hash():
    shuffled = {'f': DOC['f'], 'a': {'n': None, 'b': True, 'ü': 'ñ'}, 'z': DOC['z']}
    assert sha256_json(shuffled) == GOLDEN_DOC
    pretty = b'{\n  "f": 0.30000000000000004,\n  "z": [1, 2.5, -0.0, 1e-07, 123456789012345678901234567890],\n' \
             b'  "a": {"n": null, "b": true, "\xc3\xbc": "\xc3\xb1"}\n}'
    assert sha256_json(parse_json(pretty, 'pretty')) == GOLDEN_DOC


@pytest.mark.parametrize('bad', [float('nan'), float('inf'), object()])
def test_non_json_values_are_refused(bad):
    with pytest.raises(DataError) as err:
        canonical_json({'x': bad})
    assert err.value.code == INVALID_VALUE


def test_nan_literals_are_refused_on_parse():
    with pytest.raises(DataError) as err:
        parse_json(b'{"x": NaN}', 'x.json')
    assert err.value.code == MALFORMED_RECORD
