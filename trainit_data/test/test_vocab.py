"""The extensible vocabulary (acceptance 6, 7).

Acceptance 7: a backend, a sync policy and a whole image modality (with its encoding) are added as
vocabulary ROWS ONLY -- no code in this repository changes -- and a session using them is written, sealed
and verified. The pinned vocabulary is authoritative: the same session read with the shipped vocabulary
does not even resolve (it pins another hash).
"""
import copy

import numpy as np
import pytest

from _helpers import FRAME, SENSOR, stamp, trigger
from trainit_data.errors import INVALID_VALUE, UNKNOWN_FIELD, UNKNOWN_VOCAB, DataError
from trainit_data.integrity import verify_session
from trainit_data.vocab import Vocabulary
from trainit_data.writer import ImageSample


def extended_rows(base: Vocabulary) -> dict:
    doc = copy.deepcopy(base.to_dict())
    doc['backends'].append('gazebo')
    doc['sync_policies']['NEAREST_WITHIN_WINDOW'] = {'versions': [1]}
    doc['encodings']['mono16'] = {'dtype': 'uint16', 'channels': 1}
    doc['modalities']['infrared'] = {'data_block': 'image', 'encodings': ['mono16'],
                                     'payloads': ['image/png;bits=16']}
    return doc


def test_the_shipped_vocabulary_loads_and_is_stable(vocab):
    assert vocab.schema == 'trainit-vocab/1.0'
    assert 'FIRST_AT_OR_AFTER_BOUNDARY' in vocab.sync_policies
    assert set(vocab.modalities) == {'rgb', 'depth', 'instance_segmentation'}
    assert vocab.modalities['depth']['unit_scale_required']
    assert vocab.modalities['instance_segmentation']['data_block'] == 'instance_image'          # D0.7
    assert Vocabulary.from_dict(vocab.to_dict()).sha256 == vocab.sha256


def test_require_refuses_with_a_stable_code(vocab):
    assert vocab.require('backends', 'isaac') == 'isaac'
    with pytest.raises(DataError) as err:
        vocab.require('backends', 'gazebo', 'source.backend')
    assert err.value.code == UNKNOWN_VOCAB
    assert err.value.context == {'where': 'source.backend', 'category': 'backends', 'value': 'gazebo',
                                 'vocab_sha256': vocab.sha256}


def test_a_vocabulary_document_is_parsed_strictly(vocab):
    doc = vocab.to_dict()
    doc['colours'] = ['red']
    with pytest.raises(DataError) as err:
        Vocabulary.from_dict(doc)
    assert err.value.code == UNKNOWN_FIELD
    doc = vocab.to_dict()
    doc['modalities']['rgb']['encodings'] = ['rgb10']             # not an encodings row
    with pytest.raises(DataError) as err:
        Vocabulary.from_dict(doc)
    assert err.value.code == INVALID_VALUE
    doc = vocab.to_dict()
    doc['encodings']['rgb8']['dtype'] = 'object'
    with pytest.raises(DataError):
        Vocabulary.from_dict(doc)


def test_rows_added_as_data_only_work_end_to_end(store, vocab):
    extended = Vocabulary.from_dict(extended_rows(vocab))
    assert extended.sha256 != vocab.sha256
    sensors = [{'sensor_id': SENSOR, 'streams': [{'stream': 'ir', 'modality': 'infrared'},
                                                 {'stream': 'color', 'modality': 'rgb'}]}]
    at = stamp(5, 0, 'SIM_TIME', 'HEADER', sim_source='gazebo')
    writer = store.create_session(source={'transport': 'LIVE_ROS', 'backend': 'gazebo'},
                                  clock={'session_domain': 'SIM_TIME', 'sim_source': 'gazebo'}, sensors=sensors,
                                  vocabulary=extended)
    ir = np.arange(48, dtype=np.uint16).reshape(6, 8) * 1000
    capture = writer.append_capture(
        trigger=trigger(at), outcome='COMPLETE',
        sync={'policy': {'name': 'NEAREST_WITHIN_WINDOW', 'version': 1, 'params': {'window_ms': 5}},
              'reference': f'{SENSOR}/ir', 'status': 'SYNCED', 'max_skew_ns': 0},
        samples=[ImageSample(SENSOR, 'ir', at, ir, 'mono16', FRAME),
                 ImageSample(SENSOR, 'color', at, np.zeros((6, 8, 3), np.uint8), 'rgb8', FRAME)])
    header = writer.close()
    assert capture.sync.policy.name == 'NEAREST_WITHIN_WINDOW'
    assert header.vocab['sha256'] == extended.sha256
    report = verify_session(store, header.session_id)
    assert report.ok, report.to_dict()
    reader = store.open_session(header.session_id)
    assert reader.vocab.sha256 == extended.sha256          # the pinned vocabulary was used, not the shipped one
    observation = next(reader.captures()).observations[0]
    assert np.array_equal(reader.load_payload(observation), ir)


def test_the_shipped_vocabulary_still_refuses_the_new_rows(vocab):
    with pytest.raises(DataError) as err:
        vocab.require_sync_policy('NEAREST_WITHIN_WINDOW', 1, 'sync.policy')
    assert err.value.code == UNKNOWN_VOCAB
