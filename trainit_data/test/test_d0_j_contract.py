"""D0-J -- the contract proof (D0_DESIGN.md §21.2), with ZERO record-schema edits.

What D0-J proves, with vocabulary rows added as DATA only (the shipped vocabulary is unchanged, its golden hash is
pinned in ``test_canonical.py``):

1. a ``gazebo`` backend (SIM_TIME, ``sim_source: gazebo``) writes, seals and verifies;
2. a ``capture_tool`` backend with an ``imu`` modality whose low-rate reading is a typed JSON ``vector`` data block
   (the common observation header + the block, no payload file) writes, seals, verifies and reads back exactly;
3. a sync policy added as a vocabulary row only (``NEAREST_WITHIN_WINDOW`` v1) is recorded and verified;
4. both sessions go into ONE immutable dataset version that verifies;
5. a newer-minor record carrying an ``extensions`` entry is read by this same-major reader and preserved
   byte-for-byte (the observation's here; the capture / session / calibration ones in ``test_records.py`` and
   ``test_parsing_hardening.py``).

The shipped vocabulary still refuses every new row, and the payload-shaped rules stay enforced (a vector
observation never references a payload; an image stream never takes a vector reading).

Every value is invented and customer-neutral.
"""
import copy

import numpy as np
import pytest

from _helpers import FRAME, SENSOR, rgb, stamp
from trainit_data._canonical import canonical_json
from trainit_data.capture import Capture
from trainit_data.dataset import create_dataset
from trainit_data.dataset_integrity import verify_dataset
from trainit_data.errors import (DATA_BLOCK_UNSUPPORTED, INVALID_VALUE, SCHEMA_MAJOR_UNSUPPORTED, UNKNOWN_FIELD,
                                 UNKNOWN_VOCAB, WRITE_FAILED, DataError)
from trainit_data.integrity import verify_session
from trainit_data.observation import VectorBlock
from trainit_data.versions import create_version
from trainit_data.vocab import Vocabulary
from trainit_data.writer import ImageSample, VectorSample

TOOL = 'handheld_tool'
IMU_FRAME = 'handheld_tool_imu_frame'
IMU_UNITS = ['1', '1', '1', '1', 'rad/s', 'rad/s', 'rad/s', 'm/s^2', 'm/s^2', 'm/s^2']   # quaternion, gyro, accel


def extended_doc(base: Vocabulary) -> dict:
    """The rows D0-J adds -- data only."""
    doc = copy.deepcopy(base.to_dict())
    doc['backends'] += ['gazebo', 'capture_tool']
    doc['data_blocks']['vector'] = {'version': '1.0'}
    doc['modalities']['imu'] = {'data_block': 'vector', 'encodings': [], 'payloads': []}
    doc['sync_policies']['NEAREST_WITHIN_WINDOW'] = {'versions': [1]}
    return doc


@pytest.fixture
def extended(vocab):
    return Vocabulary.from_dict(extended_doc(vocab))


def _trigger(at, index):
    return {'trigger_id': 'manual', 'kind': 'MANUAL', 'firing_index': index, 'origin': 'OPERATOR',
            'issued': {'stamp': at.to_dict()}, 'reason': {'note': 'synthetic'}}


def _nearest(reference, status, skew, tolerance=None):
    out = {'policy': {'name': 'NEAREST_WITHIN_WINDOW', 'version': 1, 'params': {'window_ms': 5}},
           'reference': reference, 'status': status, 'max_skew_ns': skew}
    if tolerance is not None:
        out['tolerance_ns'] = tolerance
    return out


def imu_values(i):
    return [1.0, 0.0, 0.0, 0.0, 0.01 * i, -0.02, 0.0, 0.0, 0.0, 9.81]


def gazebo_session(store, extended, captures=2):
    writer = store.create_session(
        source={'transport': 'LIVE_ROS', 'backend': 'gazebo'},
        clock={'session_domain': 'SIM_TIME', 'sim_source': 'gazebo', 'use_sim_time': True},
        sensors=[{'sensor_id': SENSOR, 'streams': [{'stream': 'color', 'modality': 'rgb'}]}], vocabulary=extended)
    for i in range(captures):
        at = stamp(10 + i, 0, 'SIM_TIME', 'HEADER', sim_source='gazebo')
        writer.append_capture(trigger=_trigger(at, i), outcome='COMPLETE',
                              sync={'policy': {'name': 'NEAREST_WITHIN_WINDOW', 'version': 1,
                                               'params': {'window_ms': 5}},
                                    'reference': f'{SENSOR}/color', 'status': 'SINGLE_STREAM'},
                              samples=[ImageSample(SENSOR, 'color', at, rgb(i), 'rgb8', FRAME)])
    return writer.close()


def capture_tool_session(store, extended, captures=3):
    """A hand-held tool: a wrist camera and an IMU, ROS time; the IMU reading lags the image by 2 ms."""
    writer = store.create_session(
        source={'transport': 'LIVE_ROS', 'backend': 'capture_tool'},
        clock={'session_domain': 'ROS_TIME', 'use_sim_time': False},
        sensors=[{'sensor_id': TOOL, 'streams': [{'stream': 'color', 'modality': 'rgb'},
                                                 {'stream': 'imu', 'modality': 'imu'}]}], vocabulary=extended)
    for i in range(captures):
        at = stamp(200 + i)
        imu_at = stamp(200 + i, 2_000_000)
        writer.append_capture(
            trigger=_trigger(at, i), outcome='COMPLETE',
            sync=_nearest(f'{TOOL}/color', 'WITHIN_TOLERANCE', 2_000_000, 5_000_000),
            samples=[ImageSample(TOOL, 'color', at, rgb(10 + i), 'rgb8', FRAME),
                     VectorSample(TOOL, 'imu', imu_at, IMU_FRAME, imu_values(i), IMU_UNITS,
                                  extensions={'trainit.test.note': {'axis_order': 'wxyz'}})])
    return writer.close()


# ---- 1-3: rows only, end to end ------------------------------------------------------------------------------

def test_the_shipped_vocabulary_has_none_of_the_d0_j_rows(vocab):
    doc = vocab.to_dict()
    assert 'capture_tool' not in doc['backends'] and 'gazebo' not in doc['backends']
    assert 'imu' not in vocab.modalities and 'vector' not in vocab.data_blocks
    for category, value in (('backends', 'capture_tool'), ('modalities', 'imu'), ('data_blocks', 'vector')):
        with pytest.raises(DataError) as err:
            vocab.require(category, value, 'd0_j')
        assert err.value.code == UNKNOWN_VOCAB
    with pytest.raises(DataError) as err:
        vocab.require_sync_policy('NEAREST_WITHIN_WINDOW', 1, 'd0_j')
    assert err.value.code == UNKNOWN_VOCAB


def test_a_gazebo_backend_is_a_row_only(store, extended):
    header = gazebo_session(store, extended)
    report = verify_session(store, header.session_id)
    assert report.ok, report.to_dict()
    reader = store.open_session(header.session_id)
    capture = next(reader.captures())
    assert capture.observations[0].stamp.sim_source == 'gazebo'
    assert capture.sync.policy.name == 'NEAREST_WITHIN_WINDOW'
    assert reader.vocab.sha256 == extended.sha256


def test_a_capture_tool_imu_reading_is_a_vector_block_with_no_payload(store, extended):
    header = capture_tool_session(store, extended)
    report = verify_session(store, header.session_id)
    assert report.ok, report.to_dict()
    reader = store.open_session(header.session_id)
    captures = list(reader.captures())
    assert [c.outcome for c in captures] == ['COMPLETE'] * 3
    for i, capture in enumerate(captures):
        image, imu = capture.observations
        assert image.payload is not None and np.array_equal(reader.load_payload(image), rgb(10 + i))
        assert imu.payload is None and isinstance(imu.data, VectorBlock)
        assert list(imu.data.values) == imu_values(i) and list(imu.data.units) == IMU_UNITS
        assert imu.data.frame_id == IMU_FRAME and imu.stamp.to_ns() - image.stamp.to_ns() == 2_000_000
        assert capture.sync.status == 'WITHIN_TOLERANCE' and capture.sync.max_skew_ns == 2_000_000
        with pytest.raises(DataError) as err:
            reader.load_payload(imu)                      # its values are the record: there is no file to load
        assert err.value.code == INVALID_VALUE
    # only the images wrote files: one payload per capture
    payloads = sorted(p.name for p in (store.root / 'sessions' / header.session_id / 'payload').rglob('*')
                      if p.is_file())
    assert payloads == [f'{TOOL}__color.png'] * 3


def test_a_vector_only_capture_creates_no_payload_directory(store, extended):
    writer = store.create_session(
        source={'transport': 'LIVE_ROS', 'backend': 'capture_tool'},
        clock={'session_domain': 'ROS_TIME', 'use_sim_time': False},
        sensors=[{'sensor_id': TOOL, 'streams': [{'stream': 'imu', 'modality': 'imu'}]}], vocabulary=extended)
    at = stamp(300)
    writer.append_capture(trigger=_trigger(at, 0), outcome='COMPLETE',
                          sync={'policy': {'name': 'NEAREST_WITHIN_WINDOW', 'version': 1, 'params': {}},
                                'reference': f'{TOOL}/imu', 'status': 'SINGLE_STREAM'},
                          samples=[VectorSample(TOOL, 'imu', at, IMU_FRAME, imu_values(0), IMU_UNITS)])
    header = writer.close()
    assert verify_session(store, header.session_id).ok
    assert not any((store.root / 'sessions' / header.session_id / 'payload').glob('*'))


# ---- 4: one version over both sources ----------------------------------------------------------------------

def test_both_row_only_sources_go_into_one_immutable_version(store, extended):
    gazebo = gazebo_session(store, extended)
    tool = capture_tool_session(store, extended)
    create_dataset(store, 'd0_j', name='D0-J contract')
    version = create_version(store, 'd0_j', sessions=[gazebo.session_id, tool.session_id])
    assert version.content_kind == 'OBSERVATIONS'
    report = verify_dataset(store, 'd0_j')
    assert report.ok, report.to_dict()


# ---- 5: schema evolution inside the major ------------------------------------------------------------------

def test_an_observation_extension_is_preserved_byte_for_byte(store, extended):
    header = capture_tool_session(store, extended, captures=1)
    index = store.root / 'sessions' / header.session_id / 'captures.jsonl'
    line = index.read_bytes().splitlines()[0]
    capture = next(store.open_session(header.session_id).captures())
    assert capture.observations[1].extensions == {'trainit.test.note': {'axis_order': 'wxyz'}}
    assert canonical_json(capture.to_dict()) == line          # re-serialised: the same bytes


def test_a_newer_minor_vector_block_with_an_extension_is_read(store, extended):
    header = capture_tool_session(store, extended, captures=1)
    doc = next(store.open_session(header.session_id).captures()).to_dict()
    doc['schema'] = 'trainit-capture/1.9'
    doc['observations'][1]['extensions']['trainit.future.field'] = [1, 2]
    record = Capture.from_dict(copy.deepcopy(doc), extended)
    assert record.schema == 'trainit-capture/1.9'
    assert canonical_json(record.to_dict()) == canonical_json(doc)
    doc['observations'][1]['data']['bias'] = [0.0]            # outside extensions: strict parsing stands
    with pytest.raises(DataError) as err:
        Capture.from_dict(doc, extended)
    assert err.value.code == UNKNOWN_FIELD


# ---- the payload-shaped rules stay enforced ------------------------------------------------------------------

def _vector_obs(extended, **data_overrides):
    data = {'block': 'vector', 'version': '1.0', 'frame_id': IMU_FRAME, 'values': [0.0, 9.81], 'units': ['m/s^2'] * 2}
    data.update(data_overrides)
    return data


@pytest.mark.parametrize('overrides, code, message', [
    ({'units': ['m/s^2']}, INVALID_VALUE, 'one unit per value'),
    ({'values': []}, INVALID_VALUE, r'data\.values'),
    ({'values': [0.0] * 65, 'units': ['1'] * 65}, INVALID_VALUE, 'a series is a D1 Series record'),
    ({'values': [float('nan'), 0.0]}, INVALID_VALUE, r'data\.values\[0\]'),
    ({'values': [True, 0.0]}, INVALID_VALUE, r'data\.values\[0\]'),
    ({'values': ['1', 0.0]}, INVALID_VALUE, r'data\.values\[0\]'),
    ({'version': '2.0'}, SCHEMA_MAJOR_UNSUPPORTED, 'major'),
    ({'bias': [0.0]}, UNKNOWN_FIELD, 'bias'),
])
def test_a_malformed_vector_block_is_refused(extended, overrides, code, message):
    row, block_row = extended.modalities['imu'], extended.data_blocks['vector']
    with pytest.raises(DataError, match=message) as err:
        VectorBlock.from_dict(_vector_obs(extended, **overrides), row, block_row, 'data')
    assert err.value.code == code


def test_a_vector_observation_never_references_a_payload(store, extended):
    header = capture_tool_session(store, extended, captures=1)
    doc = next(store.open_session(header.session_id).captures()).to_dict()
    doc['observations'][1]['payload'] = copy.deepcopy(doc['observations'][0]['payload'])
    with pytest.raises(DataError, match='references no payload'):
        Capture.from_dict(doc, extended)


def test_a_sample_of_the_wrong_shape_is_refused_before_any_byte(store, extended):
    writer = store.create_session(
        source={'transport': 'LIVE_ROS', 'backend': 'capture_tool'},
        clock={'session_domain': 'ROS_TIME', 'use_sim_time': False},
        sensors=[{'sensor_id': TOOL, 'streams': [{'stream': 'color', 'modality': 'rgb'},
                                                 {'stream': 'imu', 'modality': 'imu'}]}], vocabulary=extended)
    at = stamp(400)
    single = {'policy': {'name': 'NEAREST_WITHIN_WINDOW', 'version': 1, 'params': {}}, 'status': 'SINGLE_STREAM'}
    with pytest.raises(DataError) as err:
        writer.append_capture(trigger=_trigger(at, 0), outcome='COMPLETE', sync=dict(single, reference=f'{TOOL}/color'),
                              samples=[VectorSample(TOOL, 'color', at, FRAME, [1.0], ['1'])])
    assert err.value.code == INVALID_VALUE
    with pytest.raises(DataError) as err:
        writer.append_capture(trigger=_trigger(at, 0), outcome='COMPLETE', sync=dict(single, reference=f'{TOOL}/imu'),
                              samples=[ImageSample(TOOL, 'imu', at, rgb(0), 'rgb8', FRAME)])
    assert err.value.code == INVALID_VALUE
    header = writer.close()
    assert header.counters['captures']['attempted'] == 0      # nothing was written
    assert verify_session(store, header.session_id).ok


def test_a_modality_row_lists_both_payload_fields_or_neither(vocab):
    doc = extended_doc(vocab)
    doc['modalities']['imu']['payloads'] = ['application/json']
    with pytest.raises(DataError, match='both listed') as err:
        Vocabulary.from_dict(doc)
    assert err.value.code == INVALID_VALUE


def _tool_writer(store, extended):
    return store.create_session(
        source={'transport': 'LIVE_ROS', 'backend': 'capture_tool'},
        clock={'session_domain': 'ROS_TIME', 'use_sim_time': False},
        sensors=[{'sensor_id': TOOL, 'streams': [{'stream': 'color', 'modality': 'rgb'},
                                                 {'stream': 'imu', 'modality': 'imu'}]}], vocabulary=extended)


def test_loose_values_are_refused_before_any_byte_never_coerced(store, extended):
    """The writer hands the values to the block reader as given: a bool or a string is never stored as 1.0."""
    writer = _tool_writer(store, extended)
    at = stamp(500)
    for values in ([True, 2.0], ['1', 2.0]):
        with pytest.raises(DataError) as err:
            writer.append_capture(trigger=_trigger(at, 0), outcome='COMPLETE',
                                  sync=_nearest(f'{TOOL}/imu', 'SINGLE_STREAM', None),
                                  samples=[VectorSample(TOOL, 'imu', at, IMU_FRAME, values, ['1', '1'])])
        assert err.value.code == INVALID_VALUE
    header = writer.close()
    assert header.counters['captures']['attempted'] == 0


def test_a_mixed_capture_whose_payload_directory_fails_keeps_the_reading(store, extended, monkeypatch):
    """ENOSPC on the capture's payload directory: the image becomes WRITE_FAILED, the vector reading (which needs
    no file) is committed, the capture is PARTIAL, one write error is counted, and the session verifies."""
    import pathlib
    writer = _tool_writer(store, extended)
    at, imu_at = stamp(600), stamp(600, 1_000)
    real_mkdir = pathlib.Path.mkdir

    def full(self, *args, **kwargs):
        if self.parent.name == 'payload':
            raise OSError(28, 'No space left on device')
        return real_mkdir(self, *args, **kwargs)
    monkeypatch.setattr(pathlib.Path, 'mkdir', full)
    with pytest.raises(DataError) as err:
        writer.append_capture(trigger=_trigger(at, 0), outcome='COMPLETE',
                              sync=_nearest(f'{TOOL}/color', 'WITHIN_TOLERANCE', 1_000, 5_000_000),
                              samples=[ImageSample(TOOL, 'color', at, rgb(1), 'rgb8', FRAME),
                                       VectorSample(TOOL, 'imu', imu_at, IMU_FRAME, imu_values(0), IMU_UNITS)])
    assert err.value.code == WRITE_FAILED and err.value.context['failed_streams'] == [f'{TOOL}/color']
    monkeypatch.undo()
    header = writer.close()
    assert header.counters['write_errors'] == 1                  # the reading cost no write
    report = verify_session(store, header.session_id)
    assert report.ok, report.to_dict()
    [capture] = store.open_session(header.session_id).captures()
    assert capture.outcome == 'PARTIAL' and [o.stream for o in capture.observations] == ['imu']
    assert [(m.stream, m.reason) for m in capture.missing] == [(f'{TOOL}/color', 'WRITE_FAILED')]


@pytest.mark.parametrize('row, code', [
    ({'data_block': 'image', 'encodings': [], 'payloads': []}, INVALID_VALUE),          # pixels need a payload
    ({'data_block': 'vector', 'encodings': ['rgb8'], 'payloads': ['image/png']}, INVALID_VALUE),
    ({'data_block': 'pointcloud', 'encodings': [], 'payloads': []}, DATA_BLOCK_UNSUPPORTED),     # no reader
], ids=['image_without_payloads', 'vector_with_payloads', 'block_without_reader'])
def test_a_stream_this_core_cannot_record_is_refused_when_declared(store, vocab, row, code):
    doc = extended_doc(vocab)
    doc['data_blocks'].setdefault(row['data_block'], {'version': '1.0'})
    doc['modalities']['odd'] = row
    odd = Vocabulary.from_dict(doc)
    with pytest.raises(DataError) as err:
        store.create_session(source={'transport': 'LIVE_ROS', 'backend': 'capture_tool'},
                             clock={'session_domain': 'ROS_TIME'}, vocabulary=odd,
                             sensors=[{'sensor_id': TOOL, 'streams': [{'stream': 'odd', 'modality': 'odd'}]}])
    assert err.value.code == code
    assert not list((store.root / 'sessions').glob('*'))       # refused before anything is written


def test_a_newer_minor_session_on_disk_verifies_versions_and_keeps_its_bytes(store, extended, monkeypatch):
    """A session written by a FUTURE same-major writer (trainit-capture/1.9, extensions on the capture and the
    observation) is verified, versioned and re-read by this reader, and none of its bytes change."""
    import trainit_data.writer as writer_module
    monkeypatch.setattr(writer_module, 'CAPTURE_SCHEMA', 'trainit-capture/1.9')
    writer = _tool_writer(store, extended)
    at = stamp(700)
    writer.append_capture(
        trigger=_trigger(at, 0), outcome='COMPLETE', sync=_nearest(f'{TOOL}/color', 'SYNCED', 0),
        extensions={'trainit.future.capture': {'k': 1}},
        samples=[ImageSample(TOOL, 'color', at, rgb(7), 'rgb8', FRAME),
                 VectorSample(TOOL, 'imu', at, IMU_FRAME, imu_values(1), IMU_UNITS,
                              extensions={'trainit.future.reading': [1, 2]})])
    header = writer.close()
    monkeypatch.undo()
    session_dir = store.root / 'sessions' / header.session_id
    before = {p.relative_to(session_dir): p.read_bytes() for p in session_dir.rglob('*') if p.is_file()}
    report = verify_session(store, header.session_id)
    assert report.ok, report.to_dict()
    [capture] = store.open_session(header.session_id).captures()
    assert capture.schema == 'trainit-capture/1.9' and capture.extensions == {'trainit.future.capture': {'k': 1}}
    assert canonical_json(capture.to_dict()) == (session_dir / 'captures.jsonl').read_bytes().rstrip(b'\n')
    create_dataset(store, 'd0_j_minor', name='D0-J newer minor')
    create_version(store, 'd0_j_minor', sessions=[header.session_id])
    assert verify_dataset(store, 'd0_j_minor').ok
    after = {p.relative_to(session_dir): p.read_bytes() for p in session_dir.rglob('*') if p.is_file()}
    assert after == before


def test_d0_j_needed_no_record_schema_edit():
    """The record schemas D0-J relies on are the ones D0.8 shipped: the rows, the block reader and the writer's
    VectorSample are the only additions (D0_DESIGN.md §21.2)."""
    from trainit_data import calibration, capture, dataset, exports, labels, session_record, splits, versions
    assert {m.SCHEMA for m in (capture, calibration, session_record, dataset, versions, labels, splits, exports)} == {
        'trainit-capture/1.0', 'trainit-calibration/1.0', 'trainit-session/1.1', 'trainit-dataset-identity/1.0',
        'trainit-dataset/1.0', 'trainit-labels/1.0', 'trainit-split/1.0', 'trainit-export/1.0'}
