"""Observation: ONE sample of one stream (PUBLIC).

An observation is a common HEADER (identity, sensor / stream / modality, timestamps, receipt, payload
reference, quality, extensions) + a modality-typed DATA BLOCK chosen by the modality's vocabulary row
(D0_DESIGN.md §5, D0_SCHEMA_PROPOSAL.md §3). D0.1 implements the ``image`` block (rgb, depth); D0.7 adds
``instance_image`` (an instance-id render of a synthetic camera, :class:`InstanceImageBlock`); D0.9 adds
``vector`` (one low-rate non-image reading held in the record, :class:`VectorBlock`, test D0-J).

The extension seam: a new block kind (as D0-J's ``vector`` for a low-rate IMU sample was) is a vocabulary
row + one reader registered in ``_BLOCK_READERS``; the header and every other record stay unchanged. Until
a reader exists, a vocabulary block without one is refused with ``DATA_BLOCK_UNSUPPORTED``. High-rate
series are NOT observations: they are D1's Series record.

Records are validated by ``from_dict``: code that builds an observation in Python serialises it with
``to_dict`` and parses it back, so there is one validation path.
"""
import copy
from dataclasses import dataclass, field

import numpy as np

from . import _strict as s
from .errors import (DATA_BLOCK_UNSUPPORTED, INVALID_VALUE, MISSING_FIELD, SCHEMA_MAJOR_UNSUPPORTED,
                     DataError)
from .timestamps import Timestamp

IMAGE_BLOCK_MAJOR = 1


@dataclass(frozen=True)
class PayloadRef:
    path: str                  # store-relative
    media_type: str
    sha256: str
    bytes: int

    @classmethod
    def from_dict(cls, value, where='payload'):
        f = s.Fields(value, where)
        ref = cls(path=f.req('path', s.relative_path), media_type=f.req('media_type', s.token),
                  sha256=f.req('sha256', s.sha256), bytes=f.req('bytes', s.integer(0)))
        f.done()
        return ref

    def to_dict(self):
        return {'path': self.path, 'media_type': self.media_type, 'sha256': self.sha256, 'bytes': self.bytes}


@dataclass(frozen=True)
class Receipt:
    host_monotonic_ns: int = None
    wall_utc_ns: int = None

    @classmethod
    def from_dict(cls, value, where='receipt'):
        f = s.Fields(value, where)
        receipt = cls(host_monotonic_ns=f.opt('host_monotonic_ns', s.integer(0)),
                      wall_utc_ns=f.opt('wall_utc_ns', s.integer(0)))
        f.done()
        if receipt.host_monotonic_ns is None and receipt.wall_utc_ns is None:
            s.fail(INVALID_VALUE, 'a receipt holds at least one stamp (omit the block otherwise)', where)
        return receipt

    def to_dict(self):
        out = {}
        s.put(out, 'host_monotonic_ns', self.host_monotonic_ns)
        s.put(out, 'wall_utc_ns', self.wall_utc_ns)
        return out


@dataclass(frozen=True)
class Quality:
    """Optional, typed, never fabricated (D0_DESIGN.md §10): a backend that cannot tell leaves it absent."""
    invalid_depth_fraction: float = None
    frame_counter: int = None
    gap_before: int = None
    exposure_us: float = None
    gain: float = None

    @classmethod
    def from_dict(cls, value, where='quality'):
        f = s.Fields(value, where)
        quality = cls(invalid_depth_fraction=f.opt('invalid_depth_fraction', s.real),
                      frame_counter=f.opt('frame_counter', s.integer(0)),
                      gap_before=f.opt('gap_before', s.integer(0)),
                      exposure_us=f.opt('exposure_us', s.real),
                      gain=f.opt('gain', s.real))
        f.done()
        if quality.invalid_depth_fraction is not None and not 0.0 <= quality.invalid_depth_fraction <= 1.0:
            s.fail(INVALID_VALUE, 'must be within [0, 1]', f'{where}.invalid_depth_fraction')
        if quality.exposure_us is not None and quality.exposure_us < 0.0:
            s.fail(INVALID_VALUE, 'must be >= 0', f'{where}.exposure_us')
        return quality

    def to_dict(self):
        out = {}
        for name in ('invalid_depth_fraction', 'frame_counter', 'gap_before', 'exposure_us', 'gain'):
            s.put(out, name, getattr(self, name))
        return out


@dataclass(frozen=True)
class Transform:
    """One processing step applied before storage (channel reorder, crop, ...). The MVP stores raw data."""
    op: str
    params: dict = field(default_factory=dict)

    @classmethod
    def from_dict(cls, value, where='transform'):
        f = s.Fields(value, where)
        transform = cls(op=f.req('op', s.token), params=f.opt('params', s.opaque, {}))
        f.done()
        return transform

    def to_dict(self):
        out = {'op': self.op}
        if self.params:
            out['params'] = copy.deepcopy(self.params)
        return out


@dataclass(frozen=True)
class PoseAtStamp:
    """The sensor pose at the observation stamp, for a moving camera (a TF looked up at ``stamp``)."""
    parent_frame: str
    translation: tuple
    rotation_xyzw: tuple

    @classmethod
    def from_dict(cls, value, where='pose_at_stamp'):
        f = s.Fields(value, where)
        pose = cls(parent_frame=f.req('parent_frame', s.token), translation=f.req('translation', s.vector(3)),
                   rotation_xyzw=f.req('rotation_xyzw', s.quaternion))
        f.done()
        return pose

    def to_dict(self):
        return {'parent_frame': self.parent_frame, 'translation': list(self.translation),
                'rotation_xyzw': list(self.rotation_xyzw)}


@dataclass(frozen=True)
class ImageBlock:
    """The ``image`` data block (rgb, depth): what the payload's pixels mean."""
    version: str
    width: int
    height: int
    encoding: str
    frame_id: str
    unit_scale_m: float = None          # metres per stored unit (depth): 0.001 for 16UC1 mm, 1.0 for 32FC1 m
    calibration_id: str = None
    transforms: tuple = ()
    pose_at_stamp: PoseAtStamp = None
    block = 'image'
    requires_payload = True             # the reader contract: the pixels are a payload file

    @classmethod
    def from_dict(cls, value, modality_row, block_row, where='data'):
        f = s.Fields(value, where)
        f.req('block', s.token)
        image = cls(version=f.req('version', _block_version(IMAGE_BLOCK_MAJOR, block_row)),
                    width=f.req('width', s.integer(1)),
                    height=f.req('height', s.integer(1)),
                    encoding=f.req('encoding', s.token),
                    frame_id=f.req('frame_id', s.token),
                    unit_scale_m=f.opt('unit_scale_m', s.positive_real),
                    calibration_id=f.opt('calibration_id', s.calibration_id),
                    transforms=f.opt('transforms', s.list_of(Transform.from_dict), ()),
                    pose_at_stamp=f.opt('pose_at_stamp', PoseAtStamp.from_dict))
        f.done()
        if image.encoding not in modality_row['encodings']:
            s.fail(INVALID_VALUE, f'encoding {image.encoding!r} is not allowed for this modality '
                   f'(vocabulary: {list(modality_row["encodings"])})', f'{where}.encoding')
        if modality_row['unit_scale_required'] and image.unit_scale_m is None:
            s.fail(INVALID_VALUE, 'this modality declares its unit: unit_scale_m is required', where)
        if not modality_row['unit_scale_required'] and image.unit_scale_m is not None:
            s.fail(INVALID_VALUE, 'unit_scale_m does not apply to this modality', where)
        return image

    @staticmethod
    def check_payload(payload, modality_row, where):
        """An image always references its payload, in a media type of its modality's vocabulary row."""
        if payload is None:
            s.fail(MISSING_FIELD, 'an image observation references its payload', where, field='payload')
        if payload.media_type not in modality_row['payloads']:
            s.fail(INVALID_VALUE, f'media type {payload.media_type!r} is not allowed for this modality '
                   f'(vocabulary: {list(modality_row["payloads"])})', f'{where}.payload')

    def to_dict(self):
        out = {'block': self.block, 'version': self.version, 'width': self.width, 'height': self.height,
               'encoding': self.encoding, 'frame_id': self.frame_id}
        s.put(out, 'unit_scale_m', self.unit_scale_m)
        s.put(out, 'calibration_id', self.calibration_id)
        out['transforms'] = [t.to_dict() for t in self.transforms]
        if self.pose_at_stamp is not None:
            out['pose_at_stamp'] = self.pose_at_stamp.to_dict()
        return out


INSTANCE_BLOCK_MAJOR = 1
MAX_INSTANCE_ID = 65535                 # a 16-bit PNG
BOX_MASK_TOLERANCE_PX = 1               # an annotator box and its mask's extent agree within 1 px per edge


def mask_extent(pixels) -> list:
    """``[x, y, w, h]`` of the True pixels of a boolean image (``None`` when there is none): the box of whole
    pixels, origin at the top-left pixel corner."""
    ys, xs = np.nonzero(pixels)
    if not len(xs):
        return None
    return [int(xs.min()), int(ys.min()), int(xs.max() - xs.min() + 1), int(ys.max() - ys.min() + 1)]


def box_matches_extent(box, extent) -> bool:
    """Each edge of ``box`` within ``BOX_MASK_TOLERANCE_PX`` of the same edge of ``extent`` (both ``[x, y, w, h]``)."""
    if extent is None:
        return False
    edges = lambda b: (b[0], b[1], b[0] + b[2], b[1] + b[3])        # noqa: E731
    return all(abs(a - b) <= BOX_MASK_TOLERANCE_PX for a, b in zip(edges(box), edges(extent)))


def _instance(width, height):
    def check(value, where):
        f = s.Fields(value, where)
        out = {'id': f.req('id', s.integer(1)), 'object': f.req('object', s.ident), 'class': f.req('class', s.ident),
               'pixels': f.req('pixels', s.integer(1))}
        box = f.opt('box_tight_xywh_px', s.list_of(s.integer(0), length=4))
        if box is not None:
            out['box_tight_xywh_px'] = list(box)
        s.put(out, 'occlusion', f.opt('occlusion', s.real))
        f.done()
        if out['id'] > MAX_INSTANCE_ID:
            s.fail(INVALID_VALUE, f'an instance id is at most {MAX_INSTANCE_ID} (a 16-bit mask)', f'{where}.id')
        if box is not None:
            x, y, w, h = box
            if w < 1 or h < 1 or x + w > width or y + h > height:
                s.fail(INVALID_VALUE, f'box {list(box)} is not a non-empty box inside the {width}x{height} image',
                       f'{where}.box_tight_xywh_px')
            if out['pixels'] > w * h:
                s.fail(INVALID_VALUE, f'{out["pixels"]} pixels do not fit in the {w}x{h} box', where)
        if 'occlusion' in out and not 0.0 <= out['occlusion'] <= 1.0:
            s.fail(INVALID_VALUE, 'must be within [0, 1]', f'{where}.occlusion')
        return out
    return check


@dataclass(frozen=True)
class InstanceImageBlock:
    """The ``instance_image`` data block (D0.7): what an instance-id render's pixels mean.

    The payload is a 16-bit mask of COMPACT instance ids: 0 is background (or a prim the generator does not label),
    ``1..N`` are the ``instances``, numbered in the order of their ``object`` keys. Each instance names its object
    key, the scene's semantic class of that object, its visible pixel count and, when the render's 2-D box
    annotator reported it, its tight box ``[x, y, w, h]`` in pixels (origin at the top-left pixel corner).
    ``annotator`` names what produced the mask and ``box_annotator`` what produced the boxes (required when any
    instance carries one).

    This is the ANNOTATOR OUTPUT of a synthetic camera, recorded with the session so that it is sealed, hashed and
    verified with it. It is not a label set: a canonical label is a LabelSet (``labels.py``), derived from it with
    SYNTHETIC_GROUND_TRUTH provenance (``ground_truth.py``); exporters read LabelSets only.
    """
    version: str
    width: int
    height: int
    encoding: str
    frame_id: str
    instances: tuple
    annotator: str
    box_annotator: str = None
    calibration_id: str = None
    block = 'instance_image'

    @classmethod
    def from_dict(cls, value, modality_row, block_row, where='data'):
        f = s.Fields(value, where)
        f.req('block', s.token)
        width, height = f.req('width', s.integer(1)), f.req('height', s.integer(1))
        block = cls(version=f.req('version', _block_version(INSTANCE_BLOCK_MAJOR, block_row)),
                    width=width, height=height,
                    encoding=f.req('encoding', s.token),
                    frame_id=f.req('frame_id', s.token),
                    instances=tuple(f.req('instances', s.list_of(_instance(width, height)))),
                    annotator=f.req('annotator', s.token),
                    box_annotator=f.opt('box_annotator', s.token),
                    calibration_id=f.opt('calibration_id', s.calibration_id))
        f.done()
        if any('box_tight_xywh_px' in i for i in block.instances) and block.box_annotator is None:
            s.fail(INVALID_VALUE, 'instance boxes name their box_annotator', where)
        if block.encoding not in modality_row['encodings']:
            s.fail(INVALID_VALUE, f'encoding {block.encoding!r} is not allowed for this modality '
                   f'(vocabulary: {list(modality_row["encodings"])})', f'{where}.encoding')
        ids = [i['id'] for i in block.instances]
        if ids != list(range(1, len(ids) + 1)):
            s.fail(INVALID_VALUE, 'instance ids are compact: 1..N in order', f'{where}.instances')
        objects = [i['object'] for i in block.instances]
        if objects != sorted(objects) or len(set(objects)) != len(objects):
            s.fail(INVALID_VALUE, 'instances are numbered in the order of their unique object keys',
                   f'{where}.instances')
        return block

    check_payload = staticmethod(ImageBlock.check_payload)
    requires_payload = True

    def mask_problems(self, mask) -> list:
        """What the decoded mask breaks of this block: its size, its ids (exactly ``{0} | ids``, every listed
        instance present) and each instance's pixel count and tight-box extent rule (``BOX_MASK_TOLERANCE_PX``)."""
        if mask.ndim != 2 or mask.shape != (self.height, self.width):
            return [f'the mask is {"x".join(map(str, mask.shape[::-1]))}, the block says {self.width}x{self.height}']
        values, counts = np.unique(mask, return_counts=True)
        found = {int(v): int(c) for v, c in zip(values, counts) if v != 0}
        problems = []
        listed = {i['id']: i for i in self.instances}
        if set(found) != set(listed):
            problems.append(f'mask ids {sorted(found)} are not the block\'s instances {sorted(listed)}')
        for iid in sorted(set(found) & set(listed)):
            inst = listed[iid]
            if found[iid] != inst['pixels']:
                problems.append(f'instance {iid} covers {found[iid]} pixels, the block says {inst["pixels"]}')
            box = inst.get('box_tight_xywh_px')
            if box is not None and not box_matches_extent(box, mask_extent(mask == iid)):
                problems.append(f'instance {iid}: the annotator box {box} and the mask extent '
                                f'{mask_extent(mask == iid)} differ by more than {BOX_MASK_TOLERANCE_PX} px')
        return problems

    def to_dict(self):
        out = {'block': self.block, 'version': self.version, 'width': self.width, 'height': self.height,
               'encoding': self.encoding, 'frame_id': self.frame_id,
               'instances': [copy.deepcopy(i) for i in self.instances], 'annotator': self.annotator}
        s.put(out, 'box_annotator', self.box_annotator)
        s.put(out, 'calibration_id', self.calibration_id)
        return out


VECTOR_BLOCK_MAJOR = 1
MAX_VECTOR_VALUES = 64                  # one low-rate reading; high-rate streams are D1's Series record


@dataclass(frozen=True)
class VectorBlock:
    """The ``vector`` data block (test D0-J, D0_SCHEMA_PROPOSAL.md §3): ONE low-rate non-image reading -- e.g. an
    IMU sample of a ``capture_tool`` sensor -- whose values live in the record itself, so the observation references
    NO payload. ``units[i]`` is the unit of ``values[i]``. A run of these is not a time series: 100-500 Hz streams are
    D1's Series record (D0_DESIGN.md §5)."""
    version: str
    frame_id: str
    values: tuple
    units: tuple
    block = 'vector'
    calibration_id = None               # the reader contract: a vector reading has no camera calibration
    requires_payload = False            # its values are in the record

    @classmethod
    def from_dict(cls, value, modality_row, block_row, where='data'):
        f = s.Fields(value, where)
        f.req('block', s.token)
        vector = cls(version=f.req('version', _block_version(VECTOR_BLOCK_MAJOR, block_row)),
                     frame_id=f.req('frame_id', s.token),
                     values=tuple(f.req('values', s.list_of(s.real, min_length=1))),
                     units=tuple(f.req('units', s.list_of(s.token, min_length=1))))
        f.done()
        if len(vector.values) != len(vector.units):
            s.fail(INVALID_VALUE, f'{len(vector.values)} values but {len(vector.units)} units: one unit per value',
                   where)
        if len(vector.values) > MAX_VECTOR_VALUES:
            s.fail(INVALID_VALUE, f'{len(vector.values)} values: a vector observation holds one reading (at most '
                   f'{MAX_VECTOR_VALUES} values); a series is a D1 Series record', where)
        return vector

    @staticmethod
    def check_payload(payload, modality_row, where):
        """The values are in the record: a vector observation references no payload file."""
        if payload is not None:
            s.fail(INVALID_VALUE, 'a vector observation holds its values in the record and references no payload',
                   f'{where}.payload')

    def to_dict(self):
        return {'block': self.block, 'version': self.version, 'frame_id': self.frame_id,
                'values': list(self.values), 'units': list(self.units)}


def _block_version(major, block_row):
    """The block's major must be the one this reader implements AND the one the vocabulary row declares."""
    def check(value, where):
        value = s.string(value, where)
        parts = value.split('.')
        if len(parts) != 2 or not all(p.isdigit() for p in parts):
            s.fail(INVALID_VALUE, f'{value!r} is not a <major>.<minor> block version', where)
        declared = int(block_row['version'].split('.')[0])
        if int(parts[0]) != major or int(parts[0]) != declared:
            s.fail(SCHEMA_MAJOR_UNSUPPORTED, f'block version {value!r}: this reader supports major {major}, the '
                   f'vocabulary declares {block_row["version"]}', where)
        return value
    return check


# The data-block readers this core implements. A vocabulary block without a reader is refused.
# The contract of a reader class: ``block`` (its name), ``from_dict(value, modality_row, block_row, where)``,
# ``check_payload(payload, modality_row, where)`` (which payload, if any, the block needs), ``requires_payload``,
# ``to_dict()`` and a ``calibration_id`` attribute (``None`` when the block has no calibration).
_BLOCK_READERS = {'image': ImageBlock, 'instance_image': InstanceImageBlock, 'vector': VectorBlock}


def modality_problem(vocab, modality: str):
    """Why ``modality`` of ``vocab`` cannot be recorded by this core (``None`` if it can): its data block has no
    reader, or its vocabulary row lists payload media types for a block that holds its values in the record (or
    none for a block whose data is a payload file). Checked when a session declares its streams."""
    row = vocab.modalities[modality]
    reader = _BLOCK_READERS.get(row['data_block'])
    if reader is None:
        return DATA_BLOCK_UNSUPPORTED, f'no reader for data block {row["data_block"]!r} in this core'
    if reader.requires_payload != bool(row['payloads']):
        return INVALID_VALUE, (f'data block {row["data_block"]!r} ' + (
            'stores a payload file: its modality row must list encodings and payloads' if reader.requires_payload
            else 'holds its values in the record: its modality row lists no encodings or payloads'))
    return None


@dataclass(frozen=True)
class Observation:
    observation_id: str                 # <capture_id>/<sensor_id>/<stream>
    sensor_id: str
    stream: str
    modality: str
    stamp: Timestamp
    data: object                        # a data-block reader instance (ImageBlock / InstanceImageBlock / VectorBlock)
    payload: PayloadRef = None
    receipt: Receipt = None
    device_stamp: Timestamp = None
    quality: Quality = None
    extensions: dict = field(default_factory=dict)

    @property
    def stream_key(self) -> str:
        return f'{self.sensor_id}/{self.stream}'

    @classmethod
    def from_dict(cls, value, vocab, where='observation') -> 'Observation':
        f = s.Fields(value, where)
        observation_id = f.req('observation_id', s.string)
        sensor_id = f.req('sensor_id', s.ident)
        stream = f.req('stream', s.ident)
        modality = f.req('modality', vocab.checker('modalities'))
        row = vocab.modalities[modality]
        data_value = f.req('data', s.opaque)
        block = s.Fields(data_value, f'{where}.data').opt('block', s.token)
        if block is None:
            s.fail(MISSING_FIELD, "required field 'block' is absent", f'{where}.data', field='block')
        vocab.require('data_blocks', block, f'{where}.data.block')
        if block != row['data_block']:
            s.fail(INVALID_VALUE, f'modality {modality!r} uses data block {row["data_block"]!r}, not {block!r}',
                   f'{where}.data.block')
        reader = _BLOCK_READERS.get(block)
        if reader is None:
            raise DataError(DATA_BLOCK_UNSUPPORTED, f'{where}.data: no reader for data block {block!r} in this '
                            'core', context={'where': f'{where}.data', 'block': block})
        observation = cls(
            observation_id=observation_id, sensor_id=sensor_id, stream=stream, modality=modality,
            stamp=f.req('stamp', lambda v, w: Timestamp.from_dict(v, vocab, w)),
            data=reader.from_dict(data_value, row, vocab.data_blocks[block], f'{where}.data'),
            payload=f.opt('payload', PayloadRef.from_dict),
            receipt=f.opt('receipt', Receipt.from_dict),
            device_stamp=f.opt('device_stamp', lambda v, w: Timestamp.from_dict(v, vocab, w)),
            quality=f.opt('quality', Quality.from_dict),
            extensions=f.extensions())
        f.done()
        if not observation_id.endswith(f'/{sensor_id}/{stream}'):
            s.fail(INVALID_VALUE, f'observation_id {observation_id!r} must end with "/{sensor_id}/{stream}"',
                   f'{where}.observation_id')
        reader.check_payload(observation.payload, row, where)
        return observation

    def to_dict(self) -> dict:
        out = {'observation_id': self.observation_id, 'sensor_id': self.sensor_id, 'stream': self.stream,
               'modality': self.modality, 'stamp': self.stamp.to_dict(), 'data': self.data.to_dict()}
        for name in ('payload', 'receipt', 'device_stamp', 'quality'):
            value = getattr(self, name)
            if value is not None:
                out[name] = value.to_dict()
        if self.extensions:
            out['extensions'] = copy.deepcopy(self.extensions)
        return out


__all__ = ['Observation', 'ImageBlock', 'InstanceImageBlock', 'VectorBlock', 'modality_problem', 'mask_extent',
           'box_matches_extent', 'PayloadRef', 'Receipt', 'Quality', 'Transform', 'PoseAtStamp']
