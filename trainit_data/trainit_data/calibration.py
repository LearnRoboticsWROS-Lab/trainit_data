"""CalibrationRecord: the content-addressed, immutable calibration of one stream (PUBLIC).

``trainit-calibration/1.0`` (D0_SCHEMA_PROPOSAL.md §4, D0_DESIGN.md §8). One record per stream: its
intrinsics (a CameraInfo, verbatim), their provenance (synthetic render / device reported / measured), the
depth semantics where the stream is depth, the extrinsics with their provenance (nominal description /
measured hand-eye / synthetic scene) and the transform tree they came from.

Identity: ``sha256`` = the content hash (``_canonical.sha256_json``) of the record WITHOUT its
``calibration_id`` and ``sha256``; ``calibration_id`` = ``"cal-"`` + its first 16 hex. Same canonical
content -> same id; any content change -> another id. Real-valued fields are normalised to float, so
``424`` and ``424.0`` are the same content. The opaque sub-trees (provenance ``ref``, ``extensions``) are carried
verbatim and NOT normalised: inside them ``1`` and ``1.0`` are different content.

D0.1 owns the persisted record only. Mapping a ROS CameraInfo / TF into it, querying a device and any
hand-eye computation come later (D0.2 / D0.4); the core never computes a calibration.
"""
import copy
from dataclasses import dataclass, field

from . import _strict as s
from ._canonical import sha256_json
from .errors import HASH_MISMATCH, INVALID_VALUE, DataError

SCHEMA_NAME = 'trainit-calibration'
SCHEMA_MAJOR = 1
SCHEMA = 'trainit-calibration/1.0'
DEPTH_INVALID = frozenset({'zero', 'nan_or_inf'})       # CLOSED: how a depth payload marks "no measurement"


def _roi(value, where):
    f = s.Fields(value, where)
    roi = {'x': f.req('x', s.integer(0)), 'y': f.req('y', s.integer(0)), 'width': f.req('width', s.integer(0)),
           'height': f.req('height', s.integer(0)), 'do_rectify': f.req('do_rectify', s.boolean)}
    f.done()
    return roi


@dataclass(frozen=True)
class Intrinsics:
    width: int
    height: int
    distortion_model: str
    K: tuple
    D: tuple
    R: tuple
    P: tuple
    binning: tuple = (0, 0)
    roi: dict = field(default_factory=lambda: {'x': 0, 'y': 0, 'width': 0, 'height': 0, 'do_rectify': False})

    @classmethod
    def from_dict(cls, value, where='intrinsics'):
        f = s.Fields(value, where)
        intr = cls(width=f.req('width', s.integer(1)), height=f.req('height', s.integer(1)),
                   distortion_model=f.req('distortion_model', s.token),
                   K=f.req('K', s.vector(9)), D=f.req('D', s.list_of(s.real)), R=f.req('R', s.vector(9)),
                   P=f.req('P', s.vector(12)),
                   binning=f.opt('binning', s.list_of(s.integer(0), length=2), (0, 0)),
                   roi=f.opt('roi', _roi, {'x': 0, 'y': 0, 'width': 0, 'height': 0, 'do_rectify': False}))
        f.done()
        return intr

    @property
    def calibrated(self) -> bool:
        """K is a usable pinhole camera matrix [fx s cx; 0 fy cy; 0 0 1]. REP-104 marks an uncalibrated camera with
        K[0] == 0, and some drivers publish an all-zero CameraInfo (D0-I "bad calibration")."""
        K = self.K
        exact = 1e-6            # a driver's float noise on the structural zeros / one, never a scale
        return (K[0] > 0.0 and K[4] > 0.0 and abs(K[3]) <= exact and abs(K[6]) <= exact and abs(K[7]) <= exact
                and abs(K[8] - 1.0) <= exact)

    def to_dict(self):
        return {'width': self.width, 'height': self.height, 'distortion_model': self.distortion_model,
                'K': list(self.K), 'D': list(self.D), 'R': list(self.R), 'P': list(self.P),
                'binning': list(self.binning), 'roi': dict(self.roi)}


@dataclass(frozen=True)
class Provenance:
    kind: str                           # vocabulary: intrinsic_provenance / extrinsic_provenance
    ref: dict = field(default_factory=dict)
    notes: str = None

    @classmethod
    def from_dict(cls, value, vocab, category, where='provenance'):
        f = s.Fields(value, where)
        prov = cls(kind=f.req('kind', vocab.checker(category)), ref=f.opt('ref', s.opaque, {}),
                   notes=f.opt('notes', s.text))
        f.done()
        return prov

    def to_dict(self):
        out = {'kind': self.kind}
        if self.ref:
            out['ref'] = copy.deepcopy(self.ref)
        s.put(out, 'notes', self.notes)
        return out


@dataclass(frozen=True)
class DepthSemantics:
    invalid: str                        # 'zero' | 'nan_or_inf'
    registered_to: str = None           # the stream whose pixel grid this depth is registered to
    registration_basis: str = None      # e.g. 'same_render_product', 'driver_align_depth'

    @classmethod
    def from_dict(cls, value, where='depth'):
        f = s.Fields(value, where)
        depth = cls(invalid=f.req('invalid', s.one_of(DEPTH_INVALID)),
                    registered_to=f.opt('registered_to', s.ident),
                    registration_basis=f.opt('registration_basis', s.token))
        f.done()
        return depth

    def to_dict(self):
        out = {'invalid': self.invalid}
        s.put(out, 'registered_to', self.registered_to)
        s.put(out, 'registration_basis', self.registration_basis)
        return out


@dataclass(frozen=True)
class Extrinsic:
    """What the runtime BELIEVED at capture (the TF), plus where that belief came from."""
    parent_frame: str
    child_frame: str
    translation: tuple
    rotation_xyzw: tuple
    provenance: Provenance

    @classmethod
    def from_dict(cls, value, vocab, where='extrinsic'):
        f = s.Fields(value, where)
        ext = cls(parent_frame=f.req('parent_frame', s.token), child_frame=f.req('child_frame', s.token),
                  translation=f.req('translation', s.vector(3)), rotation_xyzw=f.req('rotation_xyzw', s.quaternion),
                  provenance=f.req('provenance',
                                   lambda v, w: Provenance.from_dict(v, vocab, 'extrinsic_provenance', w)))
        f.done()
        if ext.parent_frame == ext.child_frame:
            s.fail(INVALID_VALUE, 'parent_frame and child_frame are the same frame', where)
        return ext

    def to_dict(self):
        return {'parent_frame': self.parent_frame, 'child_frame': self.child_frame,
                'translation': list(self.translation), 'rotation_xyzw': list(self.rotation_xyzw),
                'provenance': self.provenance.to_dict()}


def _transform_tree(value, where):
    f = s.Fields(value, where)
    tree = {'source': f.req('source', s.token)}
    description = f.opt('description_sha256', s.sha256)
    if description is not None:
        tree['description_sha256'] = description
    f.done()
    return tree


@dataclass(frozen=True)
class CalibrationRecord:
    schema: str
    calibration_id: str
    sha256: str
    sensor_id: str
    stream: str
    optical_frame: str
    intrinsics: Intrinsics
    intrinsics_provenance: Provenance
    extrinsics: tuple = ()
    depth: DepthSemantics = None
    transform_tree: dict = None
    extensions: dict = field(default_factory=dict)

    @classmethod
    def from_dict(cls, value, vocab, where='calibration') -> 'CalibrationRecord':
        """Parse a stored record; its id and sha256 must be those of its content (else ``HASH_MISMATCH``)."""
        f = s.Fields(value, where)
        claimed_id = f.req('calibration_id', s.calibration_id)
        claimed_sha = f.req('sha256', s.sha256)
        record = cls._parse(f, vocab)
        expected = record.to_dict()
        if (claimed_id, claimed_sha) != (expected['calibration_id'], expected['sha256']):
            raise DataError(HASH_MISMATCH, f'{where}: calibration_id / sha256 do not match the record content '
                            f'(content gives {expected["calibration_id"]})',
                            context={'where': where, 'claimed': claimed_id, 'content': expected['calibration_id']})
        return record

    @classmethod
    def build(cls, content: dict, vocab) -> 'CalibrationRecord':
        """A new record from its content (no id, no sha256): the identity is computed here."""
        if isinstance(content, dict) and ('calibration_id' in content or 'sha256' in content):
            s.fail(INVALID_VALUE, 'build() takes the content only; the identity is derived', 'calibration')
        content = dict(content)
        content.setdefault('schema', SCHEMA)
        record = cls._parse(s.Fields(content, 'calibration'), vocab)
        # A NEW record must hold a calibrated camera matrix (0.9.0). The rule is enforced here, where records are
        # created, and NOT in ``from_dict``: a stored record is immutable and stays readable; ``verify`` flags an
        # uncalibrated one written before 0.9.0 as a WARNING.
        if not record.intrinsics.calibrated:
            s.fail(INVALID_VALUE, f'K is not a calibrated pinhole camera matrix (fx > 0, fy > 0, last row 0 0 1): '
                   f'{list(record.intrinsics.K)}', 'calibration.intrinsics.K')
        return record

    @classmethod
    def _parse(cls, f, vocab):
        fields = dict(
            schema=f.req('schema', s.schema(SCHEMA_NAME, SCHEMA_MAJOR)),
            sensor_id=f.req('sensor_id', s.ident),
            stream=f.req('stream', s.ident),
            optical_frame=f.req('optical_frame', s.token),
            intrinsics=f.req('intrinsics', Intrinsics.from_dict),
            intrinsics_provenance=f.req('intrinsics_provenance',
                                        lambda v, w: Provenance.from_dict(v, vocab, 'intrinsic_provenance', w)),
            extrinsics=f.opt('extrinsics', s.list_of(lambda v, w: Extrinsic.from_dict(v, vocab, w)), ()),
            depth=f.opt('depth', DepthSemantics.from_dict),
            transform_tree=f.opt('transform_tree', _transform_tree),
            extensions=f.extensions())
        f.done()
        sha = sha256_json(_content_dict(**fields))
        return cls(calibration_id='cal-' + sha[len('sha256:'):][:16], sha256=sha, **fields)

    def content(self) -> dict:
        """The hashed content: the record without ``calibration_id`` and ``sha256``."""
        return _content_dict(schema=self.schema, sensor_id=self.sensor_id, stream=self.stream,
                             optical_frame=self.optical_frame, intrinsics=self.intrinsics,
                             intrinsics_provenance=self.intrinsics_provenance, extrinsics=self.extrinsics,
                             depth=self.depth, transform_tree=self.transform_tree, extensions=self.extensions)

    def to_dict(self) -> dict:
        out = self.content()
        out['calibration_id'] = self.calibration_id
        out['sha256'] = self.sha256
        return out


def _content_dict(*, schema, sensor_id, stream, optical_frame, intrinsics, intrinsics_provenance, extrinsics,
                  depth, transform_tree, extensions) -> dict:
    out = {'schema': schema, 'sensor_id': sensor_id, 'stream': stream, 'optical_frame': optical_frame,
           'intrinsics': intrinsics.to_dict(), 'intrinsics_provenance': intrinsics_provenance.to_dict(),
           'extrinsics': [e.to_dict() for e in extrinsics]}
    if depth is not None:
        out['depth'] = depth.to_dict()
    if transform_tree is not None:
        out['transform_tree'] = dict(transform_tree)
    if extensions:
        out['extensions'] = copy.deepcopy(extensions)
    return out


__all__ = ['CalibrationRecord', 'Intrinsics', 'Provenance', 'DepthSemantics', 'Extrinsic', 'DEPTH_INVALID']
