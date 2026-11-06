"""Lossless payload codecs (INTERNAL).

| media type          | arrays accepted                          | file  |
|---------------------|------------------------------------------|-------|
| ``image/png``         | uint8, 1 / 3 / 4 channels (mono8, rgb8)  | .png  |
| ``image/png;bits=16`` | uint16, 1 channel (16UC1 depth in mm)    | .png  |
| ``application/x-npy`` | any numeric dtype (32FC1 depth in m)     | .npy  |

Every codec round-trips bit-exactly; NaN / inf survive ``.npy``. Nothing is converted: an array whose
dtype or channel count is not the declared encoding's is refused (``ENCODING_REFUSED``), so a unit or
type change can never happen silently. Bytes are produced little-endian whatever the host.
"""
import io

import numpy as np
from PIL import Image

from .errors import ENCODING_REFUSED, PAYLOAD_INVALID, DataError

_PNG_MODES = {1: 'L', 3: 'RGB', 4: 'RGBA'}


def _refuse(message, **context):
    raise DataError(ENCODING_REFUSED, message, context=context)


def _channels(array) -> int:
    return 1 if array.ndim == 2 else array.shape[2]


def _encode_png8(array):
    mode = _PNG_MODES.get(_channels(array))
    if array.dtype != np.uint8 or mode is None:
        _refuse('image/png stores uint8 arrays with 1, 3 or 4 channels', dtype=str(array.dtype))
    out = io.BytesIO()
    Image.fromarray(np.ascontiguousarray(array)).save(out, format='PNG')     # mode inferred: L / RGB / RGBA
    return out.getvalue()


def _encode_png16(array):
    if array.dtype.kind != 'u' or array.dtype.itemsize != 2 or array.ndim != 2:
        _refuse('image/png;bits=16 stores single-channel uint16 arrays', dtype=str(array.dtype))
    out = io.BytesIO()
    Image.fromarray(np.ascontiguousarray(array, dtype='<u2')).save(out, format='PNG')   # mode inferred: I;16
    return out.getvalue()


def _encode_npy(array):
    out = io.BytesIO()
    np.save(out, np.ascontiguousarray(array, dtype=array.dtype.newbyteorder('<')), allow_pickle=False)
    return out.getvalue()


def _decode_png(data: bytes):
    with Image.open(io.BytesIO(data)) as image:
        if image.format != 'PNG':
            raise ValueError('not a PNG')
        if image.mode in ('I;16', 'I;16L', 'I'):
            # Pillow < 10 opens 16-bit grey PNGs as 32-bit 'I'; the values are the stored uint16 ones.
            array = np.asarray(image)
            if array.dtype != np.uint16:
                if array.min(initial=0) < 0 or array.max(initial=0) > 0xFFFF:
                    raise ValueError('16-bit PNG values out of range')
                array = array.astype(np.uint16)
            return array
        return np.asarray(image)


def _decode_npy(data: bytes):
    return np.load(io.BytesIO(data), allow_pickle=False)


# media type -> (extension, encoder, decoder)
CODECS = {
    'image/png': ('.png', _encode_png8, _decode_png),
    'image/png;bits=16': ('.png', _encode_png16, _decode_png),
    'application/x-npy': ('.npy', _encode_npy, _decode_npy),
}


def check_array(array, encoding_row: dict, encoding: str):
    """The array is exactly the declared encoding (dtype + channels), else ``ENCODING_REFUSED``."""
    if not isinstance(array, np.ndarray):
        _refuse('the sample is not a numpy array', encoding=encoding)
    expected = np.dtype(encoding_row['dtype'])
    if array.dtype.kind != expected.kind or array.dtype.itemsize != expected.itemsize:
        _refuse(f'{encoding} is {expected}, the array is {array.dtype} (nothing is converted)', encoding=encoding)
    if array.ndim not in (2, 3) or _channels(array) != encoding_row['channels'] or array.size == 0:
        _refuse(f'{encoding} has {encoding_row["channels"]} channel(s); the array shape is {array.shape}',
                encoding=encoding)
    if array.ndim == 3 and encoding_row['channels'] == 1:
        _refuse(f'{encoding} is single-channel: pass an HxW array', encoding=encoding)


def encode(array, media_type: str) -> tuple:
    """(bytes, file extension). Refuses media types this core has no codec for."""
    codec = CODECS.get(media_type)
    if codec is None:
        _refuse(f'no codec for media type {media_type!r}', media_type=media_type)
    return codec[1](array), codec[0]


def decode(data: bytes, media_type: str, *, where: str):
    try:
        return CODECS[media_type][2](data)
    except (KeyError, ValueError, OSError) as exc:
        raise DataError(PAYLOAD_INVALID, f'{where}: cannot decode as {media_type}', context={'where': where},
                        cause=exc) from None


def accepts(array, media_type: str) -> bool:
    """Whether ``media_type`` stores this array losslessly (probed on a 1x1 slice: no full encode)."""
    codec = CODECS.get(media_type)
    if codec is None:
        return False
    try:
        codec[1](array[:1, :1])
    except DataError:
        return False
    return True


def default_media_type(array, payloads) -> str:
    """The first media type of the modality's vocabulary row that stores this array losslessly."""
    for media_type in payloads:
        if accepts(array, media_type):
            return media_type
    _refuse(f'no payload of {list(payloads)} stores a {array.dtype} array losslessly', dtype=str(array.dtype))
