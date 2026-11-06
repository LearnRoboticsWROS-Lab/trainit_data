"""External dataset formats: exporters and label importers (PUBLIC, D0.6).

The canonical dataset is TrainIt's (``versions``, ``labels``, ``splits``); an external format is an ADAPTER at
the edge, never the layout. An exporter turns ONE dataset version into an external format and declares which
``content_kinds`` and label kinds it can represent; a format's label importer turns that format's annotations
back into a canonical label set. Both are pure (stdlib + numpy + Pillow): no training framework is imported, so
an export is plain files that a trainer reads in its own environment.

Registered in D0.6: ``yolo`` (YOLO DETECTION: ``trainit_data.formats.yolo``). Segmentation export is DEFERRED:
it needs contour extraction, and cv2 is not a core dependency.

The registry contract of an exporter: ``name``, ``version``, ``content_kinds``, ``label_kinds``,
``export(store, dataset_id, version, destination, config) -> sidecar`` and ``verify(destination, sidecar) ->
[Finding]``. An exporter never branches on a session's backend or transport: what it reads is canonical content.
"""
from ..errors import INVALID_VALUE, DataError

_EXPORTERS = {}


def register_exporter(exporter):
    if exporter.name in _EXPORTERS and _EXPORTERS[exporter.name] is not exporter:
        raise ValueError(f'an exporter named {exporter.name!r} is already registered')
    _EXPORTERS[exporter.name] = exporter
    return exporter


def get_exporter(name: str):
    if name not in _EXPORTERS:
        raise DataError(INVALID_VALUE, f'no exporter {name!r} (registered: {sorted(_EXPORTERS)})',
                        context={'exporter': name, 'registered': sorted(_EXPORTERS)})
    return _EXPORTERS[name]


def exporters() -> dict:
    return dict(_EXPORTERS)


from . import yolo  # noqa: E402,F401  (registers the built-in exporter)

__all__ = ['register_exporter', 'get_exporter', 'exporters']
