"""The session's ``acquisition`` provenance block (``trainit-acquisition/1.0``; no ROS imports).

It answers, per session: what physical device produced the data, where it was attached, which host acquired it, and
how it reached the recorder -- WITHOUT a device schema and without touching Observation / Capture:

- ``recorder``: the host this recorder runs on (OBSERVED locally) and its DECLARED logical id, if any;
- ``transport``: how the camera contract reached the recorder (``ROS2_DDS``: the RMW and the domain, OBSERVED);
- ``sensors.<id>``: the camera's own ``trainit-camera-source/1.0`` document -- the JSON object the source published,
  as parsed and validated (its
  DISCOVERED / DECLARED / OBSERVED blocks, the hardware layer's facts), its hash, and the RELATION of source and
  recorder:

  - ``LOCAL_SENSOR_LOCAL_RECORDER``: the source document's boot-id hash equals the recorder's (same host, same
    boot): the frames never left this host;
  - ``REMOTE_SENSOR_SOURCE``: both hashes are known and differ: the frames came from another host over the
    transport;
  - ``UNKNOWN``: no source document (or no boot-id hash in it). Never guessed, never read from a topic name.

The source document is also CROSS-CHECKED against the session's source descriptor before anything is recorded:
its backend must be the session's, and an OBSERVED stamp origin must be the descriptor's. A contradiction refuses
the session (``SOURCE_DESCRIPTOR_INVALID``): a claim the evidence contradicts is never persisted.

A replay records its own recorder and transport, and carries its original session's block under ``original`` (without
that block's own ``original``: a chain of replays is followed through ``replay_of``, never nested). A replay receives
no source document: one on its topic is a LIVE camera publishing into the replay's namespace, and refuses the session.

``stamp_origin_verified`` is absent without a document, ``False`` when the document establishes no stamp origin, and
``True`` when its OBSERVED origin is the descriptor's. The source's ``published_wall_utc_ns`` is its own claim, not
evidence of freshness: the recorder drops a document whose topic has no publisher any more before calling here.
"""
import copy
import json

from trainit_data._canonical import sha256_json
from trainit_data.errors import INVALID_VALUE, SOURCE_DESCRIPTOR_INVALID, DataError
from trainit_hardware_layer.camera import source_info as source_doc

SCHEMA = 'trainit-acquisition/1.0'
RELATIONS = ('LOCAL_SENSOR_LOCAL_RECORDER', 'REMOTE_SENSOR_SOURCE', 'UNKNOWN')


def recorder_host(declared_host_id=None, proc_root='/proc') -> dict:
    """The recorder's own host: OBSERVED by the recorder where it runs (+ the DECLARED logical id, if any)."""
    host = dict(source_doc.host_facts(proc_root), origin='OBSERVED')
    out = {'host': host}
    if declared_host_id:
        out['declared'] = {'origin': 'DECLARED', 'host_id': declared_host_id}
    return out


def parse_source_info(raw: str, topic: str) -> dict:
    """The validated source document of ``raw`` (the String a camera published), else ``INVALID_VALUE``."""
    try:
        return source_doc.parse(json.loads(raw))
    except (ValueError, source_doc.SourceInfoError) as exc:
        raise DataError(INVALID_VALUE, f'{topic}: not a valid {source_doc.SCHEMA} document ({exc})',
                        context={'topic': topic}) from None


def relation(document, recorder_boot):
    """(relation, basis) of a source document and the recorder's boot-id hash."""
    if document is None:
        return 'UNKNOWN', 'no source document'
    theirs = (document.get('host') or {}).get('boot_id_sha256')
    if not theirs or not recorder_boot:
        return 'UNKNOWN', 'no boot-id hash on one side'
    if theirs == recorder_boot:
        return 'LOCAL_SENSOR_LOCAL_RECORDER', 'the source and the recorder report the same boot-id hash'
    return 'REMOTE_SENSOR_SOURCE', 'the source and the recorder report different boot-id hashes'


def _cross_check(document, sensor_id, backend, descriptor):
    """Refuse a source document that contradicts the session; answer whether it VERIFIED the stamp origin."""
    if backend == 'replay':
        raise DataError(SOURCE_DESCRIPTOR_INVALID, f'{sensor_id}: a replay session receives a live camera\'s source '
                        f'document (backend {document["backend"]!r}): another source publishes into the replay\'s '
                        'namespace', context={'sensor_id': sensor_id})
    if document['backend'] != backend:
        raise DataError(SOURCE_DESCRIPTOR_INVALID, f'{sensor_id}: the camera reports backend {document["backend"]!r}, '
                        f'the session is {backend!r}', context={'sensor_id': sensor_id})
    observed = (document.get('timestamping') or {}).get('stamp_origin')
    if observed is None:
        return False
    if observed != descriptor.get('stamp_origin'):
        raise DataError(SOURCE_DESCRIPTOR_INVALID,
                        f'{sensor_id}: the camera OBSERVED stamp origin {observed!r}, the source descriptor declares '
                        f'{descriptor.get("stamp_origin")!r}: the declared timestamp semantics are contradicted',
                        context={'sensor_id': sensor_id, 'observed': observed,
                                 'declared': descriptor.get('stamp_origin')})
    return True


def build(cfg: dict, received: dict, *, transport: dict, recorder: dict, original=None) -> dict:
    """The block for a session about to start.

    ``cfg``: the resolved recording configuration. ``received``: ``{sensor_id: raw String data}`` of the latest
    source document per sensor (absent = none received). ``transport``: the OBSERVED transport facts.
    ``recorder``: :func:`recorder_host`. ``original``: ``{session_id, acquisition?}`` of a replay's original.
    """
    backend, descriptor = cfg['source']['backend'], cfg['source']['descriptor']
    recorder_boot = recorder['host'].get('boot_id_sha256')
    sensors = {}
    for sensor in cfg['sensors']:
        sid, info = sensor['sensor_id'], sensor['source_info']
        raw = received.get(sid)
        if raw is None and info['required']:
            raise DataError(SOURCE_DESCRIPTOR_INVALID, f'{sid}: no source document on {info["topic"]} (the '
                            'configuration requires one: is the camera backend publishing it?)',
                            context={'sensor_id': sid, 'topic': info['topic']})
        document = None if raw is None else parse_source_info(raw, info['topic'])
        entry = {'source_info_topic': info['topic']}
        entry['relation'], entry['relation_basis'] = relation(document, recorder_boot)
        if document is not None:
            entry['stamp_origin_verified'] = _cross_check(document, sid, backend, descriptor)
            entry['source_info'] = document
            entry['source_info_sha256'] = sha256_json(document)
        sensors[sid] = entry
    block = {'schema': SCHEMA, 'recorder': copy.deepcopy(recorder), 'transport': dict(transport), 'sensors': sensors}
    if original is not None:
        block['original'] = copy.deepcopy(original)
        block['original'].get('acquisition', {}).pop('original', None)      # one level: chains follow replay_of
    return block


__all__ = ['SCHEMA', 'RELATIONS', 'build', 'recorder_host', 'relation', 'parse_source_info']
