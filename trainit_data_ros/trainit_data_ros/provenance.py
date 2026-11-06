"""Description and scene provenance, from STABLE installed artifacts (never from transient ROS runtime state).

``description``: the ROOT file robot_state_publisher expands (``{package, file, args}``). It is expanded with xacro
at session start; every file of its include closure is hashed and named machine-independently (package + path in
its share). The identity -- ``sha256`` -- covers the root, the arguments and the closure's file hashes; the
expanded URDF's hash is recorded beside it (``expanded_sha256``) but kept out of the identity, because an
expansion may embed machine paths. Meshes are not hashed: they do not change a transform.

``scene``: the simulator stage the operator is told to open (declared, ``{package, file, prim?}``): the installed
file's hash, ``completeness: ROOT_LAYER_ONLY`` (referenced layers are not hashed). Which stage was really open is
the operator's fact; this records which file it was declared to be.
"""
from pathlib import Path

from trainit_data._canonical import sha256_bytes, sha256_json
from trainit_data.errors import INVALID_VALUE, DataError


def ament_share(package: str) -> Path:
    from ament_index_python.packages import PackageNotFoundError, get_package_share_directory
    try:
        return Path(get_package_share_directory(package))
    except PackageNotFoundError:
        raise DataError(INVALID_VALUE, f'package {package!r} is not installed', context={'package': package}) from None


def _installed(spec, share_of) -> Path:
    path = share_of(spec['package']) / spec['file']
    if not path.is_file():
        raise DataError(INVALID_VALUE, f'{spec["package"]}/{spec["file"]} is not installed', context=dict(spec))
    return path


def package_relative(path) -> dict:
    """``{package, file}`` of an installed file (``.../share/<package>/<file>``), else ``{file: <name>}``."""
    parts = Path(path).parts
    for i in range(len(parts) - 2, -1, -1):
        if parts[i] == 'share' and i + 1 < len(parts) - 1:
            return {'package': parts[i + 1], 'file': '/'.join(parts[i + 2:])}
    return {'file': Path(path).name}


def description_provenance(spec: dict, share_of=ament_share) -> dict:
    import xacro
    root = _installed(spec, share_of)
    args = dict(spec.get('args', {}))
    del xacro.all_includes[:]
    try:
        document = xacro.process_file(str(root), mappings=dict(args))    # xacro adds its defaults to the dict
    except Exception as exc:  # noqa: BLE001 -- xacro raises many types; the refusal is what matters
        raise DataError(INVALID_VALUE, f'the description {spec["package"]}/{spec["file"]} does not expand',
                        context={'package': spec['package'], 'file': spec['file']}, cause=exc) from None
    closure, seen = [], set()
    for path in [str(root)] + list(xacro.all_includes):
        if path in seen:
            continue
        seen.add(path)
        entry = package_relative(path)
        entry['sha256'] = sha256_bytes(Path(path).read_bytes())
        closure.append(entry)
    identity = {'root': {'package': spec['package'], 'file': spec['file']}, 'args': args, 'files': closure}
    return dict(identity, expanded_sha256=sha256_bytes(document.toxml().encode('utf-8')),
                sha256=sha256_json(identity))


def scene_provenance(spec: dict, share_of=ament_share) -> dict:
    path = _installed(spec, share_of)
    out = {'package': spec['package'], 'file': spec['file'], 'sha256': sha256_bytes(path.read_bytes()),
           'completeness': 'ROOT_LAYER_ONLY'}
    if 'prim' in spec:
        out['prim'] = spec['prim']
    return out
