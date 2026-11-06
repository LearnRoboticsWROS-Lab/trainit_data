"""Architecture guards of the canonical core (ADR-0029 Decisions 1 and 6; D0_DESIGN.md §3).

The core must stay usable by a training environment and by Isaac's Python 3.11, neither of which has
the ROS tree. The checks below are static: every module of the package, whatever slice adds it, is
scanned. One more check imports the package in a process whose path holds no ROS directory.
"""
import ast
import os
import subprocess
import sys
from pathlib import Path

PKG_ROOT = Path(__file__).resolve().parents[1]
PKG = PKG_ROOT / 'trainit_data'

# Top-level modules the core must never import (ADR-0029 Decision 1).
FORBIDDEN = {
    # ROS
    'rclpy', 'rosidl_runtime_py', 'rosbag2_py', 'ament_index_python', 'launch', 'launch_ros',
    'sensor_msgs', 'std_msgs', 'std_srvs', 'geometry_msgs', 'builtin_interfaces', 'diagnostic_msgs',
    'tf2_ros', 'tf2_msgs', 'vision_msgs', 'cv_bridge', 'message_filters',
    # simulator
    'omni', 'isaacsim', 'pxr', 'carb',
    # libraries kept out of the core path
    'pydantic', 'cv2',
    # training frameworks / providers
    'torch', 'lerobot', 'ultralytics', 'tensorflow', 'jax',
}
FORBIDDEN_PREFIXES = ('trainit_',)            # no other TrainIt package...
ALLOWED_TRAINIT = {'trainit_data'}            # ...except the package itself


def _modules():
    return sorted(p for p in PKG.rglob('*.py'))


def _imports(path: Path):
    tree = ast.parse(path.read_text(), filename=str(path))
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                yield alias.name, node.lineno
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            yield node.module, node.lineno


def test_the_package_has_modules_to_scan():
    assert (PKG / '__init__.py').is_file() and (PKG / 'adapters' / '__init__.py').is_file()


def test_the_core_imports_no_ros_simulator_or_training_framework():
    bad = []
    for path in _modules():
        for name, line in _imports(path):
            top = name.split('.')[0]
            if top in FORBIDDEN or (top.startswith(FORBIDDEN_PREFIXES) and top not in ALLOWED_TRAINIT):
                bad.append(f'{path.relative_to(PKG_ROOT)}:{line}: {name}')
    assert not bad, 'forbidden imports in the canonical core:\n' + '\n'.join(bad)


def test_the_core_imports_only_the_stdlib_numpy_and_pillow():
    # The ALLOW-list of ADR-0029 Decision 1 (the deny-list above stays as a second, explicit check).
    allowed = set(sys.stdlib_module_names) | {'numpy', 'PIL', 'trainit_data'}
    bad = []
    for path in _modules():
        for name, line in _imports(path):
            if name.split('.')[0] not in allowed:
                bad.append(f'{path.relative_to(PKG_ROOT)}:{line}: {name}')
    assert not bad, 'imports outside stdlib + numpy + Pillow in the canonical core:\n' + '\n'.join(bad)


def test_the_core_never_imports_its_adapters_or_examples():
    bad = []
    for path in _modules():
        if 'adapters' in path.relative_to(PKG).parts:
            continue                          # adapters may import the core, not the reverse
        for name, line in _imports(path):
            if name.startswith('trainit_data.adapters') or name.split('.')[0] == 'examples':
                bad.append(f'{path.relative_to(PKG_ROOT)}:{line}: {name}')
    assert not bad, 'the core imports an adapter or an example:\n' + '\n'.join(bad)


def test_the_package_imports_without_ros_on_the_path():
    # A process whose PYTHONPATH is ONLY this package's source: no /opt/ros, no workspace install.
    env = {k: v for k, v in os.environ.items()
           if k not in ('PYTHONPATH', 'AMENT_PREFIX_PATH', 'COLCON_PREFIX_PATH', 'ROS_DISTRO')}
    env['PYTHONPATH'] = str(PKG_ROOT)
    probe = ('import sys, trainit_data, trainit_data.adapters, trainit_data.cli, trainit_data.integrity, '
             'trainit_data.writer, trainit_data.reader, trainit_data.store, trainit_data.vocab; '
             'trainit_data.vocab.Vocabulary.default(); '
             'ros = [m for m in sys.modules if m.split(".")[0] in ("rclpy", "rosidl_runtime_py", "cv2", "pydantic")]; '
             'assert not ros, ros; '
             'assert not any("/opt/ros" in p for p in sys.path), sys.path; '
             'print(trainit_data.__version__)')
    out = subprocess.run([sys.executable, '-c', probe], env=env, capture_output=True, text=True)
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == _version_in('package.xml')


def _version_in(name: str) -> str:
    text = (PKG_ROOT / name).read_text()
    return text.split('<version>')[1].split('</version>')[0].strip()


def test_one_version_line_in_the_package():
    import re
    import trainit_data
    setup_version = re.search(r"version='([^']+)'", (PKG_ROOT / 'setup.py').read_text()).group(1)
    assert trainit_data.__version__ == setup_version == _version_in('package.xml')
