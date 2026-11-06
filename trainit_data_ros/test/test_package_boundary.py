"""Architecture guards of the ROS package (ADR-0029 Decisions 1 and 6).

ALLOW-lists, not deny-lists: every declared dependency and every imported top-level module must be one
the ADR permits. A cell, example or Pro package is therefore refused whatever its name, and no private
name needs to be written here (owner decision 2026-10-28 on guard markers).
"""
import ast
import re
import sys
from pathlib import Path

PKG_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PKG_ROOT.parent

# ADR-0029 Decision 1: trainit_data + trainit_hardware_layer (+ trainit_interfaces from D0.5) + standard ROS.
ALLOWED_DEPENDENCIES = {
    'trainit_data', 'trainit_hardware_layer', 'trainit_interfaces',
    'rclpy', 'sensor_msgs', 'std_msgs', 'std_srvs', 'diagnostic_msgs', 'tf2_ros', 'geometry_msgs',
    'builtin_interfaces',
    'launch', 'launch_ros', 'launch_testing', 'launch_testing_ros', 'ament_index_python',
    'python3-numpy', 'python3-yaml', 'xacro', 'python3-pytest', 'ament_copyright', 'ament_flake8', 'ament_pep257',
}
ALLOWED_IMPORTS = set(sys.stdlib_module_names) | {
    'trainit_data', 'trainit_data_ros', 'numpy', 'rclpy', 'sensor_msgs', 'std_msgs', 'std_srvs', 'diagnostic_msgs',
    'tf2_ros',
    'geometry_msgs', 'builtin_interfaces', 'launch', 'launch_ros', 'launch_testing', 'ament_index_python', 'pytest',
    'rosgraph_msgs',
    'setuptools', 'yaml', 'tf2_ros', 'trainit_hardware_layer', 'trainit_interfaces', 'xacro',
}


LOCAL_TEST_MODULES = {p.stem for p in (PKG_ROOT / 'test').glob('*.py')}


def _version(xml: Path) -> str:
    return re.search(r'<version>\s*([^<\s]+)\s*</version>', xml.read_text()).group(1)


def test_the_package_declares_only_its_allowed_dependencies():
    text = (PKG_ROOT / 'package.xml').read_text()
    deps = set(re.findall(r'<\w*depend[^>]*>\s*([\w-]+)\s*</\w*depend>', text))
    assert not deps - ALLOWED_DEPENDENCIES, sorted(deps - ALLOWED_DEPENDENCIES)
    assert 'trainit_data' in deps


def test_the_package_imports_only_allowed_modules():
    bad = []
    for path in sorted(PKG_ROOT.rglob('*.py')):
        if 'build' in path.relative_to(PKG_ROOT).parts:
            continue
        for node in ast.walk(ast.parse(path.read_text())):
            names = []
            if isinstance(node, ast.Import):
                names = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                names = [node.module]
            bad += [f'{path.relative_to(PKG_ROOT)}:{node.lineno}: {n}' for n in names
                    if n.split('.')[0] not in ALLOWED_IMPORTS and n.split('.')[0] not in LOCAL_TEST_MODULES]
    assert not bad, bad


def test_one_version_line_for_the_repository():
    core = REPO_ROOT / 'trainit_data' / 'package.xml'
    ros = PKG_ROOT / 'package.xml'
    setup_version = re.search(r"version='([^']+)'", (PKG_ROOT / 'setup.py').read_text()).group(1)
    assert _version(core) == _version(ros) == setup_version


def test_the_internal_core_modules_are_used_only_at_the_same_version():
    """trainit_data_ros may import the core's INTERNAL `_strict` / `_canonical` (one repository, one version line,
    ADR-0029 Decision 1); this test makes that exception safe: both packages must be the same release."""
    import trainit_data
    import trainit_data_ros
    assert trainit_data_ros.__version__ == trainit_data.__version__
