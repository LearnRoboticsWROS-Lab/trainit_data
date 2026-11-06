"""The INSTALLED Python metadata of both packages is this source's: one version, every console script.

colcon (setuptools) leaves the egg-info of an older version beside the new one when the version changes, and Python
may resolve the OLD metadata (2026-10-30: the workspace install resolved trainit_data_ros 0.3.0 beside 0.4.0; the
hardware layer's new console script failed at start the same way). Fix: delete the stale
``install/<package>/lib/python3.X/site-packages/<package>-<old>-py3.X.egg-info``.
"""
import ast
import configparser
import re
import sys
from importlib.metadata import distribution
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
PACKAGES = ('trainit_data', 'trainit_data_ros')


def _source_version(package):
    return re.search(r'<version>\s*([^<\s]+)\s*</version>', (REPO_ROOT / package / 'package.xml').read_text()).group(1)


def _source_console_scripts(package):
    """The console-script names setup.py declares (read as Python, so a comment never shortens the list)."""
    for node in ast.walk(ast.parse((REPO_ROOT / package / 'setup.py').read_text())):
        if isinstance(node, ast.keyword) and node.arg == 'entry_points':
            return sorted(entry.split('=')[0].strip() for entry in ast.literal_eval(node.value)['console_scripts'])
    return []


def _site_packages(package):
    ament = pytest.importorskip('ament_index_python.packages')
    try:
        prefix = Path(ament.get_package_prefix(package))
    except Exception:  # noqa: BLE001 -- not installed in this environment
        pytest.skip(f'{package} is not installed in the sourced workspace')
    return prefix / 'lib' / f'python{sys.version_info.major}.{sys.version_info.minor}' / 'site-packages'


@pytest.mark.parametrize('package', PACKAGES)
def test_the_install_holds_exactly_this_versions_metadata_with_every_console_script(package):
    version, site = _source_version(package), _site_packages(package)
    infos = sorted(p.name for p in site.glob(f'{package}-*.egg-info'))
    if not infos:
        pytest.skip('a symlink (develop) install keeps its metadata in the build tree')
    py = f'py{sys.version_info.major}.{sys.version_info.minor}'
    assert infos == [f'{package}-{version}-{py}.egg-info'], \
        f'stale metadata in {site}: {infos} (keep only {version}; see this module\'s docstring)'
    entry_points = configparser.ConfigParser()
    entry_points.optionxform = str                      # keep the names' case
    entry_points.read(site / infos[0] / 'entry_points.txt')
    assert sorted(entry_points['console_scripts']) == _source_console_scripts(package)


@pytest.mark.parametrize('package', PACKAGES)
def test_python_resolves_each_installed_distribution_to_this_version(package):
    _site_packages(package)
    resolved = distribution(package.replace('_', '-'))
    assert resolved.version == _source_version(package)
    assert sorted(e.name for e in resolved.entry_points if e.group == 'console_scripts') == \
        _source_console_scripts(package)
