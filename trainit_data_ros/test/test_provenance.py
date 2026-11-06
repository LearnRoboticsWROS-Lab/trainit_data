"""Description / scene provenance from installed artifacts (xacro closure), with an injected share lookup."""
import pytest

pytest.importorskip('xacro', reason='needs ROS 2 xacro')

from trainit_data.errors import INVALID_VALUE, DataError  # noqa: E402
from trainit_data_ros.provenance import description_provenance, package_relative, scene_provenance  # noqa: E402

ROOT = """<?xml version="1.0"?>
<robot xmlns:xacro="http://www.ros.org/wiki/xacro" name="example_cell">
  <xacro:arg name="mount_z" default="0.5"/>
  <xacro:include filename="parts/camera.xacro"/>
  <link name="base_link"/>
  <xacro:camera_mount z="$(arg mount_z)"/>
</robot>
"""
CAMERA = """<?xml version="1.0"?>
<robot xmlns:xacro="http://www.ros.org/wiki/xacro">
  <xacro:macro name="camera_mount" params="z">
    <link name="camera_link"/>
    <joint name="camera_joint" type="fixed"><parent link="base_link"/><child link="camera_link"/>
      <origin xyz="0 0 ${z}" rpy="0 0 0"/></joint>
  </xacro:macro>
</robot>
"""


@pytest.fixture
def share(tmp_path):
    root = tmp_path / 'install' / 'share' / 'example_description'
    (root / 'urdf' / 'parts').mkdir(parents=True)
    (root / 'urdf' / 'cell.urdf.xacro').write_text(ROOT)
    (root / 'urdf' / 'parts' / 'camera.xacro').write_text(CAMERA)
    (root / 'scene.usd').write_bytes(b'#usda 1.0\n')
    return lambda package: tmp_path / 'install' / 'share' / package


SPEC = {'package': 'example_description', 'file': 'urdf/cell.urdf.xacro', 'args': {'mount_z': '0.75'}}


def test_the_whole_include_closure_is_hashed_machine_independently(share):
    d = description_provenance(SPEC, share)
    assert [(f['package'], f['file']) for f in d['files']] == [('example_description', 'urdf/cell.urdf.xacro'),
                                                               ('example_description', 'urdf/parts/camera.xacro')]
    assert d['args'] == {'mount_z': '0.75'}                    # the declared args only, never xacro's defaults
    assert d['root'] == {'package': 'example_description', 'file': 'urdf/cell.urdf.xacro'}
    assert d == description_provenance(SPEC, share)              # deterministic


def test_any_included_file_or_argument_changes_the_identity(share):
    before = description_provenance(SPEC, share)
    assert description_provenance(dict(SPEC, args={'mount_z': '0.8'}), share)['sha256'] != before['sha256']
    camera = share('example_description') / 'urdf' / 'parts' / 'camera.xacro'
    camera.write_text(CAMERA.replace('rpy="0 0 0"', 'rpy="0 0 0.1"'))
    after = description_provenance(SPEC, share)
    assert after['sha256'] != before['sha256'] and after['expanded_sha256'] != before['expanded_sha256']


def test_a_missing_or_broken_description_is_refused(share):
    with pytest.raises(DataError) as err:
        description_provenance(dict(SPEC, file='urdf/absent.urdf.xacro'), share)
    assert err.value.code == INVALID_VALUE
    (share('example_description') / 'urdf' / 'parts' / 'camera.xacro').write_text('<robot><broken')
    with pytest.raises(DataError) as err:
        description_provenance(SPEC, share)
    assert 'does not expand' in err.value.message


def test_scene_provenance_is_the_declared_installed_file(share):
    scene = scene_provenance({'package': 'example_description', 'file': 'scene.usd', 'prim': '/World/Camera'}, share)
    assert scene['completeness'] == 'ROOT_LAYER_ONLY' and scene['prim'] == '/World/Camera'
    assert scene['sha256'].startswith('sha256:')


def test_package_relative_names():
    assert package_relative('/opt/x/share/pkg/urdf/a.xacro') == {'package': 'pkg', 'file': 'urdf/a.xacro'}
    assert package_relative('/tmp/loose.xacro') == {'file': 'loose.xacro'}
