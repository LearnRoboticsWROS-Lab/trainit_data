import os
from pathlib import Path

from setuptools import find_packages, setup

package_name = 'trainit_data'

# The reference synthetic provider (D0.8): DATA files installed beside the package (a launching tool finds its
# provider.json in share/; the core never imports it). From a source checkout only: a build without the repository's
# examples directory (an sdist) installs the core alone.
PROVIDER = Path(__file__).resolve().parent.parent / 'examples' / 'isaac_replicator_bin'
PROVIDER_FILES = ['provider.json', 'generate.py', 'bin_config.json', 'README.md']
providers = ([('share/' + package_name + '/synthetic_providers/isaac_replicator_bin',
               [os.path.relpath(PROVIDER / name) for name in PROVIDER_FILES])]
             if all((PROVIDER / name).is_file() for name in PROVIDER_FILES) else [])

# Installable both as an ament_python package (colcon) and with plain `pip install`: a training or
# provider environment uses the canonical data contracts without the ROS tree (ADR-0029 Decision 1).
setup(
    name=package_name,
    version='0.10.0',
    packages=find_packages(exclude=['test']),
    package_data={package_name: ['vocab_data/*.json']},
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ] + providers,
    python_requires='>=3.10',
    install_requires=['setuptools', 'numpy', 'Pillow'],
    zip_safe=True,
    maintainer='fra',
    maintainer_email='ros.master.ai@gmail.com',
    description='TrainIt Data Foundation: the canonical, ROS-free data contracts (ADR-0029).',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={'console_scripts': ['trainit-data = trainit_data.cli:main']},
)
