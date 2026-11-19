from setuptools import find_packages, setup

package_name = 'trainit_data_ros'

setup(
    name=package_name,
    version='0.10.2',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/config', ['config/mock_rgbd.recording.yaml',
                                               'config/realsense_d455_standalone.recording.yaml']),
        ('share/' + package_name + '/launch', ['launch/recorder.launch.py']),
        # D0.8: the exported recording vocabulary a configuration tool pins (ADR-0021)
        ('share/' + package_name + '/catalog', ['catalog/recording_vocabulary.yaml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='fra',
    maintainer_email='ros.master.ai@gmail.com',
    description='TrainIt Data Foundation: the live ROS recorder over the camera contract (ADR-0029).',
    license='Apache-2.0',
    tests_require=['pytest'],
    # console_scripts, never install(PROGRAMS) files: entry points are generated executable.
    entry_points={'console_scripts': [
        'recorder = trainit_data_ros.recorder_node:main',
        'replay = trainit_data_ros.replay_node:main',
    ]},
)
