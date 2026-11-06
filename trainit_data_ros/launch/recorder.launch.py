"""The D0 recorder in a robot / application slot (ADR-0023): ``namespace`` is the slot, ``config_file`` a
``trainit-recording/1.0`` file. The data store comes from ``TRAINIT_DATA_ROOT`` (default ``~/TrainItData``).

    ros2 launch trainit_data_ros recorder.launch.py config_file:=/path/to/x.recording.yaml namespace:=r1
    ros2 service call /r1/trainit_data_recorder/start_session std_srvs/srv/Trigger
"""
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument('config_file', description='a trainit-recording/1.0 file'),
        DeclareLaunchArgument('namespace', default_value='', description='the robot / application slot'),
        DeclareLaunchArgument('use_sim_time', default_value='false'),
        Node(package='trainit_data_ros', executable='recorder', name='trainit_data_recorder',
             namespace=LaunchConfiguration('namespace'), output='screen',
             parameters=[{'config_file': LaunchConfiguration('config_file'),
                          'use_sim_time': LaunchConfiguration('use_sim_time')}]),
    ])
