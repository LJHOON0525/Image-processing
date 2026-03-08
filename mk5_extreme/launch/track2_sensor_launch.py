# 자율 - 트랙2 센서부

from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from ament_index_python.packages import get_package_share_directory
import os
from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([

# ------------------------------------ LiDAR Data ------------------------------------

        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                os.path.join(
                    get_package_share_directory('rplidar_ros'),
                    'launch',
                    'rplidar_a3_launch.py'
                )
            )
        ),       

# ------------------------------------ IMU Data ------------------------------------
        Node(
            package='imu',
            executable='imu_node',
            name='imu_node'
        ),

        Node(
            package='imu',
            executable='imu_processing',
            name='imu_processing'
        ),        
        
# ------------------------------------ CAMERA Data ------------------------------------



        
    ])

