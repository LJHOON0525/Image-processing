# 자율 - 트랙3 주행부

from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from ament_index_python.packages import get_package_share_directory
import os

def generate_launch_description():
    return LaunchDescription([
    

        
# ------------------------------------ Mission Control ------------------------------------

        Node(
            package='mission_control',
            executable='mission_track3',
            name='mission_track3'
        ),

# ------------------------------------ Tracking ------------------------------------

        Node(
            package='mk5_carcontrol',
            executable='tracking_3',
            name='tracking_3'
        ),

# ------------------------------------ Autodrive ---------------------------

        Node(
            package='mk5_carcontrol',
            executable='autodrive',
            name='autodrive'
        ),
        
        
        
    ])

