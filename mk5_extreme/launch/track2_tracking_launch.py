# 자율 - 트랙2 주행부

from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from ament_index_python.packages import get_package_share_directory
import os

def generate_launch_description():
    return LaunchDescription([
    
# ------------------------------------ FLIP Control ------------------------------------

        Node(
            package='mk5_carcontrol',
            executable='flipcontrol',
            name='flipcontrol'
        ),

# --------------------------- ODrive Control & Control Data Process ---------------------------
        Node(
            package='mk5_carcontrol',
            executable='car_cmd_track2',
            name='car_cmd_track2'
        ),

# ------------------------------------ DWA System------------------------------------
        Node(
            package='wall_follower',
            executable='dwa',
            name='dwa'
        ),
        
# ------------------------------------ Mission Control ------------------------------------
        Node(
            package='mission_control',
            executable='mission_track2',
            name='mission_track2'
        ),        
        
        
    ])

