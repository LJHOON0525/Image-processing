# 자율 - 트랙1 주행부

from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from ament_index_python.packages import get_package_share_directory
import os
def generate_launch_description():
    return LaunchDescription([
    
# ------------------------------------ DWA System------------------------------------
        Node(
            package='wall_follower',
            executable='dwa',
            name='dwa'
        ),
        

# --------------------------- ODrive Control & Control Data Process ---------------------------
        Node(
            package='mk5_carcontrol',
            executable='car_cmd',
            name='car_cmd'
        ),
        
        
    ])

