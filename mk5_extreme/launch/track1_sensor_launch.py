# 자율 - 트랙1 센서부

from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from ament_index_python.packages import get_package_share_directory
import os

def generate_launch_description():
    return LaunchDescription([
    
    
     #camera
        Node(
            package='mk5_cam',
            executable='realsenseonly',
            name='realsenseonly'
        ),
        
      #camera
        Node(
            package='mk5_cam',
            executable='webtwo',
            name='webtwo'
        ),
        
       #multi_pub
        Node(
            package='multi_image_publisher_pkg',
            executable='multi_image_publisher_1and2',
            name='multi_image_publisher_1and2'
        ),
    

# ------------------------------------ LiDAR Data ------------------------------------

        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                os.path.join(
                    get_package_share_directory('rplidar_ros'),
                    'launch',
                    'rplidar_a3_launch.py'
                )
            )
        )       


        
    ])

