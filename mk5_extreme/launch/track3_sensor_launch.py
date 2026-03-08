# 자율 - 트랙3 센서부

from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from ament_index_python.packages import get_package_share_directory
import os
def generate_launch_description():
    return LaunchDescription([
    

# ------------------------------------ CAMERA ------------------------------------
#camera
        Node(
            package='mk5_cam',
            executable='web_switch',
            name='web_switch'
        ),
        
      #camera
        # Node(
        #     package='mk5_cam',
        #     executable='web_handle',
        #     name='web_handle'
        # ),

        #Depth camera -> handle + switch
        Node(
            package='mk5_cam',
            executable='com_fix',
            name='com_fix'
        ),




#multi_pub
        Node(
            package='multi_image_publisher_pkg',
            executable='multi_image_publisher_3',
            name='multi_image_publisher_3'
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
        ),       

        Node(
            package='lidar_distance',
            executable='lidar_processing',
            name='lidar_processing'
        ),    
        
# ------------------------------------ Manipulator ------------------------------------        

        Node(
            package='motor_control',
            executable='track3_mani_once',
            name='track3_mani_once'
        ),         
  
        
    ])

