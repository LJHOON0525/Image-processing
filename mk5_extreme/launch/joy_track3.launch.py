# JOY - TRACK3

from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([

# ------------------------------------ CAMERA ------------------------------------

#camera
        Node(
            package='mk5_cam',
            executable='sos',
            name='sos'
        ),
        
      #camera
        # Node(
        #     package='mk5_cam',
        #     executable='web_switch',
        #     name='web_switch'
        # ),
        #camera
        Node(
            package='mk5_cam',
            executable='web_fire',
            name='web_fire'
        ),

    
# ------------------------------------ ALL VIEWER ------------------------------------
#multi_pub => Self Version
        Node(
            package='multi_image_publisher_pkg',
            executable='multi_image_publisher_3_self',
            name='multi_image_publisher_3_self'
        ),


# ------------------------------------ ODrive Control ------------------------------------ 
    #JOY
        Node(
            package='joy',
            executable='joy_node',
            name='joy_node'
        ),
        
    #JOY Data Processing
        Node(
            package='mk5_carcontrol',
            executable='joy_control',
            name='joy_control'
        ),

    #ODrive Control
        Node(
            package='mk5_carcontrol',
            executable='joy_odrive',
            name='joy_odrive'
        ),
        
# ------------------------------------ MANIPULATOR ------------------------------------         
        Node(
            package='motor_control',
            executable='joy_pub_arm',
            name='joy_pub_arm'
        ),
        Node(
            package='motor_control',
            executable='joy_nuri',
            name='joy_nuri'
        ),
        Node(
            package='motor_control',
            executable='joy_dynamix',
            name='joy_dynamix'
        ),
        
    ])

