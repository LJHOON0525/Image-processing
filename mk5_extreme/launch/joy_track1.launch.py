# JOY - TRACK1

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
        Node(
            package='mk5_cam',
            executable='webtwo',
            name='webtwo'
        ),

    
# ------------------------------------ ALL VIEWER ------------------------------------
        
       #multi_pub
        Node(
            package='multi_image_publisher_pkg',
            executable='multi_image_publisher_1and2',
            name='multi_image_publisher_1and2'
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
        


        
    ])

