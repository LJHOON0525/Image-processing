from setuptools import find_packages, setup

package_name = 'mk5_extreme'


launch_files = [
    'launch/joy_track1.launch.py',
    'launch/joy_track2.launch.py',
    'launch/joy_track3.launch.py',
    'launch/joy_track4.launch.py',
    'launch/track1_sensor_launch.py',
    'launch/track2_sensor_launch.py',
    'launch/track3_sensor_launch.py',
    'launch/track1_tracking_launch.py',
    'launch/track2_tracking_launch.py',
    'launch/track3_tracking_launch.py',
]

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', launch_files), 
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='ang',
    maintainer_email='ang@todo.todo',
    description='TODO: Package description',
    license='TODO: License declaration',
    
    entry_points={
        'console_scripts': [
        ],
    },
)
