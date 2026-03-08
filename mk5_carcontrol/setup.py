from setuptools import find_packages, setup

package_name = 'mk5_carcontrol'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='ang',
    maintainer_email='ang@todo.todo',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
        'autodrive = mk5_carcontrol.autodrive:main',

        #------------ 알고리즘 기반 미션별 주행------------
        'tracking_1 = mk5_carcontrol.tracking_1:main',
        'tracking_2 = mk5_carcontrol.tracking_2:main',
        'tracking_3 = mk5_carcontrol.tracking_3:main',
        
        #------------ 라이다 기반 주행------------
        'tracking_rightwall = mk5_carcontrol.tracking_rightwall:main',

        'path_follower = mk5_carcontrol.path_follower:main',
        'car_track3 = mk5_carcontrol.car_track3:main',

        #------------ 최종 사용 노드 ------------
        #트랙1
        'car_cmd = mk5_carcontrol.car_cmd:main',
        #트랙2
        'car_cmd_track2 = mk5_carcontrol.car_cmd_track2:main',
        'flipcontrol = mk5_carcontrol.flipcontrol:main',
        
        'car_cmd_track2_timer = mk5_carcontrol.car_cmd_track2_timer:main',
        #트랙3
        'car_cmd_track3 = mk5_carcontrol.car_cmd_track3:main',


        #------------ 원격 주행------------
        'joy_control = mk5_carcontrol.joy_control:main',
        'joy_odrive = mk5_carcontrol.joy_odrive:main',          
        
        ],
    },
)
