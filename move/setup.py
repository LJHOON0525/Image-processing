from setuptools import find_packages, setup

package_name = 'move'

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
    maintainer='ljh',
    maintainer_email='ljh@todo.todo',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'imu_node = move.imu_node:main',
            'lane_imu = move.lane_imu:main',
            'center_joy_drive = move.center_joy_drive:main',
            'odrive = move.odrive:main',
            'move2 = move.move2:main',
            'move3 = move.move3:main',
            'move4 = move.move4:main',
            'move1test = move.move1test:main',
            'move1 = move.move1:main',
            'move5 = move.move5:main',
            'move2_2 = move.move2_2:main',
            'move6 = move.move6:main',
            'move7 = move.move7:main',
            'move8 = move.move8:main',
            'move9 = move.move9:main',
            'move10 = move.move10:main',
        ],
    },
)

