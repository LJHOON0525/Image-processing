from setuptools import find_packages, setup

package_name = 'coordinate'

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
        'book_parameter = coordinate.book_parameter:main',
        'parameter = coordinate.parameter:main',
        'imu = coordinate.imu:main',
        'gyro = coordinate.gyro:main',
        'complementary_gyro = coordinate.complementary_gyro:main',
        'accel_yaw_gyro = coordinate.accel_yaw_gyro:main',
        'accel_math = coordinate.accel_math:main',
        'coordinatetest = coordinate.coordinatetest:main',
        'joy_img = coordinate.joy_img:main',
        'joy_contorol = coordinate.joy_contorol:main',
        'imu_aceel_hud = coordinate.imu_aceel_hud:main',
        'joyencoder = coordinate.joyencoder:main',
        'joyrobot = coordinate.joyrobot:main',
        ],
    },
)
