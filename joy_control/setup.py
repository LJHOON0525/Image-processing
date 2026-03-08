from setuptools import find_packages, setup

package_name = 'joy_control'

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
        'joylistener=joy_control.joylistener:main',
        'drone_indicate=joy_control.drone_indicate:main',
        'drone_real=joy_control.drone_real:main',
        'real_imu=joy_control.real_imu:main',
        'real_gyro=joy_control.real_gyro:main',
        'real_balance=joy_control.real_balance:main',
        'real_imu_comple=joy_control.real_imu_comple:main',
        'real_imu_movingaverage=joy_control.real_imu_movingaverage:main',
        'real_low_pass=joy_control.real_low_pass:main',
        'real_kalman=joy_control.real_kalman:main',
        'real_madgwick=joy_control.real_madgwick:main',
        'drone_imu=joy_control.drone_imu:main',
        'test=joy_control.test:main',
        ],
    },
)
