from setuptools import find_packages, setup

package_name = 'motor_control'

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
    maintainer='angmeja',
    maintainer_email='angmeja@todo.todo',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
        ############## joy stick control ############
        'joy_pub_arm= motor_control.joy_pub_arm:main', 
        'joy_dynamix = motor_control.joy_dynamix:main', 
        'joy_nuri = motor_control.joy_nuri:main', 
        'joy_test_nuri = motor_control.joy_test_nuri:main', 
        'track3_mani = motor_control.track3_mani:main', 
        'track3_mani_once = motor_control.track3_mani_once:main', 
        
        
        
        ],
    },
)
