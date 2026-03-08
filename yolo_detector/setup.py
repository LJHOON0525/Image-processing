from setuptools import find_packages, setup

package_name = 'yolo_detector'

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
        'yolo_test = yolo_detector.yolo_test:main',
        'yolo_person = yolo_detector.yolo_person:main',
        'yolo_firststep = yolo_detector.yolo_firststep:main',
        'yolo_secondstep = yolo_detector.yolo_secondstep:main',
        'yolo_thirdstep = yolo_detector.yolo_thirdstep:main',
        'yolo_forthstep = yolo_detector.yolo_forthstep:main',
        'yolo_boxdet = yolo_detector.yolo_boxdet:main',
        'yolo_army = yolo_detector.yolo_army:main',
        'realline = yolo_detector.realline:main',
        'yolo_stair = yolo_detector.yolo_stair:main',
        'yolo_sos = yolo_detector.yolo_sos:main',
        'yolo_handle = yolo_detector.yolo_handle:main',
        ],
    },
)
