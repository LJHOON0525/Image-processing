from setuptools import find_packages, setup

package_name = 'test_jh'

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
            'aruco = test_jh.aruco:main',
            'aruco2 = test_jh.aruco2:main',
            'aruco3 = test_jh.aruco3:main',
            'lanedetect = test_jh.lanedetect:main',
            'landettest = test_jh.landettest:main',
            'lanedet = test_jh.lanedet:main',
            'reallande = test_jh.reallande:main',
            'lanetest = test_jh.lanetest:main',
            'reallande2 = test_jh.reallande2:main',
            'realanedet3 = test_jh.realanedet3:main',
            'realanedet4 = test_jh.realanedet4:main',
            'realandeyellow = test_jh.realandeyellow:main',
            'yolo_bird = test_jh.yolo_bird:main',
            'realline = test_jh.realline:main',
            'reallineimuview = test_jh.reallineimuview:main',
            'landette = test_jh.landette:main',
            'landete = test_jh.landete:main',
            'landet = test_jh.landet:main',
            'newlandet = test_jh.newlandet:main',
            'powerlande = test_jh.powerlande:main',
            'stairtest = test_jh.stairtest:main',
            'kh = test_jh.kh:main',
            'yolodom = test_jh.yolodom:main',
            
            'slamtest = test_jh.slamtest:main',
            'redfinder = test_jh.redfinder:main',
            'redfinder2 = test_jh.redfinder2:main',
        ],
    },
)

