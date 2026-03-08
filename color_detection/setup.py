from setuptools import setup

package_name = 'color_detection'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='ljh',
    maintainer_email='your_email@example.com',
    description='A ROS 2 package for detecting yellow color in video streams',
    license='Apache License 2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'color_detection = color_detection.color_detection:main',
            'capture = color_detection.capture:main',
            'realsensetest = color_detection.realsensetest:main',
            'realdepthdata = color_detection.realdepthdata:main',
            'realcolordata = color_detection.realcolordata:main',
            'realinfaredata = color_detection.realinfaredata:main',
            'realalign = color_detection.realalign:main',
            'realalign2 = color_detection.realalign2:main',
            'realalign3 = color_detection.realalign3:main',
            'realnear = color_detection.realnear:main',
            'realfar = color_detection.realfar:main',
            'realdistance = color_detection.realdistance:main',
            'realmulti = color_detection.realmulti:main',
            'clipping = color_detection.clipping:main',
            'hole_filling = color_detection.hole_filling:main',
            'spatial_filter = color_detection.spatial_filter:main',
            'temporal_filter = color_detection.temporal_filter:main',
            'realnearblue = color_detection.realnearblue:main',
            'realsenselanedet = color_detection.realsenselanedet:main',
            'realsense_land = color_detection.realsense_land:main',
            'realsensestair = color_detection.realsensestair:main',
            'realsense_rtabmap = color_detection.realsense_rtabmap:main',
            'realsensewatch = color_detection.realsensewatch:main',
        ],
    },
)

