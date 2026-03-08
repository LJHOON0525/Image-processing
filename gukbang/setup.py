from setuptools import find_packages, setup

package_name = 'gukbang'

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
    maintainer_email='ljh0525333@gmail.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
        'ocr = gukbang.ocr:main',
        'boxflag = gukbang.boxflag:main',
        'signflag = gukbang.signflag:main',
        'signflag2 = gukbang.signflag2:main',
        'smoke = gukbang.smoke:main',
        'boxdet = gukbang.boxdet:main',
        'summer = gukbang.summer:main',
        'webthreecam = gukbang.webthreecam:main',
        'realsenseonly = gukbang.realsenseonly:main',
        'webcamgrid = gukbang.webcamgrid:main',
        'jamong = gukbang.jamong:main',
        ],
    },
)
