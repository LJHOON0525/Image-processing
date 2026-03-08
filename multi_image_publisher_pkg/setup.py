from setuptools import find_packages, setup

package_name = 'multi_image_publisher_pkg'

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
        'multi_image_publisher_summer = multi_image_publisher_pkg.multi_image_publisher_summer:main',
        'multi_image_publisher_spring = multi_image_publisher_pkg.multi_image_publisher_spring:main',
        'multi_image_publisher_autumn = multi_image_publisher_pkg.multi_image_publisher_autumn:main',
        'multi_image_publisher_winter = multi_image_publisher_pkg.multi_image_publisher_winter:main',
        'multi_image_publisher_1and2 = multi_image_publisher_pkg.multi_image_publisher_self_controll:main',
        'multi_image_publisher_grapefruit = multi_image_publisher_pkg.multi_image_publisher_grapefruit:main',
        'three_view_compressed_streamer = multi_image_publisher_pkg.three_view_compressed_streamer:main',
         # 실행명령 = 패키지.파이썬파일:main
        'multi_image_publisher_4 = multi_image_publisher_pkg.multi_image_publisher_4:main',
        'multi_image_publisher_3 = multi_image_publisher_pkg.multi_image_publisher_3:main',
        'multi_image_publisher_3_self = multi_image_publisher_pkg.multi_image_publisher_3_self:main',
    ],
    },
)
