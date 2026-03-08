from setuptools import find_packages, setup

package_name = 'image_process'

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
            'image_capture = image_process.image_capture:main',
            'image_processing = image_process.image_processing:main',
            'image_roicap = image_process.image_roicap:main',
            'image_roisub = image_process.image_roisub:main',
            'image_roiextraction = image_process.image_roiextraction:main',
            'image_roiexcolor = image_process.image_roiexcolor:main',
            'image_morphocap = image_process.image_morphocap:main',
            'image_morphoerode = image_process.image_morphoerode:main',
            'image_morphodilate = image_process.image_morphodilate:main',
            'image_morphoopen = image_process.image_morphoopen:main',
            'image_morphoclose = image_process.image_morphoclose:main',
            'image_morphogradi = image_process.image_morphogradi:main',
            'image_morphotophat = image_process.image_morphotophat:main',
            'image_morphoblackhat = image_process.image_morphoblackhat:main',
            'image_morphotbhat = image_process.image_morphotbhat:main',
            'image_ocr = image_process.image_ocr:main',
            'image_ocroi = image_process.image_ocroi:main',
            'image_ocroimouse = image_process.image_ocroimouse:main',
            'image_ocroimousenum = image_process.image_ocroimousenum:main',
            'image_arucodet = image_process.image_arucodet:main',
            'image_arucomake = image_process.image_arucomake:main',
            'image_gukarucodet = image_process.image_gukarucodet:main',
        ],
    },
)
