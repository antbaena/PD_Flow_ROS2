import os
from glob import glob
from setuptools import find_packages, setup

package_name = 'pd_flow_visual'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name), glob('launch/*launch.[pxy][yma]*')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='ubuntu',
    maintainer_email='euquemada@gmail.com',
    description='TODO: Package description',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'cv2_color_from_speed = pd_flow_visual.cv2_color_from_speed:main',
            'vispy_color_from_speed = pd_flow_visual.vispy_color_from_speed:main',
            'cv2_color_superposition = pd_flow_visual.cv2_color_superposition:main',
            'cv2_rgb_point_cloud = pd_flow_visual.cv2_rgb_point_cloud:main',
            'combined_image = pd_flow_visual.combined_image:main',
            'flow_field_publisher = pd_flow_visual.flow_field_publisher:main'
        ],
    },
)
