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
            'display_color_from_speed = pd_flow_visual.display_color_from_speed:main',
            'display_combined_image = pd_flow_visual.display_combined_image:main',
            'cv2_visual = pd_flow_visual.cv2_visual:main',
            'flow_field_publisher = pd_flow_visual.flow_field_publisher:main'
        ],
    },
)
