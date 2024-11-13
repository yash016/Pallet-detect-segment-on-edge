from setuptools import setup, find_packages
from glob import glob
import os

package_name = 'object_detection_semantic_segmentation_pkg'

setup(
    name=package_name,
    version='0.0.0',
    packages=[],
    data_files=[
        ('share/ament_index/resource_index/packages',
         ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
	(os.path.join('share', package_name, 'models'), glob('models/*')),
	(os.path.join('share', package_name, 'videos'), glob('videos/*')),
    ],
    install_requires=[
        'setuptools',
        'torch',
        'torchvision',
        'opencv-python',
        'ultralytics',
        'segmentation-models-pytorch',
        'numpy',
    ],
    zip_safe=True,
    maintainer='root',
    maintainer_email='root@todo.todo',
    description='ROS2 package for object detection and segmentation',
    license='Apache License 2.0',
    tests_require=['pytest'],
    scripts=[
        'scripts/object_detection_node.py'
    ],
)


