from setuptools import setup
from glob import glob
import os

setup(
    name='object_detection_semantic_segmentation_pkg',  # Name of the package
    version='0.0.0',  # Package version
    packages=['object_detection_semantic_segmentation_pkg'],  # Python package directory
    data_files=[
        # Metadata and resource files
        ('share/ament_index/resource_index/packages',
            ['resource/object_detection_semantic_segmentation_pkg']),
        ('share/object_detection_semantic_segmentation_pkg', ['package.xml', 'setup.py']),
	('share/object_detection_semantic_segmentation_pkg/bags', glob('bags/*')),
    ],
    install_requires=['setuptools'],  # Required dependencies
    zip_safe=True,  # Package can be installed as a .zip file
    maintainer='root',  # Maintainer's name
    maintainer_email='root@todo.todo',  # Maintainer's email
    description='ROS2 package for object detection and semantic segmentation',  # Package description
    license='Apache License 2.0',  # License for the package
    entry_points={
        # Define the console scripts
        'console_scripts': [
            'object_detection_node_onnx = object_detection_semantic_segmentation_pkg.object_detection_node_onnx:main',
        ],
    },
)
