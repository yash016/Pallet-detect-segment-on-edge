import sys
if sys.prefix == '/usr':
    sys.real_prefix = sys.prefix
    sys.prefix = sys.exec_prefix = '/ros2_ws/src/object_detection_semantic_segmentation_pkg/install/object_detection_semantic_segmentation_pkg'
