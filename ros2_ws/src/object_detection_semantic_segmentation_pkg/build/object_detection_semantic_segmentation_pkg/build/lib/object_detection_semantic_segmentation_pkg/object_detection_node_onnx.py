#!/usr/bin/env python3
import sys
import sqlite3
import yaml
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np
import os
from ament_index_python.packages import get_package_share_directory
import onnxruntime as ort

class ObjectDetectionNode(Node):
    def __init__(self):
        super().__init__('object_detection_node')
        self.bridge = CvBridge()
        self.load_models()

    def load_models(self):
        package_share_directory = get_package_share_directory('object_detection_semantic_segmentation_pkg')
        segmentation_model_path = os.path.join(package_share_directory, 'models', 'best_deeplabv3plus_model.onnx')
        detection_model_path = os.path.join(package_share_directory, 'models', 'best_YOLO11x_model.onnx')

        if not os.path.exists(segmentation_model_path):
            self.get_logger().error(f"Segmentation model file does not exist at {segmentation_model_path}")
            sys.exit(1)

        if not os.path.exists(detection_model_path):
            self.get_logger().error(f"Detection model file does not exist at {detection_model_path}")
            sys.exit(1)

        self.segmentation_session = ort.InferenceSession(segmentation_model_path)
        self.detection_session = ort.InferenceSession(detection_model_path)

        self.get_logger().info("ONNX models loaded successfully.")
    
    def preprocess_image(self, cv_image, model_input_shape):
        preprocessed = cv2.resize(cv_image, (model_input_shape[3], model_input_shape[2]))
        preprocessed = preprocessed.astype(np.float32) / 255.0
        preprocessed = preprocessed.transpose((2, 0, 1))
        preprocessed = np.expand_dims(preprocessed, axis=0)
        return preprocessed

    def run_segmentation(self, input_image):
        inputs = {self.segmentation_session.get_inputs()[0].name: input_image}
        seg_output = self.segmentation_session.run(None, inputs)[0]
        seg_prediction = np.argmax(seg_output, axis=1)[0].astype(np.uint8)
        return seg_prediction

    def run_detection(self, input_image, original_image_shape):
        inputs = {self.detection_session.get_inputs()[0].name: input_image}
        detection_output = self.detection_session.run(None, inputs)[0]
        detection_results = self.postprocess_detections(detection_output, original_image_shape)
        return detection_results

    def postprocess_detections(self, output, original_image_shape):
        detections = []
        num_detections = int(output[0])
        for i in range(num_detections):
            index = 1 + i * 7
            class_id = int(output[index + 1])
            confidence = float(output[index + 2])
            x1 = int(output[index + 3] * original_image_shape[1])
            y1 = int(output[index + 4] * original_image_shape[0])
            x2 = int(output[index + 5] * original_image_shape[1])
            y2 = int(output[index + 6] * original_image_shape[0])
            detections.append({
                'class_id': class_id,
                'confidence': confidence,
                'bbox': [x1, y1, x2, y2]
            })
        return detections

    def visualize_results(self, cv_image, seg_mask, detection_results):
        seg_mask_resized = cv2.resize(seg_mask, (cv_image.shape[1], cv_image.shape[0]), interpolation=cv2.INTER_NEAREST)
        seg_rgb = self.decode_segmap(seg_mask_resized)
        overlay = cv2.addWeighted(cv_image, 0.6, seg_rgb, 0.4, 0)

        for det in detection_results:
            x1, y1, x2, y2 = det['bbox']
            class_id = det['class_id']
            confidence = det['confidence']
            cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(overlay, f"{class_id}: {confidence:.2f}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        cv2.imshow("Object Detection & Segmentation", overlay)
        cv2.waitKey(1)

    def decode_segmap(self, mask):
        label_colors = np.array([
            (0, 0, 0),          # Background - Black
            (128, 128, 128),    # Ground - Gray
            (255, 255, 255)     # Pallet - White
        ])

        r = np.zeros_like(mask).astype(np.uint8)
        g = np.zeros_like(mask).astype(np.uint8)
        b = np.zeros_like(mask).astype(np.uint8)

        for l in range(len(label_colors)):
            idx = mask == l
            r[idx], g[idx], b[idx] = label_colors[l]

        seg_rgb = np.stack([r, g, b], axis=2)
        return seg_rgb

    def read_and_process_messages(self, db_path, metadata_path):
        with open(metadata_path, 'r') as f:
            metadata = yaml.safe_load(f)

        topic_dict = {}
        for idx, topic in enumerate(metadata['rosbag2_bagfile_information']['topics_with_message_count'], start=1):
            topic_metadata = topic['topic_metadata']
            name = topic_metadata['name']
            type_ = topic_metadata['type']
            topic_dict[idx] = {'name': name, 'type': type_}

        self.get_logger().info("Topics found in metadata:")
        for tid, info in topic_dict.items():
            self.get_logger().info(f"Topic ID: {tid}, Name: {info['name']}, Type: {info['type']}")

        image_topic_id = None
        target_image_topic = '/robot1/zed2i/left/image_rect_color'  # Adjust if necessary
        for tid, info in topic_dict.items():
            if info['name'] == target_image_topic and info['type'] == 'sensor_msgs/msg/Image':
                image_topic_id = tid
                break

        if image_topic_id is None:
            self.get_logger().error(f"Image topic '{target_image_topic}' not found in the metadata.")
            return

        self.get_logger().info(f"Image topic ID: {image_topic_id}")

        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        cursor.execute("SELECT data FROM messages WHERE topic_id=?", (image_topic_id,))
        messages = cursor.fetchall()

        self.get_logger().info(f"Number of image messages to process: {len(messages)}")

        for idx, row in enumerate(messages, start=1):
            data = row[0]
            try:
                image_msg = deserialize_message(data, Image)
                cv_image = self.bridge.imgmsg_to_cv2(image_msg, desired_encoding='bgr8')
                self.process_image(cv_image)

                if idx % 10 == 0:
                    self.get_logger().info(f"Processed {idx}/{len(messages)} images.")

            except Exception as e:
                self.get_logger().error(f"Failed to process message ID {idx}: {e}")

        conn.close()
        self.get_logger().info("Completed processing all image messages.")

    def destroy_node(self):
        cv2.destroyAllWindows()
        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    node = ObjectDetectionNode()
    try:
        package_share_directory = get_package_share_directory('object_detection_semantic_segmentation_pkg')
        ros2_ws_directory = os.path.abspath(os.path.join(package_share_directory, '..', '..'))
        bags_dir = os.path.join(ros2_ws_directory, 'bags')

        db_path = os.path.join(bags_dir, 'internship_assignment_sample_bag_0.db3')
        metadata_path = os.path.join(bags_dir, 'metadata.yaml')

        if not os.path.exists(db_path):
            node.get_logger().error(f"Database file does not exist at {db_path}")
            sys.exit(1)
        if not os.path.exists(metadata_path):
            node.get_logger().error(f"Metadata file does not exist at {metadata_path}")
            sys.exit(1)

        node.get_logger().info("Starting to read and process messages from the bag file.")
        node.read_and_process_messages(db_path, metadata_path)
    except KeyboardInterrupt:
        node.get_logger().info("Keyboard interrupt received. Shutting down.")
    except Exception as e:
        node.get_logger().error(f"An unexpected error occurred: {e}")
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()

