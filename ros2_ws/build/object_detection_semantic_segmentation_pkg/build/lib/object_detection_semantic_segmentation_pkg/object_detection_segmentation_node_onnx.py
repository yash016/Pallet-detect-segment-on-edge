import os
import sys
import sqlite3
import numpy as np
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image
from rclpy.serialization import deserialize_message
from rclpy.node import Node
from ament_index_python.packages import get_package_share_directory
import onnxruntime as ort
import rclpy
import cv2


class ObjectDetectionNode(Node):
    def __init__(self):
        super().__init__('object_detection_node')
        self.bridge = CvBridge()
        self.segmentation_input_size = (512, 512)
        self.detection_input_size = (640, 640)
        self.output_dir = "/ros2_ws/output_images"
        os.makedirs(self.output_dir, exist_ok=True)
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

    def preprocess_image(self, cv_image, model_input_size):
        self.get_logger().info(f"Resizing image to {model_input_size}")
        resized_image = cv2.resize(cv_image, model_input_size)
        normalized_image = resized_image.astype(np.float32) / 255.0
        preprocessed = np.transpose(normalized_image, (2, 0, 1))  # Channels first
        preprocessed = np.expand_dims(preprocessed, axis=0)  # Add batch dimension
        return preprocessed

    def postprocess_detections(self, output, original_image_shape):
        self.get_logger().info(f"Post-processing Detection Output Shape: {output.shape}")
        output = np.squeeze(output)  # Remove batch dimension
        self.get_logger().info(f"Detection Output Shape after squeezing: {output.shape}")

        # Correctly reshape the output
        if output.shape[0] == 6:
            output = output.T  # Now shape is (8400, 6)
            self.get_logger().info(f"Detection Output Shape after transpose: {output.shape}")
        elif output.shape[1] == 6:
            self.get_logger().info(f"Detection Output Shape is (num_predictions, 6): {output.shape}")
        else:
            self.get_logger().error(f"Unexpected output shape after squeezing: {output.shape}")
            return []

        detections = []
        confidence_threshold = 0.1  # Adjust as needed

        for idx, det in enumerate(output, start=1):
            try:
                # Ensure det is a 1D array
                if det.ndim != 1:
                    self.get_logger().error(f"Detection {idx} has incorrect dimensions: {det.shape}")
                    continue

                # Extract detection attributes
                x_center = float(det[0])
                y_center = float(det[1])
                width = float(det[2])
                height = float(det[3])
                confidence = float(det[4])
                class_probs = det[5]  # Correct extraction

                # Log the detection details
                self.get_logger().debug(f"Detection {idx}: x_center={x_center}, y_center={y_center}, "
                                        f"width={width}, height={height}, confidence={confidence}, "
                                        f"class_probs={class_probs}, type={type(class_probs)}")

                # Check if class_probs is iterable
                if not isinstance(class_probs, (list, np.ndarray)):
                    self.get_logger().error(f"Detection {idx}: class_probs is not a list or ndarray.")
                    continue

                # Convert class_probs to a NumPy array if it's a list
                if isinstance(class_probs, list):
                    class_probs = np.array(class_probs, dtype=np.float32)

                # If class_probs is multi-dimensional, flatten it
                if class_probs.ndim > 1:
                    class_probs = class_probs.flatten()

                # Ensure class_probs is a 1D array
                if class_probs.ndim != 1:
                    self.get_logger().error(f"Detection {idx}: class_probs has invalid shape: {class_probs.shape}")
                    continue

                # Skip detections below the confidence threshold
                if confidence < confidence_threshold:
                    continue  # Skip low-confidence detections

                # Find the class with the highest probability
                class_id = int(np.argmax(class_probs))
                class_confidence = float(class_probs[class_id])

                # Scale coordinates to original image size
                x_scale = original_image_shape[1] / self.detection_input_size[0]
                y_scale = original_image_shape[0] / self.detection_input_size[1]

                x1 = int((x_center - width / 2) * x_scale)
                y1 = int((y_center - height / 2) * y_scale)
                x2 = int((x_center + width / 2) * x_scale)
                y2 = int((y_center + height / 2) * y_scale)

                detections.append({
                    'class_id': class_id,
                    'confidence': class_confidence,
                    'bbox': [x1, y1, x2, y2]
                })

            except Exception as e:
                self.get_logger().error(f"Error processing detection {idx}: {e}")
                continue  # Skip problematic detections

        # Apply Non-Maximum Suppression (NMS)
        detections = self.non_max_suppression(detections)
        self.get_logger().info(f"Total Valid Detections after NMS: {len(detections)}")
        return detections

    def non_max_suppression(self, detections, iou_threshold=0.5):
        if not detections:
            return []

        # Sort detections by confidence score descending
        detections = sorted(detections, key=lambda x: x['confidence'], reverse=True)
        suppressed = [False] * len(detections)
        results = []

        for i in range(len(detections)):
            if suppressed[i]:
                continue
            det_i = detections[i]
            results.append(det_i)
            for j in range(i + 1, len(detections)):
                if suppressed[j]:
                    continue
                det_j = detections[j]
                iou = self.compute_iou(det_i['bbox'], det_j['bbox'])
                if iou > iou_threshold:
                    suppressed[j] = True
        return results

    def compute_iou(self, boxA, boxB):
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])
        interArea = max(0, xB - xA) * max(0, yB - yA)
        boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
        boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
        if boxAArea + boxBArea - interArea == 0:
            return 0.0
        return interArea / float(boxAArea + boxBArea - interArea)

    def process_image(self, cv_image, message_id):
        self.get_logger().info(f"Processing image {message_id}")
        detection_input = self.preprocess_image(cv_image, self.detection_input_size)
        input_name = self.detection_session.get_inputs()[0].name
        detection_output = self.detection_session.run(None, {input_name: detection_input})[0]

        detections = self.postprocess_detections(detection_output, cv_image.shape)
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            confidence = det['confidence']
            class_id = det['class_id']
            cv2.rectangle(cv_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(cv_image, f"ID:{class_id} {confidence:.2f}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        output_path = os.path.join(self.output_dir, f"prediction_{message_id}.png")
        cv2.imwrite(output_path, cv_image)
        self.get_logger().info(f"Saved prediction image to {output_path}")

    def read_and_process_messages(self, db_path):
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        # Fetch topics from the database
        cursor.execute("SELECT id, name, type FROM topics")
        topics_in_db = cursor.fetchall()
        self.get_logger().info("Topics in the database:")
        for topic in topics_in_db:
            self.get_logger().info(f"Topic ID: {topic[0]}, Name: {topic[1]}, Type: {topic[2]}")

        # Find the topic ID for the image topic
        image_topic_id = None
        target_image_topic = '/robot1/zed2i/left/image_rect_color'
        for topic in topics_in_db:
            tid, name, type_ = topic
            if name == target_image_topic and type_ == 'sensor_msgs/msg/Image':
                image_topic_id = tid
                break

        if image_topic_id is None:
            self.get_logger().error(f"Image topic '{target_image_topic}' not found in the database.")
            return

        self.get_logger().info(f"Image topic ID: {image_topic_id}")

        # Fetch messages with the correct topic ID
        cursor.execute("SELECT data FROM messages WHERE topic_id=?", (image_topic_id,))
        messages = cursor.fetchall()

        self.get_logger().info(f"Number of image messages to process: {len(messages)}")

        for idx, row in enumerate(messages, start=1):
            data = row[0]
            try:
                image_msg = deserialize_message(data, Image)
                self.get_logger().info(f"Message {idx} encoding: {image_msg.encoding}")

                cv_image = self.bridge.imgmsg_to_cv2(image_msg, desired_encoding='bgr8')

                if cv_image.size == 0:
                    self.get_logger().error(f"Empty image data for message {idx}.")
                    continue

                self.process_image(cv_image, idx)

                if idx % 10 == 0:
                    self.get_logger().info(f"Processed {idx}/{len(messages)} images.")

            except CvBridgeError as e:
                self.get_logger().error(f"CvBridgeError for message ID {idx}: {e}")
            except Exception as e:
                self.get_logger().error(f"Failed to process message ID {idx}: {e}")

        conn.close()
        self.get_logger().info("Completed processing all image messages.")

    def destroy_node(self):
        super().destroy_node()
        cv2.destroyAllWindows()


def main(args=None):
    rclpy.init(args=args)
    node = ObjectDetectionNode()
    try:
        db_path = "/ros2_ws/bags/internship_assignment_sample_bag_0.db3"
        node.read_and_process_messages(db_path)
    except KeyboardInterrupt:
        node.get_logger().info("Keyboard interrupt detected, shutting down.")
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
