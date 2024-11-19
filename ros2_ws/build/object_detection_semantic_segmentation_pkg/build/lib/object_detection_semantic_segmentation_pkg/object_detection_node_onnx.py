import sys
import sqlite3
from rclpy.serialization import deserialize_message
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import cv2
import numpy as np
import os
from ament_index_python.packages import get_package_share_directory
import onnxruntime as ort
import rclpy
from rclpy.node import Node

class ObjectDetectionNode(Node):
    def __init__(self):
        super().__init__('object_detection_node')
        self.bridge = CvBridge()
        self.load_models()
        # Create output directory for saving images
        self.output_dir = os.path.join(os.getcwd(), 'output_images')
        os.makedirs(self.output_dir, exist_ok=True)
        self.image_counter = 0  # Counter for naming saved images

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

    def preprocess_image(self, cv_image, target_size):
        """Resize image to the target size for the model."""
        height, width = target_size
        self.get_logger().info(f"Resizing image to {width}x{height}")
        
        # Resize the image
        preprocessed = cv2.resize(cv_image, (width, height))
        
        # Normalize the image
        preprocessed = preprocessed.astype(np.float32) / 255.0
        
        # Rearrange dimensions
        preprocessed = preprocessed.transpose((2, 0, 1))
        
        # Add batch dimension
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
        
        # Log detection_output details
        self.get_logger().info(f"Detection Output Type: {type(detection_output)}")
        self.get_logger().info(f"Detection Output Shape: {detection_output.shape}")
        self.get_logger().info(f"Detection Output Data Sample: {detection_output[:1]}")  # Log first detection
        
        detection_results = self.postprocess_detections(detection_output, original_image_shape)
        return detection_results

    def postprocess_detections(self, output, original_image_shape):
        self.get_logger().info(f"Post-processing Detection Output Shape: {output.shape}")
        
        detections = []
        # Adjust parsing based on the actual output format
        # For instance, if the output shape is (num_detections, attributes)
        for idx, det in enumerate(output, start=1):
            self.get_logger().info(f"Processing Detection {idx}: {det}")
            
            try:
                # Ensure det is a 1D array with expected number of elements
                if isinstance(det, np.ndarray):
                    det = det.tolist()
                
                if len(det) < 6:
                    self.get_logger().warn(f"Detection {idx} has insufficient attributes: {det}")
                    continue  # Skip incomplete detections
                
                # Extract detection attributes
                x1, y1, x2, y2, confidence, class_id = det[:6]
                
                # Convert normalized coordinates to image coordinates
                x1 = int(float(x1) * original_image_shape[1])
                y1 = int(float(y1) * original_image_shape[0])
                x2 = int(float(x2) * original_image_shape[1])
                y2 = int(float(y2) * original_image_shape[0])
                
                detections.append({
                    'class_id': int(class_id),
                    'confidence': float(confidence),
                    'bbox': [x1, y1, x2, y2]
                })
            except Exception as e:
                self.get_logger().error(f"Error processing detection {idx}: {e}")
                continue  # Skip problematic detections
        
        self.get_logger().info(f"Total Valid Detections: {len(detections)}")
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

        # Save the overlay image to the output directory
        output_path = os.path.join(self.output_dir, f"prediction_{self.image_counter}.png")
        cv2.imwrite(output_path, overlay)
        self.get_logger().info(f"Saved prediction image to {output_path}")
        self.image_counter += 1

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

    def process_image(self, cv_image):
        if cv_image is None or cv_image.size == 0:
            self.get_logger().error("Received an empty or invalid image.")
            return

        if cv_image.shape[0] == 0 or cv_image.shape[1] == 0:
            self.get_logger().error(f"Invalid image dimensions: {cv_image.shape}")
            return

        try:
            self.get_logger().info(f"Processing image with shape: {cv_image.shape} and type: {cv_image.dtype}")

            # Preprocess images for each model
            segmentation_input = self.preprocess_image(cv_image, (512, 512))  # Updated to 512x512
            detection_input = self.preprocess_image(cv_image, (640, 640))  # Unchanged

            # Run models
            seg_mask = self.run_segmentation(segmentation_input)
            detection_results = self.run_detection(detection_input, cv_image.shape)

            # Visualize and save results
            self.visualize_results(cv_image, seg_mask, detection_results)

        except Exception as e:
            self.get_logger().error(f"Error during image processing: {e}")

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

                self.process_image(cv_image)

                if idx % 10 == 0:
                    self.get_logger().info(f"Processed {idx}/{len(messages)} images.")

            except CvBridgeError as e:
                self.get_logger().error(f"CvBridgeError for message ID {idx}: {e}")
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

        if not os.path.exists(db_path):
            node.get_logger().error(f"Database file does not exist at {db_path}")
            sys.exit(1)

        node.get_logger().info("Starting to read and process messages from the bag file.")
        node.read_and_process_messages(db_path)
    except KeyboardInterrupt:
        node.get_logger().info("Keyboard interrupt received. Shutting down.")
    except Exception as e:
        node.get_logger().error(f"An unexpected error occurred: {e}")
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
