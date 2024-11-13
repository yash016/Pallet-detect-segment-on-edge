#!/usr/bin/env python3
import sys
print(f"Python executable: {sys.executable}")
print(f"Python sys.path: {sys.path}")

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image  # Now using ROS Image messages
from cv_bridge import CvBridge     # For converting ROS Image messages to OpenCV format
import cv2
import numpy as np
import os
from ament_index_python.packages import get_package_share_directory

# TensorRT imports
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit

class ObjectDetectionNode(Node):
    def __init__(self):
        super().__init__('object_detection_node')

        # Load models
        self.load_models()

        # ROS subscription to image topic
        self.subscription = self.create_subscription(
            Image,
            '/camera/image_raw',  # Change this to your image topic
            self.listener_callback,
            10)

        # Bridge for converting ROS images to OpenCV format
        self.bridge = CvBridge()

    def load_models(self):
        # Get the share directory of the package
        package_share_directory = get_package_share_directory('object_detection_semantic_segmentation_pkg')

        # Define paths to the TensorRT engine files relative to the package share directory
        segmentation_engine_path = os.path.join(package_share_directory, 'models', 'segmentation_model.engine')
        detection_engine_path = os.path.join(package_share_directory, 'models', 'detection_model.engine')

        # Check if segmentation engine exists
        if not os.path.exists(segmentation_engine_path):
            self.get_logger().error(f"Segmentation engine file does not exist at {segmentation_engine_path}")
            sys.exit(1)

        # Check if detection engine exists
        if not os.path.exists(detection_engine_path):
            self.get_logger().error(f"Detection engine file does not exist at {detection_engine_path}")
            sys.exit(1)

        # Load TensorRT engines
        self.trt_logger = trt.Logger(trt.Logger.WARNING)

        # Load segmentation engine
        with open(segmentation_engine_path, 'rb') as f, trt.Runtime(self.trt_logger) as runtime:
            self.segmentation_engine = runtime.deserialize_cuda_engine(f.read())
        self.segmentation_context = self.segmentation_engine.create_execution_context()

        # Load detection engine
        with open(detection_engine_path, 'rb') as f, trt.Runtime(self.trt_logger) as runtime:
            self.detection_engine = runtime.deserialize_cuda_engine(f.read())
        self.detection_context = self.detection_engine.create_execution_context()

        # Prepare buffers for TensorRT engines
        self.allocate_buffers()

    def allocate_buffers(self):
        # Allocate buffers for segmentation model
        self.segmentation_inputs = []
        self.segmentation_outputs = []
        self.segmentation_bindings = []
        self.segmentation_stream = cuda.Stream()

        for binding in self.segmentation_engine:
            size = trt.volume(self.segmentation_engine.get_binding_shape(binding)) * self.segmentation_engine.max_batch_size
            dtype = trt.nptype(self.segmentation_engine.get_binding_dtype(binding))
            # Allocate host and device buffers
            host_mem = cuda.pagelocked_empty(size, dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)
            # Append the device buffer to device bindings.
            self.segmentation_bindings.append(int(device_mem))
            # Append to the appropriate list.
            if self.segmentation_engine.binding_is_input(binding):
                self.segmentation_inputs.append({'host': host_mem, 'device': device_mem})
            else:
                self.segmentation_outputs.append({'host': host_mem, 'device': device_mem})

        # Allocate buffers for detection model
        self.detection_inputs = []
        self.detection_outputs = []
        self.detection_bindings = []
        self.detection_stream = cuda.Stream()

        for binding in self.detection_engine:
            size = trt.volume(self.detection_engine.get_binding_shape(binding)) * self.detection_engine.max_batch_size
            dtype = trt.nptype(self.detection_engine.get_binding_dtype(binding))
            # Allocate host and device buffers
            host_mem = cuda.pagelocked_empty(size, dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)
            # Append the device buffer to device bindings.
            self.detection_bindings.append(int(device_mem))
            # Append to the appropriate list.
            if self.detection_engine.binding_is_input(binding):
                self.detection_inputs.append({'host': host_mem, 'device': device_mem})
            else:
                self.detection_outputs.append({'host': host_mem, 'device': device_mem})

    def listener_callback(self, msg):
        # Convert ROS Image message to OpenCV format
        cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

        # Process the image
        self.process_image(cv_image)

    def preprocess_image(self, cv_image, model_input_shape):
        # Resize and normalize the image as required by the model
        preprocessed = cv2.resize(cv_image, (model_input_shape[2], model_input_shape[1]))
        preprocessed = preprocessed.astype(np.float32) / 255.0
        preprocessed = preprocessed.transpose((2, 0, 1))  # HWC to CHW
        return preprocessed

    def process_image(self, cv_image):
        # Get input shapes
        segmentation_input_shape = self.segmentation_engine.get_binding_shape(0)  # Assuming input index 0
        detection_input_shape = self.detection_engine.get_binding_shape(0)  # Assuming input index 0

        # Preprocess images
        segmentation_input = self.preprocess_image(cv_image, segmentation_input_shape)
        detection_input = self.preprocess_image(cv_image, detection_input_shape)

        # Run inference
        seg_mask = self.run_segmentation(segmentation_input)
        detection_results = self.run_detection(detection_input, cv_image.shape)

        # Visualize the results
        self.visualize_results(cv_image, seg_mask, detection_results)

    def run_segmentation(self, input_image):
        # Copy input image to host input buffer
        np.copyto(self.segmentation_inputs[0]['host'], input_image.ravel())

        # Transfer input data to the GPU.
        cuda.memcpy_htod_async(self.segmentation_inputs[0]['device'], self.segmentation_inputs[0]['host'], self.segmentation_stream)
        # Run inference.
        self.segmentation_context.execute_async(bindings=self.segmentation_bindings, stream_handle=self.segmentation_stream.handle)
        # Transfer predictions back from the GPU.
        cuda.memcpy_dtoh_async(self.segmentation_outputs[0]['host'], self.segmentation_outputs[0]['device'], self.segmentation_stream)
        # Synchronize the stream
        self.segmentation_stream.synchronize()

        # Get output and reshape
        output = self.segmentation_outputs[0]['host']
        seg_output = output.reshape(self.segmentation_engine.get_binding_shape(1))  # Assuming output index 1
        seg_prediction = np.argmax(seg_output, axis=0).astype(np.uint8)
        return seg_prediction

    def run_detection(self, input_image, original_image_shape):
        # Copy input image to host input buffer
        np.copyto(self.detection_inputs[0]['host'], input_image.ravel())

        # Transfer input data to the GPU.
        cuda.memcpy_htod_async(self.detection_inputs[0]['device'], self.detection_inputs[0]['host'], self.detection_stream)
        # Run inference.
        self.detection_context.execute_async(bindings=self.detection_bindings, stream_handle=self.detection_stream.handle)
        # Transfer predictions back from the GPU.
        cuda.memcpy_dtoh_async(self.detection_outputs[0]['host'], self.detection_outputs[0]['device'], self.detection_stream)
        # Synchronize the stream
        self.detection_stream.synchronize()

        # Get output and process detections
        output = self.detection_outputs[0]['host']
        # The output processing depends on your model's output format
        detection_results = self.postprocess_detections(output, original_image_shape)
        return detection_results

    def postprocess_detections(self, output, original_image_shape):
        # Process detection output to extract bounding boxes, class labels, and confidences
        # This depends on your specific model's output format
        # Placeholder implementation:

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
        # Resize segmentation mask to match original frame size
        seg_mask_resized = cv2.resize(seg_mask, (cv_image.shape[1], cv_image.shape[0]), interpolation=cv2.INTER_NEAREST)

        # Decode segmentation mask to RGB
        seg_rgb = self.decode_segmap(seg_mask_resized)

        # Overlay segmentation mask on the original image
        overlay = cv2.addWeighted(cv_image, 0.6, seg_rgb, 0.4, 0)

        # Draw bounding boxes from detection results
        for det in detection_results:
            x1, y1, x2, y2 = det['bbox']
            class_id = det['class_id']
            confidence = det['confidence']
            cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(overlay, f"{class_id}: {confidence:.2f}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # Optionally, display the frame (only if you have GUI support)
        # cv2.imshow("Object Detection & Segmentation", overlay)
        # cv2.waitKey(1)

    def decode_segmap(self, mask):
        # Decode the segmentation mask to RGB
        label_colors = np.array([
            (0, 0, 0),          # Background - Black
            (128, 128, 128),    # Ground - Gray
            (255, 255, 255)     # Pallet - White
        ])
        r, g, b = [np.zeros_like(mask).astype(np.uint8) for _ in range(3)]
        for l in range(len(label_colors)):
            idx = mask == l
            r[idx], g[idx], b[idx] = label_colors[l]
        return np.stack([r, g, b], axis=2)

    def destroy_node(self):
        # Release resources
        super().destroy_node()

    def main(args=None):
        rclpy.init(args=args)
        node = ObjectDetectionNode()
        try:
            rclpy.spin(node)
        except KeyboardInterrupt:
            pass
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    ObjectDetectionNode.main()
