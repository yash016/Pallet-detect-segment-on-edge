#!/usr/bin/python3
import sys
print(f"Python executable: {sys.executable}")
print(f"Python sys.path: {sys.path}")

import rclpy
from rclpy.node import Node
import torch
import cv2
import numpy as np
from ultralytics import YOLO
import segmentation_models_pytorch as smp
from torchvision import transforms
import os
from ament_index_python.packages import get_package_share_directory

class ObjectDetectionNode(Node):
    def __init__(self):
        super().__init__('object_detection_node')
        
        # Load models
        self.load_models()
        
        # Get the share directory of the package
        package_share_directory = get_package_share_directory('object_detection_semantic_segmentation_pkg')
        
        # Path to the input video file
        input_video_path = os.path.join(package_share_directory, 'videos', "input_video.mp4")
        
        # Initialize video capture
        self.cap = cv2.VideoCapture(input_video_path)
        
        # Check if video opened successfully
        if not self.cap.isOpened():
            self.get_logger().error(f"Error opening video stream or file at {input_video_path}")
            sys.exit(1)
        
        # Get video properties
        self.frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        if self.fps == 0:
            self.get_logger().warn("FPS value is 0. Setting default FPS to 30.")
            self.fps = 30  # Default FPS
        
        # Define the codec and create VideoWriter object
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        output_video_path = os.path.join(package_share_directory, 'videos', 'output_video.mp4')
        self.out = cv2.VideoWriter(output_video_path, fourcc, self.fps, (self.frame_width, self.frame_height))
        
        # Initialize variables for processing every 1 second
        self.current_time = 0.0  # Start time in seconds
        self.duration = self.cap.get(cv2.CAP_PROP_FRAME_COUNT) / self.fps  # Total duration in seconds
        
        # Create a timer to process frames every 1 second
        self.timer = self.create_timer(1.0, self.timer_callback)

    def load_models(self):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Get the share directory of the package
        package_share_directory = get_package_share_directory('object_detection_semantic_segmentation_pkg')
        
        # Define paths to the models relative to the package share directory
        segmentation_model_path = os.path.join(package_share_directory, 'models', 'best_deeplabv3plus_model.pth')
        detection_model_path = os.path.join(package_share_directory, 'models', 'best_YOLO11x_model.pt')
        
        # Check if segmentation model exists
        if not os.path.exists(segmentation_model_path):
            self.get_logger().error(f"Segmentation model file does not exist at {segmentation_model_path}")
            sys.exit(1)
        
        # Check if detection model exists
        if not os.path.exists(detection_model_path):
            self.get_logger().error(f"Detection model file does not exist at {detection_model_path}")
            sys.exit(1)
        
        # Load segmentation model
        self.segmentation_model = self.create_segmentation_model()
        try:
            self.segmentation_model.load_state_dict(
                torch.load(segmentation_model_path, map_location=device))
        except Exception as e:
            self.get_logger().error(f"Failed to load segmentation model state_dict: {e}")
            sys.exit(1)
        self.segmentation_model.to(device)
        self.segmentation_model.eval()
        
        # Load detection model for inference
        try:
            self.detection_model = YOLO(detection_model_path)
            self.detection_model.to(device)
        except Exception as e:
            self.get_logger().error(f"Failed to load detection model: {e}")
            sys.exit(1)
        
        # Set device for use in callbacks
        self.device = device
    
    def create_segmentation_model(self):
        # Define the DeepLabV3+ segmentation model with ResNet-101 backbone
        model = smp.DeepLabV3Plus(
            encoder_name='resnet101',
            encoder_weights=None,
            in_channels=3,
            classes=3,
            activation=None
        )
        return model
    
    # Timer callback to process frames every 1 second of video
    def timer_callback(self):
        # Set the video position to the current time in milliseconds
        self.cap.set(cv2.CAP_PROP_POS_MSEC, self.current_time * 1000)
        
        ret, frame = self.cap.read()
        if ret:
            # Process the frame
            self.process_image(frame)
            self.current_time += 1.0  # Move to the next second
        else:
            # If unable to read frame, stop the node since we've reached the end of the video
            self.get_logger().info("End of video stream. Stopping the node.")
            self.out.release()
            rclpy.shutdown()  # Stop the node

    def preprocess_image(self, cv_image):
        # Preprocess the image as required by the segmentation model
        preprocess = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((512, 512)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
        input_tensor = preprocess(cv_image).unsqueeze(0).to(self.device)
        return input_tensor
    
    def process_image(self, cv_image):
        # Perform preprocessing for segmentation model
        input_tensor = self.preprocess_image(cv_image)
        
        # Perform segmentation
        with torch.no_grad():
            seg_output = self.segmentation_model(input_tensor)
            seg_prediction = torch.argmax(seg_output.squeeze(), dim=0).cpu().numpy()
        
        # Resize segmentation mask to match original frame size
        seg_prediction_resized = cv2.resize(seg_prediction, (cv_image.shape[1], cv_image.shape[0]), interpolation=cv2.INTER_NEAREST)
        
        # Decode segmentation mask to RGB
        seg_rgb = self.decode_segmap(seg_prediction_resized)
        
        # Perform object detection
        detection_results = self.detection_model.predict(cv_image, conf=0.1)
        
        # Visualize the results by overlaying on the frame
        self.visualize_results(cv_image, seg_rgb, detection_results)
    
    def post_process_segmentation(self, mask):
        # Apply morphological operations to clean up the segmentation mask
        mask = mask.astype(np.uint8)
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        return mask

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
    
    def visualize_results(self, cv_image, seg_rgb, detection_results):
        # Post-process the segmentation mask
        seg_mask = self.post_process_segmentation(seg_rgb)
        
        # Ensure seg_rgb has the same size as cv_image
        if seg_rgb.shape[:2] != cv_image.shape[:2]:
            seg_rgb = cv2.resize(seg_rgb, (cv_image.shape[1], cv_image.shape[0]), interpolation=cv2.INTER_NEAREST)
        
        # Overlay segmentation mask on the original image
        overlay = cv2.addWeighted(cv_image, 0.6, seg_rgb, 0.4, 0)
        
        # Draw bounding boxes from detection results
        for result in detection_results:
            boxes = result.boxes
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                label = int(box.cls)
                confidence = float(box.conf)
                cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(overlay, f"{label}: {confidence:.2f}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Write the frame to the output video
        self.out.write(overlay)
        
        # Optionally, display the frame (only if you have GUI support)
        # cv2.imshow("Object Detection & Segmentation", overlay)
        # cv2.waitKey(1)
    
    def destroy_node(self):
        # Release video writer and video capture
        self.out.release()
        self.cap.release()
        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    node = ObjectDetectionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        # Only call shutdown if rclpy is still running
        if rclpy.ok():
            rclpy.shutdown()

if __name__ == '__main__':
    main()
