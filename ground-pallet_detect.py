import os
import glob
import re
import random
from ultralytics import YOLO
from IPython.display import Image, display

# Import utility functions and constants
from utils import test_images_dir, DATASET_PATH

# Load the best-trained model
best_model_path = os.path.join('runs', 'pallet_detection', 'best_model.pt')
model = YOLO(best_model_path)

# Define inference function
def run_inference_on_test_images(model, test_images_dir, num_images=5):
    test_images = glob.glob(os.path.join(test_images_dir, '*.jpg')) + \
                  glob.glob(os.path.join(test_images_dir, '*.png')) + \
                  glob.glob(os.path.join(test_images_dir, '*.jpeg'))
    if not test_images:
        print("No test images found.")
        return

    selected_images = random.sample(test_images, min(num_images, len(test_images)))
    results = model.predict(
        source=selected_images,
        conf=0.1,
        save=True,
        imgsz=640,
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )
    print(f"Processed {len(selected_images)} images.")

    # Display predicted images
    detect_dir = os.path.join('runs', 'detect')
    if os.path.exists(detect_dir):
        predict_dirs = [d for d in os.listdir(detect_dir) if re.match(r'^predict\d+$', d)]
        if predict_dirs:
            latest_predict_dir = os.path.join(detect_dir, sorted(predict_dirs)[-1])
            predicted_images = glob.glob(os.path.join(latest_predict_dir, '*.jpg')) + \
                               glob.glob(os.path.join(latest_predict_dir, '*.png'))
            for img_path in predicted_images[:num_images]:
                display(Image(filename=img_path, width=416))

# Run inference
if __name__ == "__main__":
    run_inference_on_test_images(model, test_images_dir)
