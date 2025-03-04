import cv2
import numpy as np
import torch
from PIL import Image
import torchvision.transforms as transforms
import sys
import os

# Add YOLOv5 to path
yolov5_path = r'C:\Users\kenjo\Documents\GitHub\Child-Safety-System\yolov5'
if yolov5_path not in sys.path:
    sys.path.append(yolov5_path)

# Confidence threshold (adjust as needed)
confidence_threshold = 0.1  # Very low threshold for better detection

# COCO class IDs and labels
COCO_LABELS = {
    1: 'person', 2: 'bicycle', 3: 'car', 4: 'motorcycle', 5: 'airplane', 6: 'bus',
    7: 'train', 8: 'truck', 9: 'boat', 10: 'traffic light', 11: 'fire hydrant',
    13: 'stop sign', 14: 'parking meter', 15: 'bench', 16: 'bird', 17: 'cat',
    18: 'dog', 19: 'horse', 20: 'sheep', 21: 'cow', 22: 'elephant', 23: 'bear',
    24: 'zebra', 25: 'giraffe', 27: 'backpack', 28: 'umbrella', 31: 'handbag',
    32: 'tie', 33: 'suitcase', 34: 'frisbee', 35: 'skis', 36: 'snowboard',
    37: 'sports ball', 38: 'kite', 39: 'baseball bat', 40: 'baseball glove',
    41: 'skateboard', 42: 'surfboard', 43: 'tennis racket', 44: 'bottle',
    46: 'knife', 47: 'bowl', 48: 'banana', 49: 'apple', 50: 'sandwich',
    51: 'orange', 52: 'broccoli', 53: 'carrot', 54: 'hot dog', 55: 'pizza',
    56: 'donut', 57: 'cake', 58: 'chair', 59: 'couch', 60: 'potted plant',
    61: 'bed', 62: 'dining table', 63: 'toilet', 65: 'tv', 67: 'laptop',
    70: 'mouse', 71: 'remote', 72: 'keyboard', 73: 'cell phone', 74: 'microwave',
    75: 'oven', 76: 'toaster', 77: 'sink', 78: 'refrigerator', 79: 'scissors',
    80: 'teddy bear', 81: 'hair drier', 82: 'toothbrush'
}

# Simplify dangerous classes to focus on core items
DANGEROUS_CLASSES = {
    46: 'knife',
    79: 'scissors',
    39: 'baseball bat',
    44: 'bottle',
    81: 'hair drier'
}

# Remove detection history for immediate detection
HISTORY_FRAMES = 1  # Set to 1 for immediate detection

def trigger_alert(label, confidence, bbox):
    print(f"ALERT: Dangerous item '{label}' detected with confidence {confidence:.2f}")

# Load YOLOv5n model
try:
    model = torch.hub.load('ultralytics/yolov5', 'yolov5n', pretrained=True)
    model.conf = confidence_threshold  # Set confidence threshold
    print("Using YOLOv5n model from hub")
except Exception as e:
    print(f"Error loading YOLO model: {e}")
    print("Please make sure you have installed the required packages:")
    print("pip install torch torchvision")
    print("pip install ultralytics")
    exit()

# Initialize camera with basic settings
try:
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open video/webcam.")
        exit()
    
    # Set camera properties to basic values
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)
    cap.set(cv2.CAP_PROP_AUTOFOCUS, 1)
    cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)
except Exception as e:
    print(f"Error initializing camera: {e}")
    exit()

while cap.isOpened():
    try:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to grab frame")
            break

        if frame is None:
            print("Error: Frame is None")
            continue

        # Convert frame to RGB for YOLOv5
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Run YOLOv5 inference
        results = model(frame_rgb)
        
        # Process detections
        for det in results.xyxy[0]:  # xyxy format
            x1, y1, x2, y2, conf, cls = det.cpu().numpy()
            class_id = int(cls)
            
            # Convert to integers
            x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
            
            # Get label
            label = COCO_LABELS.get(class_id, 'unknown')
            
            # Print all detections for debugging
            print(f"\nDetection: {label}")
            print(f"Confidence: {conf:.2f}")
            print(f"Class ID: {class_id}")
            print(f"Box coordinates: ({x1}, {y1}, {x2}, {y2})")
            
            # Choose color based on object type
            if class_id in DANGEROUS_CLASSES:
                box_color = (0, 0, 255)  # Red for dangerous items
                text_color = (0, 0, 255)
                trigger_alert(label, conf, (x1, y1, x2, y2))
            else:
                box_color = (0, 255, 0)  # Green for other objects
                text_color = (0, 255, 0)

            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)

            # Draw label and confidence
            label_text = f"{label} ({conf:.2f})"
            cv2.putText(frame, label_text, 
                      (x1, y1 - 10 if y1 - 10 > 10 else y1 + 10),
                      cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 2)

        # Display frame
        cv2.imshow('Object Detection', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    except Exception as e:
        print(f"Error in main loop: {e}")
        continue

# Clean up
try:
    cap.release()
    cv2.destroyAllWindows()
except Exception as e:
    print(f"Error during cleanup: {e}")
