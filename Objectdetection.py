import cv2
import numpy as np
import onnxruntime as ort

# Confidence threshold (adjust as needed)
confidence_threshold = 0.3  # You can lower this value for testing

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

# COCO class IDs for dangerous items
DANGEROUS_CLASSES = {
    46: 'knife',
    79: 'scissors'
}

def trigger_alert(label, confidence, bbox):
    print(f"ALERT: Dangerous item '{label}' detected with confidence {confidence:.2f}")

# Load ONNX model (update the path as needed)
session = ort.InferenceSession(r"C:\Users\kenjo\Documents\GitHub\Child-Safety-System\models\yolov8n.onnx")
input_name = session.get_inputs()[0].name

cap = cv2.VideoCapture(0)  # Use 0 for webcam

if not cap.isOpened():
    print("Error: Could not open video/webcam.")
    exit()

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    original_height, original_width = frame.shape[:2]

    # Preprocess: convert to RGB and letterbox to target size (e.g., 640x640)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    target_size = 640

    # Calculate scale for resizing while maintaining aspect ratio
    scale = min(target_size / original_width, target_size / original_height)
    resized_width = int(original_width * scale)
    resized_height = int(original_height * scale)

    resized_frame = cv2.resize(rgb_frame, (resized_width, resized_height))

    # Create a new image of target size and center the resized frame
    padded_frame = np.zeros((target_size, target_size, 3), dtype=np.uint8)
    pad_width = (target_size - resized_width) // 2
    pad_height = (target_size - resized_height) // 2
    padded_frame[pad_height:pad_height+resized_height, pad_width:pad_width+resized_width] = resized_frame

    # Normalize and prepare input tensor
    input_image = padded_frame.astype(np.float32) / 255.0  # Normalize to [0,1]
    input_tensor = np.expand_dims(input_image, axis=0)      # [1, 640, 640, 3]
    input_tensor = np.transpose(input_tensor, (0, 3, 1, 2)) # [1, 3, 640, 640]

    # Run inference
    outputs = session.run(None, {input_name: input_tensor})
    predictions = outputs[0]
    detections = predictions[0]  # Assuming the output is [num_detections, 6]

    detected_objects = []  # For overlaying labels

    for detection in detections:
        x1, y1, x2, y2, conf, cls = detection[:6]

        if conf < confidence_threshold:
            continue

        # If coordinates are normalized between 0 and 1, scale them back to image size
        if 0 <= x1 <= 1 and 0 <= y1 <= 1 and 0 <= x2 <= 1 and 0 <= y2 <= 1:
            x1 *= target_size
            y1 *= target_size
            x2 *= target_size
            y2 *= target_size

        # Adjust for padding (subtract pad_width and pad_height)
        x1 -= pad_width
        y1 -= pad_height
        x2 -= pad_width
        y2 -= pad_height

        # Scale coordinates back to original image dimensions
        x1 = x1 / scale
        y1 = y1 / scale
        x2 = x2 / scale
        y2 = y2 / scale

        # Convert to integers
        x1 = int(max(0, min(original_width - 1, x1)))
        y1 = int(max(0, min(original_height - 1, y1)))
        x2 = int(max(0, min(original_width - 1, x2)))
        y2 = int(max(0, min(original_height - 1, y2)))

        label = COCO_LABELS.get(int(cls), 'unknown')
        detected_objects.append(label)

        # Choose color based on object type
        if int(cls) in DANGEROUS_CLASSES:
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
        cv2.putText(frame, label_text, (x1, y1 - 10 if y1 - 10 > 10 else y1 + 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 2)

    # Overlay unique detected objects on the top-left
    if detected_objects:
        overlay_text = "Detected: " + ", ".join(set(detected_objects))
        cv2.putText(frame, overlay_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

    cv2.imshow('Object Detection', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
