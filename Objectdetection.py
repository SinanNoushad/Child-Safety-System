import cv2
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub

# Confidence threshold 
CONFIDENCE_THRESHOLD = 0.6

# COCO class labels (reduced for simplicity)
COCO_LABELS = {
    1: 'person', 2: 'bicycle', 3: 'car', 4: 'motorcycle', 
    10: 'traffic light', 11: 'fire hydrant',
    46: 'knife', 47: 'bowl', 58: 'chair', 59: 'couch',
    73: 'cell phone', 79: 'scissors', 80: 'teddy bear', 84:'book'
}

# Dangerous classes to trigger alerts
DANGEROUS_CLASSES = {
    46: 'knife',
    79: 'scissors',
    84: 'book'
}

def trigger_alert(label, confidence, bbox):
    """Trigger an alert for potentially dangerous objects."""
    print(f"ALERT: Dangerous item '{label}' detected with confidence {confidence:.2f}")

def load_tensorflow_model():
    """
    Load a pre-trained TensorFlow object detection model from TensorFlow Hub.
    Using a lightweight model for better performance.
    """
    try:
        # Load MobileNet V2 SSD model
        model = hub.load("https://tfhub.dev/tensorflow/ssd_mobilenet_v2/2")
        print("TensorFlow object detection model loaded successfully")
        return model
    except Exception as e:
        print(f"Error loading TensorFlow model: {e}")
        print("Ensure you have tensorflow and tensorflow-hub installed:")
        print("pip install tensorflow tensorflow-hub")
        exit(1)

def main():
    # Load the TensorFlow object detection model
    detect_fn = load_tensorflow_model()

    # Initialize camera
    try:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Error: Could not open video/webcam.")
            return

        # Set camera properties
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)
    except Exception as e:
        print(f"Camera initialization error: {e}")
        return

    while cap.isOpened():
        try:
            # Capture frame-by-frame
            ret, frame = cap.read()
            if not ret or frame is None:
                print("Failed to capture frame")
                break

            # Preprocess the image for TensorFlow detection
            input_tensor = tf.convert_to_tensor(frame)
            input_tensor = input_tensor[tf.newaxis, ...]

            # Perform detection
            detections = detect_fn(input_tensor)

            # Convert detections to numpy arrays
            num_detections = int(detections['num_detections'])
            classes = detections['detection_classes'][0][:num_detections].numpy().astype(np.int32)
            scores = detections['detection_scores'][0][:num_detections].numpy()
            boxes = detections['detection_boxes'][0][:num_detections].numpy()

            # Process each detection
            for i in range(num_detections):
                class_id = classes[i]
                confidence = scores[i]

                # Skip low confidence detections
                if confidence < CONFIDENCE_THRESHOLD:
                    continue

                # Get label
                label = COCO_LABELS.get(class_id, 'unknown')

                # Get bounding box coordinates
                ymin, xmin, ymax, xmax = boxes[i]
                im_height, im_width, _ = frame.shape
                
                # Convert normalized coordinates to pixel coordinates
                left = int(xmin * im_width)
                top = int(ymin * im_height)
                right = int(xmax * im_width)
                bottom = int(ymax * im_height)

                # Print detection details
                print(f"\nDetection: {label}")
                print(f"Confidence: {confidence:.2f}")
                print(f"Box coordinates: ({left}, {top}, {right}, {bottom})")

                # Determine box and text colors
                if class_id in DANGEROUS_CLASSES:
                    box_color = (0, 0, 255)  # Red for dangerous items
                    trigger_alert(label, confidence, (left, top, right, bottom))
                else:
                    box_color = (0, 255, 0)  # Green for other objects

                # Draw bounding box
                cv2.rectangle(frame, (left, top), (right, bottom), box_color, 2)

                # Draw label and confidence
                label_text = f"{label} ({confidence:.2f})"
                cv2.putText(frame, label_text, 
                            (left, top - 10 if top - 10 > 10 else top + 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_color, 2)

            # Display the resulting frame
            cv2.imshow('TensorFlow Object Detection', frame)

            # Break loop on 'q' key press
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        except Exception as e:
            print(f"Error in detection loop: {e}")
            break

    # Clean up
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()