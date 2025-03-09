import os
os.environ["LIBGL_ALWAYS_SOFTWARE"] = "1"  # Suppress GPU warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"   # Reduce TensorFlow logging

import cv2
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import dlib
import pickle
from mtcnn import MTCNN
from imutils import face_utils
from imutils.object_detection import non_max_suppression
from collections import deque
import math

# ------------------------- 
# Configuration
# -------------------------

# Object Detection Config
CONFIDENCE_THRESHOLD = 0.6
COCO_LABELS = {1: 'person', 2: 'bicycle', 3: 'car', 4: 'motorcycle', 
               10: 'traffic light', 11: 'fire hydrant', 46: 'knife', 
               47: 'bowl', 58: 'chair', 59: 'couch', 73: 'cell phone', 
               79: 'scissors', 80: 'teddy bear', 84: 'book'}
DANGEROUS_CLASSES = {46: 'knife', 79: 'scissors', 84: 'book'}

# Pose Estimation Config
ACTION_HISTORY_LENGTH = 7
action_history = deque(maxlen=ACTION_HISTORY_LENGTH)
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose(min_detection_confidence=0.7, min_tracking_confidence=0.7)

# Facial Recognition Config
KNOWN_FACES_DIR = "faces"
ENCODINGS_FILE = "face_encodings.pkl"
face_encoding_dict = {}
PROCESS_EVERY_N_FRAMES = 2  # Process face detection every other frame
SCALE_FACTOR = 0.5  # Reduced resolution for detection
TRACKING_THRESHOLD = 0.4

# Video Input (0 for webcam, or path to video file)
VIDEO_SOURCE = 0

# -------------------------
# Object Detection Functions
# -------------------------

def load_tensorflow_model():
    try:
        model = hub.load("https://tfhub.dev/tensorflow/ssd_mobilenet_v2/2")
        print("TensorFlow object detection model loaded successfully")
        return model
    except Exception as e:
        print(f"Error loading TensorFlow model: {e}")
        exit(1)

def trigger_alert(label, confidence, bbox):
    print(f"ALERT: Dangerous item '{label}' detected with confidence {confidence:.2f}")

# -------------------------
# Pose Estimation Functions
# -------------------------

def calculate_angle(a, b, c):
    ba = [a[0]-b[0], a[1]-b[1]]
    bc = [c[0]-b[0], c[1]-b[1]]
    dot_product = ba[0]*bc[0] + ba[1]*bc[1]
    mag_ba = math.sqrt(ba[0]**2 + ba[1]**2)
    mag_bc = math.sqrt(bc[0]**2 + bc[1]**2)
    cosine_angle = dot_product / (mag_ba * mag_bc + 1e-10)
    angle = math.degrees(math.acos(max(min(cosine_angle, 1), -1)))
    return angle

def get_action(landmarks, frame_width, frame_height):
    actions = []
    try:
        # Key landmarks (simplified for brevity)
        left_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                         landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
        right_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                          landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
        left_hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,
                    landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
        right_hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,
                     landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
        left_knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,
                     landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
        right_knee = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x,
                      landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y]
        left_ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x,
                      landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]
        right_ankle = [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x,
                       landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y]
        left_wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,
                      landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
        right_wrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,
                       landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]
    except:
        return "Landmarks missing"

    # Simplified action detection
    avg_knee_angle = (calculate_angle(left_hip, left_knee, left_ankle) + 
                      calculate_angle(right_hip, right_knee, right_ankle)) / 2
    torso_angle = calculate_angle(
        [(left_shoulder[0] + right_shoulder[0])/2, (left_shoulder[1] + right_shoulder[1])/2],
        [(left_hip[0] + right_hip[0])/2, (left_hip[1] + right_hip[1])/2],
        [right_hip[0], left_hip[1] + 0.2]
    )
    
    if avg_knee_angle > 160 and torso_angle > 70:
        actions.append("Standing")
    elif avg_knee_angle < 110:
        actions.append("Sitting")
    elif torso_angle < 30:
        actions.append("Laying Down")
    
    return ", ".join(actions) if actions else "Neutral"

# -------------------------
# Facial Recognition Functions
# -------------------------

detector = MTCNN()
predictor = dlib.shape_predictor("models/shape_predictor_68_face_landmarks.dat")
face_encoder = dlib.face_recognition_model_v1("models/dlib_face_recognition_resnet_model_v1.dat")

def get_face_encoding(image, face):
    x, y, w, h = face
    dlib_rect = dlib.rectangle(x, y, x + w, y + h)
    shape = predictor(image, dlib_rect)
    return np.array(face_encoder.compute_face_descriptor(image, shape, num_jitters=1))

def find_best_match(unknown_encoding, threshold=0.5):
    if not face_encoding_dict:
        return "Intruder"
    known_names = list(face_encoding_dict.keys())
    known_encodings = np.vstack(list(face_encoding_dict.values()))
    distances = np.linalg.norm(known_encodings - unknown_encoding, axis=1)
    min_idx = np.argmin(distances)
    return "Intruder" if distances[min_idx] > threshold else known_names[min_idx]

def load_face_encodings():
    global face_encoding_dict
    if os.path.exists(ENCODINGS_FILE):
        with open(ENCODINGS_FILE, "rb") as f:
            face_encoding_dict = pickle.load(f)
        print(f"Loaded {len(face_encoding_dict)} encodings")
    else:
        print("Processing known faces...")
        for person_name in os.listdir(KNOWN_FACES_DIR):
            person_dir = os.path.join(KNOWN_FACES_DIR, person_name)
            if not os.path.isdir(person_dir):
                continue
            encodings = []
            for filename in os.listdir(person_dir):
                if not filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                    continue
                image = cv2.imread(os.path.join(person_dir, filename))
                rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                detected_faces = detector.detect_faces(rgb_image)
                if detected_faces:
                    best_face = max(detected_faces, key=lambda f: f['box'][2] * f['box'][3])
                    box = best_face['box']
                    encoding = get_face_encoding(rgb_image, box)
                    encodings.append(encoding)
            if encodings:
                face_encoding_dict[person_name] = np.mean(encodings, axis=0)
        with open(ENCODINGS_FILE, "wb") as f:
            pickle.dump(face_encoding_dict, f)
        print(f"Saved {len(face_encoding_dict)} encodings")

# -------------------------
# Main Processing Loop
# -------------------------

def main():
    # Load models
    detect_fn = load_tensorflow_model()
    load_face_encodings()
    
    # Initialize video capture
    cap = cv2.VideoCapture(VIDEO_SOURCE)
    if not cap.isOpened():
        print("Error: Could not open video/webcam.")
        return
    
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)
    
    tracked_faces = {}
    next_face_id = 0
    frame_count = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Object Detection
        input_tensor = tf.convert_to_tensor(frame)[tf.newaxis, ...]
        detections = detect_fn(input_tensor)
        num_detections = int(detections['num_detections'])
        classes = detections['detection_classes'][0][:num_detections].numpy().astype(np.int32)
        scores = detections['detection_scores'][0][:num_detections].numpy()
        boxes = detections['detection_boxes'][0][:num_detections].numpy()
        
        im_height, im_width = frame.shape[:2]
        for i in range(num_detections):
            if scores[i] < CONFIDENCE_THRESHOLD:
                continue
            class_id = classes[i]
            label = COCO_LABELS.get(class_id, 'unknown')
            ymin, xmin, ymax, xmax = boxes[i]
            left, top, right, bottom = (int(xmin * im_width), int(ymin * im_height),
                                       int(xmax * im_width), int(ymax * im_height))
            box_color = (0, 0, 255) if class_id in DANGEROUS_CLASSES else (0, 255, 0)
            cv2.rectangle(frame, (left, top), (right, bottom), box_color, 2)
            cv2.putText(frame, f"{label} ({scores[i]:.2f})", (left, top - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_color, 2)
            if class_id in DANGEROUS_CLASSES:
                trigger_alert(label, scores[i], (left, top, right, bottom))
        
        # Face Detection and Recognition (every N frames)
        if frame_count % PROCESS_EVERY_N_FRAMES == 0:
            small_frame = cv2.resize(frame, (0, 0), fx=SCALE_FACTOR, fy=SCALE_FACTOR)
            rgb_small = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
            detected = detector.detect_faces(rgb_small)
            faces = [(int(v/SCALE_FACTOR) for v in face['box']) for face in detected]
            rects = [(x, y, x+w, y+h) for (x, y, w, h) in faces]
            pick = non_max_suppression(np.array(rects), overlapThresh=0.3)
            final_faces = [(x, y, ex-x, ey-y) for (x, y, ex, ey) in pick]
            
            encodings = []
            valid_faces = []
            for (x, y, w, h) in final_faces:
                try:
                    encoding = get_face_encoding(rgb_frame, (x, y, w, h))
                    encodings.append(encoding)
                    valid_faces.append((x, y, w, h))
                except:
                    continue
            
            current_ids = {}
            for face, encoding in zip(valid_faces, encodings):
                x, y, w, h = face
                best_match = None
                min_dist = TRACKING_THRESHOLD
                for fid, data in tracked_faces.items():
                    dist = np.linalg.norm(encoding - data['encoding'])
                    if dist < min_dist:
                        min_dist = dist
                        best_match = fid
                if best_match is not None:
                    current_ids[best_match] = {'box': face, 'encoding': encoding}
                else:
                    current_ids[next_face_id] = {'box': face, 'encoding': encoding}
                    next_face_id += 1
            tracked_faces = current_ids
        
        # Pose Estimation and Action Recognition
        for fid, data in tracked_faces.items():
            x, y, w, h = data['box']
            name = find_best_match(data['encoding'])
            roi = frame[y:y+h, x:x+w]
            roi_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
            mp_roi = mp.Image(image_format=mp.ImageFormat.SRGB, data=roi_rgb)
            pose_results = pose.process(mp_roi.numpy_view())
            
            if pose_results.pose_landmarks:
                mp_drawing.draw_landmarks(
                    frame[y:y+h, x:x+w], pose_results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2),
                    mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
                )
                action = get_action(pose_results.pose_landmarks.landmark, im_width, im_height)
                action_history.append(action)
                smoothed_action = max(set(action_history), key=lambda x: action_history.count(x))
                
                color = (0, 255, 0) if name != "Intruder" else (0, 0, 255)
                if "Potential Fall!" in smoothed_action:
                    color = (0, 0, 255)
                elif "Sitting" in smoothed_action:
                    color = (255, 165, 0)
                
                cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
                cv2.putText(frame, f"{name}: {smoothed_action}", (x, y-10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        # Display frame
        cv2.imshow('Child Safety Monitor', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()