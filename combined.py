import os
import cv2
import numpy as np
import dlib
import pickle
import math
import mediapipe as mp
from mtcnn import MTCNN
from collections import deque
from imutils import face_utils
from imutils.object_detection import non_max_suppression
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# Suppress unnecessary warnings
os.environ["LIBGL_ALWAYS_SOFTWARE"] = "1"  # Suppress GPU warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"   # Reduce TensorFlow logging

# Initialize MTCNN detector
detectorf = MTCNN()

# Load dlib models
predictor = dlib.shape_predictor("models/shape_predictor_68_face_landmarks.dat")
face_encoder = dlib.face_recognition_model_v1("models/dlib_face_recognition_resnet_model_v1.dat")

# -------------------------
# Core Functions (Optimized)
# -------------------------

def get_face_encoding(image, face):
    """Optimized face encoding with reduced jitters"""
    x, y, w, h = face
    dlib_rect = dlib.rectangle(x, y, x + w, y + h)
    shape = predictor(image, dlib_rect)
    return np.array(face_encoder.compute_face_descriptor(image, shape, num_jitters=1))

def find_best_match(unknown_encoding, threshold=0.5):
    """Vectorized comparison using proper array stacking"""
    if not face_encoding_dict:
        return "Intruder"
    
    known_names = list(face_encoding_dict.keys())
    # Properly stack encodings into 2D array
    known_encodings = np.vstack(list(face_encoding_dict.values()))
    
    # Calculate distances efficiently
    distances = np.linalg.norm(known_encodings - unknown_encoding, axis=1)
    min_idx = np.argmin(distances)
    
    return "Intruder" if distances[min_idx] > threshold else known_names[min_idx]



# -------------------------
# Step 1: Load/Save Known Faces (Optimized)
# -------------------------

KNOWN_FACES_DIR = "faces"
ENCODINGS_FILE = "face_encodings.pkl"
face_encoding_dict = {}

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
            if image is None:
                continue

            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            detected_faces = detectorf.detect_faces(rgb_image)
            
            if detected_faces:
                best_face = max(detected_faces, key=lambda f: f['box'][2] * f['box'][3])
                box = best_face['box']
                
                try:
                    encoding = get_face_encoding(rgb_image, box)
                    encodings.append(encoding)
                except Exception as e:
                    print(f"Error encoding {filename}: {str(e)}")

        if encodings:
            # Store single average encoding per person
            face_encoding_dict[person_name] = np.mean(encodings, axis=0)

    with open(ENCODINGS_FILE, "wb") as f:
        pickle.dump(face_encoding_dict, f)
    print(f"Saved {len(face_encoding_dict)} encodings")


# Initialize MediaPipe components
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose(min_detection_confidence=0.7, min_tracking_confidence=0.7)

# Initialize Object Detector
model_path = "C:\\Users\\kenjo\\Documents\\GitHub\\Child-Safety-System\\models\\efficientdet_lite0.tflite"
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model file not found at: {model_path}")

# Load the model file as a byte buffer
with open(model_path, "rb") as f:
    model_content = f.read()

# Pass the model as a buffer instead of a path
base_options = python.BaseOptions(model_asset_buffer=model_content)
options = vision.ObjectDetectorOptions(
    base_options=base_options,
    score_threshold=0.65,
    category_allowlist=["person"],
    max_results=5  # Limit to 5 people for better performance
)
detector = vision.ObjectDetector.create_from_options(options)

# Action smoothing parameters
ACTION_HISTORY_LENGTH = 7
action_history = deque(maxlen=ACTION_HISTORY_LENGTH)

def calculate_angle(a, b, c):
    """Calculate the angle between three points using vector math"""
    ba = [a[0]-b[0], a[1]-b[1]]
    bc = [c[0]-b[0], c[1]-b[1]]
    
    dot_product = ba[0]*bc[0] + ba[1]*bc[1]
    mag_ba = math.sqrt(ba[0]**2 + ba[1]**2)
    mag_bc = math.sqrt(bc[0]**2 + bc[1]**2)
    
    cosine_angle = dot_product / (mag_ba * mag_bc + 1e-10)  # Avoid division by zero
    angle = math.degrees(math.acos(max(min(cosine_angle, 1), -1)))
    return angle

def get_action(landmarks, previous_landmarks=None, frame_width=None, frame_height=None):
    """Determine actions with improved detection for child safety monitoring"""
    actions = []
    
    # Key landmarks
    try:
        # Head landmarks
        nose = [landmarks[mp_pose.PoseLandmark.NOSE.value].x,
               landmarks[mp_pose.PoseLandmark.NOSE.value].y]
        left_ear = [landmarks[mp_pose.PoseLandmark.LEFT_EAR.value].x,
                   landmarks[mp_pose.PoseLandmark.LEFT_EAR.value].y]
        right_ear = [landmarks[mp_pose.PoseLandmark.RIGHT_EAR.value].x,
                    landmarks[mp_pose.PoseLandmark.RIGHT_EAR.value].y]
        
        # Torso landmarks
        left_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                        landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
        right_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                         landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
        left_hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,
                   landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
        right_hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,
                    landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
        
        # Arm landmarks
        left_elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
                     landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
        right_elbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,
                      landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
        left_wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,
                     landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
        right_wrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,
                      landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]
        
        # Leg landmarks
        left_knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,
                    landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
        right_knee = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x,
                     landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y]
        left_ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x,
                     landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]
        right_ankle = [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x,
                      landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y]
    except:
        return "Landmarks missing"

    # Calculate body angles
    left_knee_angle = calculate_angle(left_hip, left_knee, left_ankle)
    right_knee_angle = calculate_angle(right_hip, right_knee, right_ankle)
    avg_knee_angle = (left_knee_angle + right_knee_angle) / 2
    
    torso_angle = calculate_angle(
        [(left_shoulder[0] + right_shoulder[0])/2, (left_shoulder[1] + right_shoulder[1])/2],
        [(left_hip[0] + right_hip[0])/2, (left_hip[1] + right_hip[1])/2],
        [right_hip[0], left_hip[1] + 0.2]  # Point below hips
    )
    
    # Calculate average positions
    shoulder_avg_y = (left_shoulder[1] + right_shoulder[1]) / 2
    hip_avg_y = (left_hip[1] + right_hip[1]) / 2
    ankle_avg_y = (left_ankle[1] + right_ankle[1]) / 2
    wrist_avg_y = (left_wrist[1] + right_wrist[1]) / 2
    knee_avg_y = (left_knee[1] + right_knee[1]) / 2
    
    # Body alignment
    is_vertical = torso_angle > 70
    is_horizontal = torso_angle < 30
    
    # Standing/sitting detection (refined)
    if avg_knee_angle < 110:
        if wrist_avg_y > knee_avg_y and knee_avg_y > ankle_avg_y:
            actions.append("Crawling")
        else:
            actions.append("Sitting")
    elif avg_knee_angle > 160 and is_vertical:
        actions.append("Standing")
    
    # Laying down detection
    if is_horizontal and hip_avg_y > shoulder_avg_y - 0.05:
        actions.append("Laying Down")
    
    # Jumping detection
    if previous_landmarks and frame_height:
        prev_hip_avg_y = (previous_landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y + 
                          previous_landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y) / 2
        
        hip_y_change = (hip_avg_y - prev_hip_avg_y) * frame_height
        
        # Upward movement
        if hip_y_change < -15:  # Moving up more than 15 pixels
            actions.append("Jumping Up")
        # Downward movement (landing)
        elif hip_y_change > 15 and "Jumping Up" in action_history:
            actions.append("Landing")
    
    # Running detection
    if previous_landmarks and is_vertical:
        prev_ankle_dist = abs(previous_landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x - 
                             previous_landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x)
        current_ankle_dist = abs(left_ankle[0] - right_ankle[0])
        
        ankle_change = abs(current_ankle_dist - prev_ankle_dist)
        
        if ankle_change > 0.05 and avg_knee_angle > 130:
            actions.append("Running")
        elif ankle_change > 0.02 and avg_knee_angle > 130:
            actions.append("Walking")
    
    # Hand position detection
    if left_wrist[1] < shoulder_avg_y - 0.15:
        if left_wrist[1] < nose[1]:
            actions.append("Left Hand High")
        else:
            actions.append("Left Hand Up")
            
    if right_wrist[1] < shoulder_avg_y - 0.15:
        if right_wrist[1] < nose[1]:
            actions.append("Right Hand High")
        else:
            actions.append("Right Hand Up")
    
    # Reaching detection
    arm_extension = max(
        calculate_angle(left_shoulder, left_elbow, left_wrist),
        calculate_angle(right_shoulder, right_elbow, right_wrist)
    )
    if arm_extension > 150:
        actions.append("Reaching")
    
    # Climbing detection
    hands_high = left_wrist[1] < shoulder_avg_y - 0.1 and right_wrist[1] < shoulder_avg_y - 0.1
    legs_bent = avg_knee_angle < 140
    if hands_high and legs_bent and is_vertical:
        actions.append("Climbing")
    
    # Fall detection (enhanced)
    if nose[1] > hip_avg_y + 0.15:
        if is_horizontal:
            actions.append("Fallen")
        else:
            actions.append("Potential Fall!")
    
    # Stumbling detection
    if is_vertical and (torso_angle < 70 and torso_angle > 45):
        if left_wrist[1] > shoulder_avg_y or right_wrist[1] > shoulder_avg_y:
            actions.append("Stumbling")
    
    # Tantrum detection - high motion variability in multiple limbs
    if previous_landmarks:
        limb_movement = sum([
            abs(left_wrist[0] - previous_landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x),
            abs(left_wrist[1] - previous_landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y),
            abs(right_wrist[0] - previous_landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x),
            abs(right_wrist[1] - previous_landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y),
            abs(left_ankle[0] - previous_landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x),
            abs(left_ankle[1] - previous_landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y),
            abs(right_ankle[0] - previous_landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x),
            abs(right_ankle[1] - previous_landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y)
        ])
        
        if limb_movement > 0.15:
            actions.append("High Activity")
            if len(action_history) > 3 and action_history.count("High Activity") > 3:
                actions.append("Possible Tantrum")
    
    # Idle/Still detection
    if previous_landmarks:
        total_movement = sum([
            abs(nose[0] - previous_landmarks[mp_pose.PoseLandmark.NOSE.value].x),
            abs(nose[1] - previous_landmarks[mp_pose.PoseLandmark.NOSE.value].y),
            abs(left_wrist[0] - previous_landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x),
            abs(left_wrist[1] - previous_landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y),
            abs(right_wrist[0] - previous_landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x),
            abs(right_wrist[1] - previous_landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y)
        ])
        
        if total_movement < 0.01:
            actions.append("Still")
    
    return ", ".join(actions) if actions else "Neutral"

# cap = cv2.VideoCapture("/mnt/c/Users/Sinan/Desktop/new childsafety/test/test.mp4")
VIDEO_PATH = 0
PROCESS_EVERY_N_FRAMES = 2  # Process every other frame
SCALE_FACTOR = 0.5  # Reduced resolution for detection
TRACKING_THRESHOLD = 0.4  # Lower threshold for tracking

cap = cv2.VideoCapture(0)
tracked_faces = {}
next_face_id = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    frame_number = cap.get(cv2.CAP_PROP_POS_FRAMES)
    if frame_number % PROCESS_EVERY_N_FRAMES != 0:
        continue

    # Optimized face detection with scaled frame
    small_frame = cv2.resize(frame, (0,0), fx=SCALE_FACTOR, fy=SCALE_FACTOR)
    rgb_small = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
    detected = detectorf.detect_faces(rgb_small)

    # Scale boxes back to original size
    faces = []
    for face in detected:
        x, y, w, h = [int(v/SCALE_FACTOR) for v in face['box']]
        faces.append((x, y, w, h))

    # Non-max suppression on original coordinates
    rects = [(x, y, x+w, y+h) for (x,y,w,h) in faces]
    pick = non_max_suppression(np.array(rects), overlapThresh=0.3)
    final_faces = [(x, y, ex-x, ey-y) for (x,y,ex,ey) in pick]

    # Batch encoding computation
    encodings = []
    valid_faces = []
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    for (x,y,w,h) in final_faces:
        try:
            encodings.append(get_face_encoding(rgb_frame, (x,y,w,h)))
            valid_faces.append((x,y,w,h))
        except:
            continue

    # Vectorized duplicate removal
    if len(encodings) > 1:
        enc_array = np.array(encodings)
        dist_matrix = np.linalg.norm(enc_array[:, None] - enc_array, axis=2)
        mask = np.ones(len(encodings), dtype=bool)
        for i in range(len(encodings)):
            if mask[i]:
                mask[np.where(dist_matrix[i] < TRACKING_THRESHOLD)] = False
                mask[i] = True
        encodings = [enc for enc, m in zip(encodings, mask) if m]
        final_faces = [face for face, m in zip(valid_faces, mask) if m]

    # Optimized tracking with combined checks
    current_ids = {}
    for face, encoding in zip(final_faces, encodings):
        x,y,w,h = face
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

    # Person detection
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
    detection_result = detector.detect(mp_image)

    # Process each detection
    for detection in detection_result.detections:
        bbox = detection.bounding_box
        x, y, w, h = bbox.origin_x, bbox.origin_y, bbox.width, bbox.height
        roi = frame[y:y+h, x:x+w]

        # Default color
        color = (0, 255, 0)  # Green

        # Pose estimation
        roi_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
        mp_roi = mp.Image(image_format=mp.ImageFormat.SRGB, data=roi_rgb.astype('uint8'))
        pose_results = pose.process(mp_roi.numpy_view())

        if pose_results.pose_landmarks:
            # Draw landmarks
            mp_drawing.draw_landmarks(
                frame[y:y+h, x:x+w],
                pose_results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2),
                mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
            )

            # Action detection and smoothing
            action = get_action(pose_results.pose_landmarks.landmark)
            action_history.append(action)
            smoothed_action = max(set(action_history), key=lambda x: action_history.count(x))

            # Determine text color
            if "Potential Fall!" in smoothed_action:
                color = (0, 0, 255)  # Red
            elif "Sitting" in smoothed_action:
                color = (255, 165, 0)  # Orange
            else:
                color = (0, 255, 0)  # Green

            # Display action
            cv2.putText(frame, f"{smoothed_action}", (x, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    # Update tracked faces and draw results
    tracked_faces = current_ids
    for fid, data in tracked_faces.items():
        x,y,w,h = data['box']
        name = find_best_match(data['encoding'])
        color = (0,255,0) if name != "Intruder" else (0,0,255)
        cv2.rectangle(frame, (x,y), (x+w,y+h), color, 2)
        cv2.putText(frame, name, (x+5,y+h-5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

        # Draw bounding box
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

    cv2.imshow('Advanced Action Recognition', frame)
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
