import os
import cv2
import numpy as np
import dlib
import pickle
from mtcnn import MTCNN
from imutils import face_utils
from imutils.object_detection import non_max_suppression

# Initialize MTCNN detector
detector = MTCNN()

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
            detected_faces = detector.detect_faces(rgb_image)
            
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

# -------------------------
# Step 2: Optimized Video Processing
# -------------------------

# VIDEO_PATH = r"C:\Users\Sinan\Desktop\new_childsafety\test\test.mp4"
VIDEO_PATH = 0
PROCESS_EVERY_N_FRAMES = 1  # Process every other frame
SCALE_FACTOR = 0.5  # Reduced resolution for detection
TRACKING_THRESHOLD = 0.4  # Lower threshold for tracking

video_capture = cv2.VideoCapture(VIDEO_PATH)
tracked_faces = {}
next_face_id = 0

while True:
    ret, frame = video_capture.read()
    if not ret:
        break

    frame_number = video_capture.get(cv2.CAP_PROP_POS_FRAMES)
    if frame_number % PROCESS_EVERY_N_FRAMES != 0:
        continue

    # Optimized face detection with scaled frame
    small_frame = cv2.resize(frame, (0,0), fx=SCALE_FACTOR, fy=SCALE_FACTOR)
    rgb_small = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
    detected = detector.detect_faces(rgb_small)

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

    # Update tracked faces and draw results
    tracked_faces = current_ids
    for fid, data in tracked_faces.items():
        x,y,w,h = data['box']
        name = find_best_match(data['encoding'])
        color = (0,255,0) if name != "Intruder" else (0,0,255)
        cv2.rectangle(frame, (x,y), (x+w,y+h), color, 2)
        cv2.putText(frame, name, (x+5,y+h-5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

    cv2.imshow('Optimized Recognition', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()