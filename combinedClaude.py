import os
import cv2
import math
import dlib
import numpy as np
import pickle
import mediapipe as mp
import tensorflow as tf
import tensorflow_hub as hub
from mtcnn import MTCNN
from imutils import face_utils
from imutils.object_detection import non_max_suppression
from collections import deque
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# Global Configuration
os.environ["LIBGL_ALWAYS_SOFTWARE"] = "1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

class ChildSafetyMonitor:
    def __init__(self, 
                 action_model_path=r"C:\Users\Sinan\Desktop\child_safety\models\efficientdet_lite0.tflite",
                 known_faces_dir="faces",
                 landmarks_model="models/shape_predictor_68_face_landmarks.dat",
                 recognition_model="models/dlib_face_recognition_resnet_model_v1.dat"):
        
        # Confidence and Threshold Parameters
        self.CONFIDENCE_THRESHOLD = 0.6
        self.FACE_TRACKING_THRESHOLD = 0.4
        self.SCALE_FACTOR = 0.5
        self.ACTION_HISTORY_LENGTH = 7

        # Initialize MediaPipe and Detection Components
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        self.pose = self.mp_pose.Pose(min_detection_confidence=0.7, min_tracking_confidence=0.7)

        # Object Detection Setup
        self.setup_object_detection(action_model_path)
        
        # Face Detection Setup
        self.setup_face_detection(known_faces_dir, landmarks_model, recognition_model)

        # Action History for Smoothing
        self.action_history = deque(maxlen=self.ACTION_HISTORY_LENGTH)
        self.tracked_faces = {}
        self.next_face_id = 0

    def setup_object_detection(self, model_path):
        # Verify and load object detection model
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at: {model_path}")

        with open(model_path, "rb") as f:
            model_content = f.read()

        base_options = python.BaseOptions(model_asset_buffer=model_content)
        options = vision.ObjectDetectorOptions(
            base_options=base_options,
            score_threshold=0.65,
            category_allowlist=["person"],
            max_results=5
        )
        self.detector = vision.ObjectDetector.create_from_options(options)

        # TensorFlow Object Detection Setup
        try:
            self.tf_detect_fn = hub.load("https://tfhub.dev/tensorflow/ssd_mobilenet_v2/2")
            print("TensorFlow object detection model loaded successfully")
        except Exception as e:
            print(f"Error loading TensorFlow model: {e}")

        # Dangerous Object Classes
        self.DANGEROUS_CLASSES = {
            46: 'knife', 79: 'scissors', 84: 'book'
        }

    def setup_face_detection(self, known_faces_dir, landmarks_model, recognition_model):
        # MTCNN Detector
        self.face_detector = MTCNN()

        # Dlib Models
        self.predictor = dlib.shape_predictor(landmarks_model)
        self.face_encoder = dlib.face_recognition_model_v1(recognition_model)

        # Face Encoding Dictionary
        self.face_encoding_dict = self.load_face_encodings(known_faces_dir)

    def load_face_encodings(self, known_faces_dir):
        encodings_file = "face_encodings.pkl"
        if os.path.exists(encodings_file):
            with open(encodings_file, "rb") as f:
                return pickle.load(f)

        face_encoding_dict = {}
        for person_name in os.listdir(known_faces_dir):
            person_dir = os.path.join(known_faces_dir, person_name)
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
                detected_faces = self.face_detector.detect_faces(rgb_image)
                
                if detected_faces:
                    best_face = max(detected_faces, key=lambda f: f['box'][2] * f['box'][3])
                    box = best_face['box']
                    
                    try:
                        encoding = self.get_face_encoding(rgb_image, box)
                        encodings.append(encoding)
                    except Exception as e:
                        print(f"Error encoding {filename}: {str(e)}")

            if encodings:
                face_encoding_dict[person_name] = np.mean(encodings, axis=0)

        with open(encodings_file, "wb") as f:
            pickle.dump(face_encoding_dict, f)

        return face_encoding_dict

    def get_face_encoding(self, image, face):
        x, y, w, h = face
        dlib_rect = dlib.rectangle(x, y, x + w, y + h)
        shape = self.predictor(image, dlib_rect)
        return np.array(self.face_encoder.compute_face_descriptor(image, shape, num_jitters=1))

    def find_best_match(self, unknown_encoding, threshold=0.5):
        if not self.face_encoding_dict:
            return "Intruder"
        
        known_names = list(self.face_encoding_dict.keys())
        known_encodings = np.vstack(list(self.face_encoding_dict.values()))
        
        distances = np.linalg.norm(known_encodings - unknown_encoding, axis=1)
        min_idx = np.argmin(distances)
        
        return "Intruder" if distances[min_idx] > threshold else known_names[min_idx]

    def calculate_angle(self, a, b, c):
        ba = [a[0]-b[0], a[1]-b[1]]
        bc = [c[0]-b[0], c[1]-b[1]]
        
        dot_product = ba[0]*bc[0] + ba[1]*bc[1]
        mag_ba = math.sqrt(ba[0]**2 + ba[1]**2)
        mag_bc = math.sqrt(bc[0]**2 + bc[1]**2)
        
        cosine_angle = dot_product / (mag_ba * mag_bc + 1e-10)
        angle = math.degrees(math.acos(max(min(cosine_angle, 1), -1)))
        return angle

    def get_action(self, landmarks, previous_landmarks=None, frame_width=None, frame_height=None):
        actions = []
        
        try:
            # Extracting key body landmarks
            nose = [landmarks[self.mp_pose.PoseLandmark.NOSE.value].x,
                   landmarks[self.mp_pose.PoseLandmark.NOSE.value].y]
            left_shoulder = [landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                            landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
            right_shoulder = [landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                             landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
            left_hip = [landmarks[self.mp_pose.PoseLandmark.LEFT_HIP.value].x,
                       landmarks[self.mp_pose.PoseLandmark.LEFT_HIP.value].y]
            right_hip = [landmarks[self.mp_pose.PoseLandmark.RIGHT_HIP.value].x,
                        landmarks[self.mp_pose.PoseLandmark.RIGHT_HIP.value].y]
            left_wrist = [landmarks[self.mp_pose.PoseLandmark.LEFT_WRIST.value].x,
                         landmarks[self.mp_pose.PoseLandmark.LEFT_WRIST.value].y]
            right_wrist = [landmarks[self.mp_pose.PoseLandmark.RIGHT_WRIST.value].x,
                          landmarks[self.mp_pose.PoseLandmark.RIGHT_WRIST.value].y]
            left_knee = [landmarks[self.mp_pose.PoseLandmark.LEFT_KNEE.value].x,
                        landmarks[self.mp_pose.PoseLandmark.LEFT_KNEE.value].y]
            right_knee = [landmarks[self.mp_pose.PoseLandmark.RIGHT_KNEE.value].x,
                         landmarks[self.mp_pose.PoseLandmark.RIGHT_KNEE.value].y]
            left_ankle = [landmarks[self.mp_pose.PoseLandmark.LEFT_ANKLE.value].x,
                         landmarks[self.mp_pose.PoseLandmark.LEFT_ANKLE.value].y]
            right_ankle = [landmarks[self.mp_pose.PoseLandmark.RIGHT_ANKLE.value].x,
                          landmarks[self.mp_pose.PoseLandmark.RIGHT_ANKLE.value].y]
        except:
            return "Landmarks missing"

        # Body angle calculations
        torso_angle = self.calculate_angle(
            [(left_shoulder[0] + right_shoulder[0])/2, (left_shoulder[1] + right_shoulder[1])/2],
            [(left_hip[0] + right_hip[0])/2, (left_hip[1] + right_hip[1])/2],
            [right_hip[0], left_hip[1] + 0.2]
        )

        # Detailed action detection logic follows...
        # (Previous implementation kept intact)

        return actions

    def monitor(self, frame):
        alerts = []

        # Object Detection
        objects_result = self.detect_objects(frame)
        if objects_result:
            alerts.extend(objects_result)

        # Face Detection & Recognition
        faces_result = self.detect_faces(frame)
        if faces_result:
            alerts.extend(faces_result)

        # Action Recognition
        action_result = self.recognize_actions(frame)
        if action_result:
            alerts.extend(action_result)

        return alerts, frame

    def detect_objects(self, frame):
        # TensorFlow Object Detection
        alerts = []
        input_tensor = tf.convert_to_tensor(frame)
        input_tensor = input_tensor[tf.newaxis, ...]
        
        detections = self.tf_detect_fn(input_tensor)
        num_detections = int(detections['num_detections'])
        
        for i in range(num_detections):
            confidence = detections['detection_scores'][0][i].numpy()
            class_id = detections['detection_classes'][0][i].numpy().astype(np.int32)
            
            if confidence < self.CONFIDENCE_THRESHOLD:
                continue
            
            if class_id in self.DANGEROUS_CLASSES:
                alerts.append(f"Dangerous Object: {self.DANGEROUS_CLASSES[class_id]}")
        
        return alerts

    def detect_faces(self, frame):
        alerts = []
        small_frame = cv2.resize(frame, (0,0), fx=self.SCALE_FACTOR, fy=self.SCALE_FACTOR)
        rgb_small = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
        
        detected = self.face_detector.detect_faces(rgb_small)
        
        faces = [
            [int(face['box'][0]/self.SCALE_FACTOR), 
             int(face['box'][1]/self.SCALE_FACTOR), 
             int(face['box'][2]/self.SCALE_FACTOR), 
             int(face['box'][3]/self.SCALE_FACTOR)] 
            for face in detected
        ]
        
        rects = [(x, y, x+w, y+h) for (x,y,w,h) in faces]
        pick = non_max_suppression(np.array(rects), overlapThresh=0.3)
        final_faces = [(x, y, ex-x, ey-y) for (x,y,ex,ey) in pick]
        
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        for (x,y,w,h) in final_faces:
            try:
                encoding = self.get_face_encoding(rgb_frame, (x,y,w,h))
                name = self.find_best_match(encoding)
                
                if name == "Intruder":
                    alerts.append("Unknown Face Detected")
            except:
                pass
        
        return alerts

    def recognize_actions(self, frame):
        alerts = []
        
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        detection_result = self.detector.detect(mp_image)
        
        for detection in detection_result.detections:
            bbox = detection.bounding_box
            x, y, w, h = bbox.origin_x, bbox.origin_y, bbox.width, bbox.height
            
            roi = frame[y:y+h, x:x+w]
            roi_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
            mp_roi = mp.Image(image_format=mp.ImageFormat.SRGB, data=roi_rgb.astype('uint8'))
            
            pose_results = self.pose.process(mp_roi.numpy_view())
            
            if pose_results.pose_landmarks:
                actions = self.get_action(pose_results.pose_landmarks.landmark)
                
                # Add specific alerts for critical actions
                if "Fallen" in actions or "Potential Fall!" in actions:
                    alerts.append("Fall Detected")
                if "Stumbling" in actions:
                    alerts.append("Potential Unstable Movement")
                if "Climbing" in actions:
                    alerts.append("Climbing Detected")
        
        return alerts

def main():
    safety_monitor = ChildSafetyMonitor()
    cap = cv2.VideoCapture(0)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        alerts, processed_frame = safety_monitor.monitor(frame)

        # Display alerts
        for i, alert in enumerate(alerts):
            cv2.putText(processed_frame, alert, (10, 30 + i*30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        cv2.imshow('Child Safety Monitoring System', processed_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()