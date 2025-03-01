import os
import cv2
import numpy as np
import math
import mediapipe as mp
from collections import deque
import onnxruntime as ort
import pickle
from insightface.app import FaceAnalysis
from pathlib import Path
import yaml
from typing import Dict, List, Tuple, Optional, Any, Union
import logging
import sys  # Add sys import for stderr output

# Suppress unnecessary warnings
os.environ["LIBGL_ALWAYS_SOFTWARE"] = "1" # Suppress GPU warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # Reduce TensorFlow logging

# --- ConfigManager Class ---
class ConfigManager:
    """Handles configuration loading and management for the application."""

    DEFAULT_CONFIG = {
        "detection": {
            "confidence_threshold": 0.3,
            "tracking_threshold": 0.4,
            "process_every_n_frames": 2,
            "scale_factor": 0.5,
            "action_history_length": 7
        },
        "paths": {
            "models": {
                "yolo": "models/yolov8n.onnx",
            },
            "faces_dir": "faces",
            "encodings_file": "face_encodings.pkl"
        },
        "classes": {
            "coco_labels": {1: "person", 46: "knife", 79: "scissors"},
            "dangerous_classes": [46, 79]
        },
        "logging": {
            "level": "INFO",
            "file": "safety_monitor.log"
        }
    }

    def __init__(self, config_path: str = "config.yaml"):
        self.config_path = config_path
        self.config = self._load_config()

    def _load_config(self) -> Dict:
        config = self.DEFAULT_CONFIG.copy()
        if os.path.exists(self.config_path):
            try:
                with open(self.config_path, "r") as f:
                    loaded_config = yaml.safe_load(f)
                    if loaded_config:
                        self._update_dict(config, loaded_config)
            except Exception as e:
                logging.warning(f"Failed to load config from {self.config_path}: {e}")
                logging.warning("Using default configuration")
        else:
            logging.info(f"Config file not found at {self.config_path}, using defaults")
            try:
                os.makedirs(os.path.dirname(self.config_path), exist_ok=True)  # Create directory if needed
                with open(self.config_path, "w") as f:
                    yaml.dump(config, f, default_flow_style=False)
                logging.info(f"Default configuration saved to {self.config_path}")
            except Exception as e:
                logging.warning(f"Failed to save default config: {e}")
        return config

    def _update_dict(self, target: Dict, source: Dict) -> None:
        for key, value in source.items():
            if key in target and isinstance(target[key], dict) and isinstance(value, dict):
                self._update_dict(target[key], value)
            else:
                target[key] = value

    def get(self, *keys: str, default: Any = None) -> Any:
        result = self.config
        try:
            for key in keys:
                result = result[key]
            return result
        except (KeyError, TypeError):
            return default

# --- FaceManager Class with InsightFace ---
class FaceManager:
    """Handles face recognition using InsightFace for improved accuracy."""

    def __init__(self, config: ConfigManager):
        self.config = config
        self.known_faces_dir = config.get("paths", "faces_dir")
        self.encodings_file = config.get("paths", "encodings_file")
        self.tracking_threshold = config.get("detection", "tracking_threshold")

        # Initialize InsightFace for face detection and recognition
        try:
            self.app = FaceAnalysis()
            self.app.prepare(ctx_id=0, det_size=(640, 640))
            logging.info("InsightFace initialized successfully")
        except Exception as e:
            logging.error(f"Failed to initialize InsightFace: {e}")
            self.app = None  # Set to None if initialization fails

        # Load or create face encodings
        self.face_encoding_dict = self._load_face_encodings()

        # Tracking state
        self.tracked_faces = {}
        self.next_face_id = 0

    def _load_face_encodings(self) -> Dict:
        encodings_dict = {}
        if self.app is None:  # Skip if InsightFace failed to initialize
            return encodings_dict
            
        faces_dir = Path(self.known_faces_dir)
        if not faces_dir.exists():
            logging.warning(f"Faces directory not found: {faces_dir}")
            try:
                os.makedirs(faces_dir, exist_ok=True)
                logging.info(f"Created faces directory: {faces_dir}")
            except Exception as e:
                logging.error(f"Failed to create faces directory: {e}")
            return encodings_dict
            
        for person_dir in faces_dir.iterdir():
            if not person_dir.is_dir():
                continue
            person_name = person_dir.name
            embeddings = []
            for img_path in person_dir.glob("*.{jpg,jpeg,png}"):
                try:
                    image = cv2.imread(str(img_path))
                    if image is None:
                        logging.warning(f"Failed to load image: {img_path}")
                        continue
                    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    faces = self.app.get(rgb_image)
                    if faces:
                        embedding = faces[0].embedding
                        embeddings.append(embedding)
                except Exception as e:
                    logging.error(f"Error processing {img_path}: {e}")
            if embeddings:
                encodings_dict[person_name] = np.mean(embeddings, axis=0)
        # Save encodings
        try:
            os.makedirs(os.path.dirname(self.encodings_file), exist_ok=True)
            with open(self.encodings_file, "wb") as f:
                pickle.dump(encodings_dict, f)
        except Exception as e:
            logging.error(f"Failed to save face encodings: {e}")
        return encodings_dict

    def find_best_match(self, unknown_embedding: np.ndarray) -> str:
        if not self.face_encoding_dict:
            return "Intruder"
        try:
            known_names = list(self.face_encoding_dict.keys())
            known_embeddings = np.vstack(list(self.face_encoding_dict.values()))
            similarities = np.dot(known_embeddings, unknown_embedding) / (
                np.linalg.norm(known_embeddings, axis=1) * np.linalg.norm(unknown_embedding)
            )
            max_sim = np.max(similarities)
            if max_sim > self.tracking_threshold:
                return known_names[np.argmax(similarities)]
            return "Intruder"
        except Exception as e:
            logging.error(f"Error in face matching: {e}")
            return "Intruder"

    def process_faces(self, frame: np.ndarray) -> np.ndarray:
        if self.app is None:  # Skip if InsightFace failed to initialize
            cv2.putText(frame, "Face recognition unavailable", (10, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            return frame
            
        try:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            faces = self.app.get(rgb_frame)
            current_ids = {}
            for face in faces:
                bbox = face.bbox.astype(int)
                x1, y1, x2, y2 = bbox
                embedding = face.embedding
                name = self.find_best_match(embedding)
                # Tracking
                best_match = None
                min_dist = 1 - self.tracking_threshold
                for fid, data in self.tracked_faces.items():
                    dist = 1 - np.dot(data['embedding'], embedding) / (
                        np.linalg.norm(data['embedding']) * np.linalg.norm(embedding)
                    )
                    if dist < min_dist:
                        min_dist = dist
                        best_match = fid
                if best_match is not None:
                    current_ids[best_match] = {'box': bbox, 'embedding': embedding}
                    track_id = best_match
                else:
                    current_ids[self.next_face_id] = {'box': bbox, 'embedding': embedding}
                    track_id = self.next_face_id
                    self.next_face_id += 1
                # Draw annotations
                color = (0, 255, 0) if name != "Intruder" else (0, 0, 255)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, f"{name} [ID:{track_id}]", (x1+5, y2-5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            self.tracked_faces = current_ids
            return frame
        except Exception as e:
            logging.error(f"Error in face processing: {e}")
            cv2.putText(frame, "Face processing error", (10, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            return frame

# --- PoseAnalyzer Class with Enhanced Fall Detection ---
class PoseAnalyzer:
    """Analyzes human poses with improved action and fall detection."""

    def __init__(self, config: ConfigManager):
        self.config = config
        self.action_history_length = config.get("detection", "action_history_length")
        self.action_history = deque(maxlen=self.action_history_length)
        self.previous_landmarks = None
        try:
            self.mp_pose = mp.solutions.pose
            self.mp_drawing = mp.solutions.drawing_utils
            self.pose = self.mp_pose.Pose(min_detection_confidence=0.7, min_tracking_confidence=0.7)
            logging.info("MediaPipe Pose initialized successfully")
        except Exception as e:
            logging.error(f"Failed to initialize MediaPipe Pose: {e}")
            self.mp_pose = None
            self.mp_drawing = None
            self.pose = None

    def calculate_angle(self, a: List[float], b: List[float], c: List[float]) -> float:
        ba = [a[0]-b[0], a[1]-b[1]]
        bc = [c[0]-b[0], c[1]-b[1]]
        dot_product = ba[0]*bc[0] + ba[1]*bc[1]
        mag_ba = math.sqrt(ba[0]**2 + ba[1]**2)
        mag_bc = math.sqrt(bc[0]**2 + bc[1]**2)
        cosine_angle = dot_product / (mag_ba * mag_bc + 1e-10)
        return math.degrees(math.acos(max(min(cosine_angle, 1), -1)))

    def get_action(self, landmarks: List, frame_width: Optional[int] = None,
                   frame_height: Optional[int] = None) -> str:
        try:
            # Helper function to get landmarks with visibility check
            def get_landmark(lm):
                return [lm.x, lm.y] if lm.visibility > 0.5 else None

            nose = get_landmark(landmarks[self.mp_pose.PoseLandmark.NOSE])
            left_shoulder = get_landmark(landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER])
            right_shoulder = get_landmark(landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER])
            left_hip = get_landmark(landmarks[self.mp_pose.PoseLandmark.LEFT_HIP])
            right_hip = get_landmark(landmarks[self.mp_pose.PoseLandmark.RIGHT_HIP])
            left_knee = get_landmark(landmarks[self.mp_pose.PoseLandmark.LEFT_KNEE])
            right_knee = get_landmark(landmarks[self.mp_pose.PoseLandmark.RIGHT_KNEE])
            left_ankle = get_landmark(landmarks[self.mp_pose.PoseLandmark.LEFT_ANKLE])
            right_ankle = get_landmark(landmarks[self.mp_pose.PoseLandmark.RIGHT_ANKLE])

            # Calculate angles
            left_knee_angle = self.calculate_angle(left_hip, left_knee, left_ankle) if all([left_hip, left_knee, left_ankle]) else None
            right_knee_angle = self.calculate_angle(right_hip, right_knee, right_ankle) if all([right_hip, right_knee, right_ankle]) else None
            avg_knee_angle = ((left_knee_angle + right_knee_angle) / 2 if left_knee_angle and right_knee_angle else
                                 left_knee_angle or right_knee_angle or None)

            if all([left_shoulder, right_shoulder, left_hip, right_hip]):
                shoulder_mid = [(left_shoulder[0] + right_shoulder[0])/2, (left_shoulder[1] + right_shoulder[1])/2]
                hip_mid = [(left_hip[0] + right_hip[0])/2, (left_hip[1] + right_hip[1])/2]
                vertical = [hip_mid[0], hip_mid[1] - 0.1]
                torso_angle = self.calculate_angle(shoulder_mid, hip_mid, vertical)
            else:
                torso_angle = None

            # Determine actions
            actions = []
            if avg_knee_angle:
                if avg_knee_angle < 110:
                    actions.append("Sitting")
                elif avg_knee_angle > 160 and torso_angle and torso_angle > 70:
                    actions.append("Standing")
            if torso_angle and torso_angle < 30 and left_hip and right_hip and left_shoulder and right_shoulder:
                hip_avg_y = (left_hip[1] + right_hip[1]) / 2
                shoulder_avg_y = (left_shoulder[1] + right_shoulder[1]) / 2
                if hip_avg_y > shoulder_avg_y - 0.05:
                    actions.append("Laying Down")

            # Enhanced fall detection with velocity
            if (self.previous_landmarks and nose and left_shoulder and right_shoulder and
                    all(landmarks[lm].visibility > 0.5 and self.previous_landmarks[lm].visibility > 0.5
                        for lm in [self.mp_pose.PoseLandmark.NOSE, self.mp_pose.PoseLandmark.LEFT_SHOULDER, self.mp_pose.PoseLandmark.RIGHT_SHOULDER])):
                vel_nose = nose[1] - self.previous_landmarks[self.mp_pose.PoseLandmark.NOSE].y
                vel_left_shoulder = left_shoulder[1] - self.previous_landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER].y
                vel_right_shoulder = right_shoulder[1] - self.previous_landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER].y
                avg_vel = (vel_nose + vel_left_shoulder + vel_right_shoulder) / 3
                if avg_vel > 0.03 and "Laying Down" in actions:
                    actions.append("Potential Fall!")

            return ", ".join(actions) if actions else "Neutral"
        except Exception as e:
            logging.error(f"Error in pose analysis: {e}")
            return "Analysis Error"

    def process_pose(self, frame: np.ndarray) -> Tuple[np.ndarray, Optional[str]]:
        if self.pose is None:  # Skip if MediaPipe Pose failed to initialize
            cv2.putText(frame, "Pose analysis unavailable", (10, 90), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            return frame, None
            
        try:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            pose_results = self.pose.process(mp_image.numpy_view())
            if pose_results.pose_landmarks:
                self.mp_drawing.draw_landmarks(frame, pose_results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS)
                action = self.get_action(pose_results.pose_landmarks.landmark)
                self.action_history.append(action)
                if len(self.action_history) >= 3:
                    smoothed_action = max(set(self.action_history), key=lambda x: self.action_history.count(x))
                    color = (0, 0, 255) if "Potential Fall!" in smoothed_action else (0, 255, 0)
                    cv2.putText(frame, smoothed_action, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                    self.previous_landmarks = pose_results.pose_landmarks.landmark
                    return frame, smoothed_action
            return frame, None
        except Exception as e:
            logging.error(f"Error in pose processing: {e}")
            cv2.putText(frame, "Pose processing error", (10, 90), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            return frame, None

# --- ObjectDetector Class ---
class ObjectDetector:
    """Detects objects in video frames using YOLOv8."""

    def __init__(self, config: ConfigManager):
        self.config = config
        self.confidence_threshold = config.get("detection", "confidence_threshold")
        self.coco_labels = config.get("classes", "coco_labels")
        self.dangerous_classes = config.get("classes", "dangerous_classes")
        yolo_path = config.get("paths", "models", "yolo")
        
        try:
            if not os.path.exists(yolo_path):
                logging.error(f"YOLO model not found at {yolo_path}")
                self.session = None
            else:
                self.session = ort.InferenceSession(yolo_path)
                self.input_name = self.session.get_inputs()[0].name
                logging.info("YOLO model loaded successfully")
        except Exception as e:
            logging.error(f"Failed to load YOLO model: {e}")
            self.session = None
            
        self.alert_callback = None

    def set_alert_callback(self, callback):
        self.alert_callback = callback

    def detect_objects(self, frame: np.ndarray) -> np.ndarray:
        if self.session is None:  # Skip if YOLO model failed to load
            cv2.putText(frame, "Object detection unavailable", (10, 120), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            return frame
            
        try:
            frame_height, frame_width = frame.shape[:2]
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            target_size = 640
            scale = min(target_size / frame_height, target_size / frame_width)
            resized_frame = cv2.resize(rgb_frame, (int(frame_width * scale), int(frame_height * scale)))
            padded_frame = np.zeros((target_size, target_size, 3), dtype=np.uint8)
            dx = (target_size - int(frame_width * scale)) // 2
            dy = (target_size - int(frame_height * scale)) // 2
            padded_frame[dy:dy + int(frame_height * scale), dx:dx + int(frame_width * scale)] = resized_frame
            input_image = padded_frame.astype(np.float32) / 255.0
            input_tensor = np.expand_dims(input_image, axis=0)
            input_tensor = np.transpose(input_tensor, (0, 3, 1, 2))
            outputs = self.session.run(None, {self.input_name: input_tensor})
            detections = outputs[0][0]
            for detection in detections:
                x1, y1, x2, y2, conf, cls, *_ = detection
                if conf < self.confidence_threshold:
                    continue
                class_id = int(cls)
                label = self.coco_labels.get(class_id, f"class_{class_id}")
                x1_orig = max(0, int((x1 - dx) / scale))
                y1_orig = max(0, int((y1 - dy) / scale))
                x2_orig = min(frame_width, int((x2 - dx) / scale))
                y2_orig = min(frame_height, int((y2 - dy) / scale))
                is_dangerous = class_id in self.dangerous_classes
                color = (0, 0, 255) if is_dangerous else (0, 255, 0)
                cv2.rectangle(frame, (x1_orig, y1_orig), (x2_orig, y2_orig), color, 2)
                cv2.putText(frame, f"{label} ({conf:.2f})", (x1_orig, y1_orig - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                if is_dangerous and self.alert_callback:
                    self.alert_callback(label, float(conf), (x1_orig, y1_orig, x2_orig, y2_orig))
            return frame
        except Exception as e:
            logging.error(f"Error in object detection: {e}")
            cv2.putText(frame, "Object detection error", (10, 120), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            return frame

# --- SafetyMonitor Class ---
class SafetyMonitor:
    """Main class for child safety monitoring system."""

    def __init__(self, config_path: str = "config.yaml"):
        # Set up logging before anything else
        self.config = ConfigManager(config_path)
        log_level = getattr(logging, self.config.get("logging", "level", default="INFO"))
        log_file = self.config.get("logging", "file", default="safety_monitor.log")
        
        # Configure logging to output to console as well
        logging.basicConfig(
            level=log_level,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file), 
                logging.StreamHandler(sys.stdout)  # Explicitly output to stdout
            ]
        )
        
        print("Starting Child Safety Monitoring System")  # Direct console output
        logging.info("Initializing Child Safety Monitoring System")
        
        # Initialize components with better error handling
        try:
            self.face_manager = FaceManager(self.config)
        except Exception as e:
            logging.error(f"Failed to initialize FaceManager: {e}")
            self.face_manager = None
            
        try:
            self.pose_analyzer = PoseAnalyzer(self.config)
        except Exception as e:
            logging.error(f"Failed to initialize PoseAnalyzer: {e}")
            self.pose_analyzer = None
            
        try:
            self.object_detector = ObjectDetector(self.config)
            self.object_detector.set_alert_callback(self.trigger_alert)
        except Exception as e:
            logging.error(f"Failed to initialize ObjectDetector: {e}")
            self.object_detector = None
            
        self.cap = None
        self.processing_frame = 0
        self.test_mode = False  # Enable test mode by default for debugging

    def trigger_alert(self, label: str, confidence: float, bbox: Tuple[int, int, int, int]) -> None:
        logging.warning(f"ALERT: Dangerous item '{label}' detected with confidence {confidence:.2f}")
        print(f"ALERT: Dangerous item '{label}' detected")  # Direct console output

    def start_capture(self, source: Union[int, str] = 0) -> bool:
        try:
            # Convert string to integer for camera indices
            if isinstance(source, str) and source.isdigit():
                source = int(source)
                
            print(f"Attempting to open video source: {source}")  # Direct console output
            logging.info(f"Attempting to open video source: {source}")
            
            # Try to open with timeout
            self.cap = cv2.VideoCapture(source)
            if isinstance(source, str):
                if source.isdigit():
                    source = int(source)
                elif not os.path.exists(source):
                    logging.error(f"Video file not found: {source}")
                    return False

            # Additional checks for successful capture
            if not self.cap.isOpened():
                print(f"Failed to open video source: {source}")  # Direct console output
                logging.error(f"Failed to open video source: {source}")
                
                # Check if we need to try DirectShow on Windows
                if isinstance(source, int) and os.name == 'nt':
                    print("Trying with DirectShow API...")  # Direct console output
                    self.cap = cv2.VideoCapture(source, cv2.CAP_DSHOW)
                    if not self.cap.isOpened():
                        print("DirectShow also failed")  # Direct console output
                        return False
                    print("DirectShow connection successful")  # Direct console output
                else:
                    return False
            
            # Verify we can actually get a frame
            ret, test_frame = self.cap.read()
            if not ret or test_frame is None:
                print("Camera opened but couldn't read frame")  # Direct console output
                logging.error("Camera opened but couldn't read frame")
                return False
                
            print(f"Capture started successfully, frame size: {test_frame.shape[:2]}")  # Direct console output
            logging.info(f"Capture started successfully, frame size: {test_frame.shape[:2]}")
            return True
        except Exception as e:
            print(f"Error starting capture: {e}")  # Direct console output
            logging.error(f"Error starting capture: {e}")
            return False

    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        try:
            process_every_n = self.config.get("detection", "process_every_n_frames")
            self.processing_frame += 1
            
            # Add a basic timestamp and status
            cv2.putText(frame, f"Frame: {self.processing_frame}", (10, frame.shape[0] - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Test mode just displays the camera feed with minimal processing
            if self.test_mode:
                cv2.putText(frame, "TEST MODE - Basic camera feed only", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                return frame
                
            # Only do full processing if not in test mode
            if self.object_detector:
                frame = self.object_detector.detect_objects(frame)
                
            if self.pose_analyzer:
                frame, action = self.pose_analyzer.process_pose(frame)
                
            if self.processing_frame % process_every_n == 0 and self.face_manager:
                frame = self.face_manager.process_faces(frame)

            return frame

        except Exception as e:
            logging.error(f"Error processing frame: {e}")
            # Return frame with error message
            cv2.putText(frame, f"Processing error: {str(e)[:50]}", (10, 150),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            return frame

    def run(self) -> None:
        if self.cap is None or not self.cap.isOpened():
            print("Failed to initialize video capture")
            return

        # Create window first to ensure GUI is available
        cv2.namedWindow('Child Safety Monitoring', cv2.WINDOW_NORMAL)
        if cv2.getWindowProperty('Child Safety Monitoring', cv2.WND_PROP_VISIBLE) < 1:
            print("ERROR: Failed to create OpenCV window - no GUI available?")
            return

        print("Starting video processing loop")
        logging.info("Starting video processing loop")

        try:
            while True:
                ret, frame = self.cap.read()
                if not ret or frame is None:
                    logging.error("Failed to capture frame")
                    break

                processed_frame = self.process_frame(frame)
                
                # Show processed frame
                cv2.imshow('Child Safety Monitoring', processed_frame)
                
                # Check for exit key with proper delay
                key = cv2.waitKey(25)
                if key == ord('q') or key == 27:  # 27 is ESC
                    break
                elif key == ord('t'):
                    self.toggle_test_mode()

        finally:
            self.cleanup()

    def toggle_test_mode(self) -> None:
        """Toggle between test mode and full processing mode."""
        self.test_mode = not self.test_mode
        status = "enabled" if self.test_mode else "disabled"
        logging.info(f"Test mode {status}")
        print(f"Test mode {status}")  # Direct console output

    def cleanup(self) -> None:
        """Release resources explicitly."""
        if self.cap is not None:
            self.cap.release()
        cv2.destroyAllWindows()
        logging.info("Resources released")
        print("Resources released")  # Direct console output

    def save_screenshot(self, filename: Optional[str] = None) -> Optional[str]:
        """Save current frame as a screenshot."""
        if self.cap is None or not self.cap.isOpened():
            logging.error("Cannot take screenshot - no active video source")
            return None

        ret, frame = self.cap.read()
        if not ret:
            logging.error("Failed to capture frame for screenshot")
            return None

        try:
            # Process the frame to include annotations
            processed_frame = self.process_frame(frame)
            
            # Generate filename if not provided
            if filename is None:
                import datetime
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"screenshot_{timestamp}.jpg"
                
            # Ensure directory exists
            directory = os.path.dirname(filename)
            if directory and not os.path.exists(directory):
                os.makedirs(directory)
                
            # Save the image
            cv2.imwrite(filename, processed_frame)
            logging.info(f"Screenshot saved to {filename}")
            print(f"Screenshot saved to {filename}")  # Direct console output
            return filename
        except Exception as e:
            logging.error(f"Error saving screenshot: {e}")
            return None

# --- Main entry point ---
def main():
    """Main entry point for the application."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Child Safety Monitoring System')
    parser.add_argument('--config', default='config.yaml', help='Path to configuration file')
    parser.add_argument('--source', default='0', help='Video source (camera index or file path)')
    parser.add_argument('--test', action='store_true', help='Start in test mode')
    args = parser.parse_args()
    
    # Print startup information
    print(f"Child Safety Monitoring System")
    print(f"Configuration: {args.config}")
    print(f"Video source: {args.source}")
    print(f"Test mode: {'Enabled' if args.test else 'Disabled'}")
    
    # Initialize the safety monitor
    monitor = SafetyMonitor(config_path=args.config)
    
    monitor.test_mode = args.test  # Changed from forcing to True

    # Set test mode if requested
    if args.test:
        monitor.test_mode = True
    
    # Attempt to start the capture
    if monitor.start_capture(args.source):
        try:
            # Display keyboard controls
            print("\nKeyboard Controls:")
            print("  'q' - Quit the application")
            print("  't' - Toggle test mode")
            print("  's' - Save screenshot")
            print("\nStarting monitoring...")
            
            # Start monitoring
            monitor.run()
        except Exception as e:
            logging.error(f"Error during monitoring: {e}")
            print(f"Error during monitoring: {e}")
        finally:
            monitor.cleanup()
    else:
        print("Failed to start video capture. Exiting.")
        logging.error("Failed to start video capture. Exiting.")

if __name__ == "__main__":
    main()