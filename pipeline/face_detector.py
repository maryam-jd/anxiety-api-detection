# pipeline/face_detector.py
import cv2
import numpy as np
import os
import urllib.request

MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', 'models', 'face_landmarker.task')

def download_model():
    if not os.path.exists(MODEL_PATH):
        print("Downloading face landmarker model...")
        url = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
        urllib.request.urlretrieve(url, MODEL_PATH)
        print("✅ Model downloaded")

download_model()

from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
from mediapipe.tasks.python.vision import FaceLandmarker, FaceLandmarkerOptions, RunningMode


class FaceDetector:
    def __init__(self):
        options = FaceLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=MODEL_PATH),
            running_mode=RunningMode.IMAGE,
            num_faces=1,
            min_face_detection_confidence=0.5,
            min_face_presence_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.detector = FaceLandmarker.create_from_options(options)
        print("✅ FaceDetector initialised")

    def get_landmarks(self, frame_bgr):
        import mediapipe as mp
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        h, w = frame_bgr.shape[:2]
        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=frame_rgb
        )
        result = self.detector.detect(mp_image)
        if not result.face_landmarks:
            return None
        landmarks = result.face_landmarks[0]
        points = np.array(
            [[lm.x * w, lm.y * h, lm.z * w] for lm in landmarks],
            dtype=np.float32
        )
        return points

    def get_face_crop(self, frame_bgr, landmarks):
        h, w = frame_bgr.shape[:2]
        x_coords = landmarks[:, 0]
        y_coords = landmarks[:, 1]
        x_min = max(int(np.min(x_coords)) - 10, 0)
        x_max = min(int(np.max(x_coords)) + 10, w)
        y_min = max(int(np.min(y_coords)) - 10, 0)
        y_max = min(int(np.max(y_coords)) + 10, h)
        face_crop = frame_bgr[y_min:y_max, x_min:x_max]
        if face_crop.size == 0:
            return None
        face_grey = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
        face_resized = cv2.resize(face_grey, (48, 48))
        face_norm = face_resized.astype(np.float32) / 255.0
        return face_norm.reshape(48, 48, 1)

    def close(self):
        self.detector.close()


if __name__ == "__main__":
    import sys
    detector = FaceDetector()
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("[ERROR] Could not open webcam")
        sys.exit(1)

    print("📷 Webcam open — press Q to quit")

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        landmarks = detector.get_landmarks(frame)
        if landmarks is not None:
            for pt in landmarks:
                cv2.circle(frame, (int(pt[0]), int(pt[1])), 1, (0, 255, 0), -1)
            cv2.putText(frame, f"✅ {len(landmarks)} landmarks",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        else:
            cv2.putText(frame, "No face detected",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.imshow("Face Detector Test", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    detector.close()
    print("✅ Test complete")