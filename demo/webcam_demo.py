# demo/webcam_demo.py
# Full real-time anxiety detection demo — webcam or video file

import cv2
import sys
import os
import argparse
import collections
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipeline.face_detector import FaceDetector
from pipeline.geometry      import compute_geometry
from pipeline.cnn_inference import CNNInference
from pipeline.score_fusion  import compute_score

EMOTION_LABELS = ["angry","disgust","fear","happy","neutral","sad","surprise"]


def run_demo(source=0):
    print("🚀 Initialising pipeline...")
    detector = FaceDetector()
    cnn      = CNNInference()

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open source: {source}")
        sys.exit(1)

    # Rolling average for smooth score
    score_history = collections.deque(maxlen=15)

    print("📷 Running — press Q to quit")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        landmarks = detector.get_landmarks(frame)

        if landmarks is not None:
            face_crop = detector.get_face_crop(frame, landmarks)
            geo       = compute_geometry(landmarks)

            if face_crop is not None:
                cnn_result = cnn.predict(face_crop)
                fusion     = compute_score(cnn_result, geo)

                score_history.append(fusion["anxiety_score"])
                smooth_score = round(float(np.mean(score_history)), 1)

                level = "LOW" if smooth_score < 33 else "MEDIUM" if smooth_score < 66 else "HIGH"
                color = (0,255,0) if smooth_score < 33 else (0,165,255) if smooth_score < 66 else (0,0,255)
                comp  = fusion["components"]

                # Score bar
                cv2.rectangle(frame, (10, 10), (310, 45), (40,40,40), -1)
                bar = int(smooth_score * 3)
                cv2.rectangle(frame, (10, 10), (10 + bar, 45), color, -1)

                # Main score
                cv2.putText(frame,
                    f"ANXIETY: {smooth_score:.1f}/100  [{level}]",
                    (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2)

                # Emotion
                top_emotion = max(EMOTION_LABELS, key=lambda e: cnn_result[e])
                cv2.putText(frame,
                    f"Emotion : {top_emotion} ({cnn_result[top_emotion]*100:.1f}%)",
                    (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2)

                # Components
                cv2.putText(frame,
                    f"CNN:{comp['cnn_anxiety']:.0f}% EAR:{comp['ear_norm']:.0f}% BCR:{comp['bcr_norm']:.0f}% LCR:{comp['lcr_norm']:.0f}%",
                    (10, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200,200,200), 1)

                # Geometric raw values
                cv2.putText(frame,
                    f"EAR:{comp['ear_raw']:.3f}  BCR:{comp['bcr_raw']:.3f}  LCR:{comp['lcr_raw']:.3f}",
                    (10, 155), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200,200,200), 1)

        else:
            cv2.putText(frame, "No face detected",
                        (10, 40), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (0,0,255), 2)

        cv2.imshow("Real-Time Anxiety Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    detector.close()
    print("✅ Demo complete")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--file', type=str, default=None,
                        help='Path to video file (omit for webcam)')
    args = parser.parse_args()
    source = args.file if args.file else 0
    run_demo(source)