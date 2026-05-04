# pipeline/cnn_inference.py
import numpy as np
import os

WEIGHTS_PATH = os.path.join(os.path.dirname(__file__), '..', 'models', 'combined_weights.weights.h5')
EMOTION_LABELS = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]


class CNNInference:
    def __init__(self):
        import tensorflow as tf
        from tensorflow import keras
        from tensorflow.keras import layers

        print("⏳ Building CNN model...")
        inputs = keras.Input(shape=(48, 48, 1))
        x = layers.Conv2D(32,(3,3),activation='relu',padding='same')(inputs)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D(2,2)(x)
        x = layers.Dropout(0.25)(x)
        x = layers.Conv2D(64,(3,3),activation='relu',padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D(2,2)(x)
        x = layers.Dropout(0.25)(x)
        x = layers.Conv2D(128,(3,3),activation='relu',padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D(2,2)(x)
        x = layers.Dropout(0.25)(x)
        x = layers.Conv2D(256,(3,3),activation='relu',padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dropout(0.50)(x)
        x = layers.Dense(256,activation='relu')(x)
        x = layers.Dropout(0.30)(x)
        outputs = layers.Dense(7,activation='softmax')(x)
        self.model = keras.Model(inputs, outputs)
        self.model.load_weights(WEIGHTS_PATH)

    def predict(self, face_crop_48x48):
        x = face_crop_48x48.reshape(1, 48, 48, 1)
        probabilities = self.model.predict(x, verbose=0)[0].copy()

        # Boost anxiety-relevant classes
        probabilities[2] *= 2.0  # Fear
        probabilities[1] *= 2.0  # Disgust
        probabilities[0] *= 1.5  # Angry

        # Renormalize to sum to 1
        probabilities = probabilities / probabilities.sum()

        result = {label: float(probabilities[i])
                for i, label in enumerate(EMOTION_LABELS)}

        result["anxiety_prob"] = float(
            0.40 * probabilities[2] +
            0.25 * probabilities[1] +
            0.20 * probabilities[0] +
            0.15 * probabilities[4]
        )
        return result


if __name__ == "__main__":
    import cv2, sys
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from pipeline.face_detector import FaceDetector

    detector = FaceDetector()
    cnn = CNNInference()
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("[ERROR] Cannot open webcam")
        sys.exit(1)

    print("📷 CNN test — press Q to quit")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        landmarks = detector.get_landmarks(frame)
        if landmarks is not None:
            face_crop = detector.get_face_crop(frame, landmarks)
            if face_crop is not None:
                result = cnn.predict(face_crop)
                top = max(EMOTION_LABELS, key=lambda e: result[e])
                cv2.putText(frame,
                    f"Emotion: {top} ({result[top]*100:.1f}%)",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(frame,
                    f"Anxiety: {result['anxiety_prob']*100:.1f}%",
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2)
                cv2.putText(frame,
                    f"Fear: {result['fear']*100:.1f}%  Disgust: {result['disgust']*100:.1f}%",
                    (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 200, 0), 2)
        else:
            cv2.putText(frame, "No face detected",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        cv2.imshow("CNN Inference Test", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    detector.close()
    print("✅ Test complete")