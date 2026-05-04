# pipeline/score_fusion.py
# Combines CNN probabilities + geometric ratios → final anxiety score 0-100

import numpy as np


def compute_score(cnn_result, geometry_result):
    """
    Inputs:
        cnn_result      — dict from CNNInference.predict()
        geometry_result — dict from compute_geometry() with ear, bcr, lcr

    Output:
        dict with:
            anxiety_score  — float 0.0 to 100.0
            level          — "LOW" / "MEDIUM" / "HIGH"
            components     — breakdown of each signal
    """

    # ── CNN Component ─────────────────────────────────────
    cnn_anxiety = float(
        0.40 * cnn_result.get('fear',    0) +
        0.25 * cnn_result.get('disgust', 0) +
        0.20 * cnn_result.get('angry',   0) +
        0.15 * cnn_result.get('sad',     0)
    )

    # ── Geometric Components (normalise to 0-1 anxiety scale) ──
    ear = geometry_result.get('ear', 0.30)
    bcr = geometry_result.get('bcr', 0.35)
    lcr = geometry_result.get('lcr', 0.15)

    # Lower value = more tension = higher anxiety
    ear_norm = float(np.clip(1.0 - (ear / 0.30), 0.0, 1.0))
    bcr_norm = float(np.clip(1.0 - (bcr / 0.40), 0.0, 1.0))
    lcr_norm = float(np.clip(1.0 - (lcr / 0.15), 0.0, 1.0))

    # ── Weighted Fusion ───────────────────────────────────
    raw_score = (
        0.50 * cnn_anxiety +
        0.20 * ear_norm    +
        0.15 * bcr_norm    +
        0.15 * lcr_norm
    )

    # Scale to 0-100
    final_score = round(float(np.clip(raw_score * 100, 0.0, 100.0)), 1)

    # ── Level ─────────────────────────────────────────────
    if final_score < 33:
        level = "LOW"
        color = (0, 255, 0)      # Green
    elif final_score < 66:
        level = "MEDIUM"
        color = (0, 165, 255)    # Orange
    else:
        level = "HIGH"
        color = (0, 0, 255)      # Red

    return {
        "anxiety_score" : final_score,
        "level"         : level,
        "color"         : color,
        "components"    : {
            "cnn_anxiety" : round(cnn_anxiety * 100, 1),
            "ear_norm"    : round(ear_norm * 100, 1),
            "bcr_norm"    : round(bcr_norm * 100, 1),
            "lcr_norm"    : round(lcr_norm * 100, 1),
            "ear_raw"     : round(ear, 3),
            "bcr_raw"     : round(bcr, 3),
            "lcr_raw"     : round(lcr, 3),
        }
    }


# ── Quick test ────────────────────────────────────────────
if __name__ == "__main__":
    import cv2, sys, os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from pipeline.face_detector import FaceDetector
    from pipeline.geometry      import compute_geometry
    from pipeline.cnn_inference import CNNInference

    detector = FaceDetector()
    cnn      = CNNInference()
    cap      = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("[ERROR] Cannot open webcam")
        sys.exit(1)

    print("📷 Full pipeline test — press Q to quit")

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

                score = fusion["anxiety_score"]
                level = fusion["level"]
                color = fusion["color"]
                comp  = fusion["components"]

                # ── Draw overlay ──────────────────────────
                # Score bar background
                cv2.rectangle(frame, (10, 10), (310, 50), (50, 50, 50), -1)
                bar_width = int(score * 3)
                cv2.rectangle(frame, (10, 10), (10 + bar_width, 50), color, -1)

                # Text overlays
                cv2.putText(frame,
                    f"ANXIETY: {score:.1f}/100  [{level}]",
                    (10, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

                cv2.putText(frame,
                    f"CNN: {comp['cnn_anxiety']:.1f}%  EAR: {comp['ear_norm']:.1f}%",
                    (10, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)

                cv2.putText(frame,
                    f"BCR: {comp['bcr_norm']:.1f}%  LCR: {comp['lcr_norm']:.1f}%",
                    (10, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)

                cv2.putText(frame,
                    f"Emotion: {max(['angry','disgust','fear','happy','neutral','sad','surprise'], key=lambda e: cnn_result[e])}",
                    (10, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        else:
            cv2.putText(frame, "No face detected",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (0, 0, 255), 2)

        cv2.imshow("Anxiety Detection — Full Pipeline", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    detector.close()
    print("✅ Full pipeline test complete")