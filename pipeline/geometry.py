# pipeline/geometry.py
# Computes EAR, BCR, LCR — the three geometric anxiety indicators
# These are classical Digital Image Processing feature extraction techniques

import numpy as np


# ══════════════════════════════════════════════════════════════════════════
# LANDMARK INDICES (MediaPipe 478-point face mesh)
# ══════════════════════════════════════════════════════════════════════════

# Eye landmarks (6 points per eye — used for EAR)
LEFT_EYE  = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]

# Brow landmarks (inner corners — used for BCR)
LEFT_INNER_BROW  = 107
RIGHT_INNER_BROW = 336

# Face width landmarks (cheek points — used for normalisation)
LEFT_CHEEK  = 234
RIGHT_CHEEK = 454

# Lip landmarks (used for LCR)
UPPER_LIP_CENTER = 13
LOWER_LIP_CENTER = 14
LEFT_MOUTH_CORNER  = 61
RIGHT_MOUTH_CORNER = 291


# ══════════════════════════════════════════════════════════════════════════
# HELPER
# ══════════════════════════════════════════════════════════════════════════

def _dist(p1, p2):
    """Euclidean distance between two 2D points."""
    return np.linalg.norm(p1[:2] - p2[:2])


# ══════════════════════════════════════════════════════════════════════════
# EAR — Eye Aspect Ratio
# ══════════════════════════════════════════════════════════════════════════

def compute_ear(landmarks):
    """
    Eye Aspect Ratio — measures how open the eyes are.
    Anxiety causes squinting → lower EAR.

    Formula: EAR = (|p2-p6| + |p3-p5|) / (2 × |p1-p4|)
    Points:  p1=corner, p2-p3=upper lid, p4=other corner, p5-p6=lower lid

    Typical values:
        Relaxed : ~0.30
        Squinting: < 0.20
    """
    def eye_ear(indices):
        p = landmarks[indices]
        vertical_1 = _dist(p[1], p[5])
        vertical_2 = _dist(p[2], p[4])
        horizontal = _dist(p[0], p[3])
        if horizontal < 1e-6:
            return 0.0
        return (vertical_1 + vertical_2) / (2.0 * horizontal)

    left_ear  = eye_ear(LEFT_EYE)
    right_ear = eye_ear(RIGHT_EYE)
    return float((left_ear + right_ear) / 2.0)


# ══════════════════════════════════════════════════════════════════════════
# BCR — Brow Compression Ratio
# ══════════════════════════════════════════════════════════════════════════

def compute_bcr(landmarks):
    """
    Brow Compression Ratio — measures how furrowed the brows are.
    Anxiety causes brows to pull inward → lower BCR.

    Formula: BCR = distance(left_inner_brow, right_inner_brow) / face_width

    Typical values:
        Relaxed  : ~0.35
        Furrowed : < 0.25
    """
    left_brow  = landmarks[LEFT_INNER_BROW]
    right_brow = landmarks[RIGHT_INNER_BROW]
    left_cheek  = landmarks[LEFT_CHEEK]
    right_cheek = landmarks[RIGHT_CHEEK]

    brow_distance = _dist(left_brow, right_brow)
    face_width    = _dist(left_cheek, right_cheek)

    if face_width < 1e-6:
        return 0.0
    return float(brow_distance / face_width)


# ══════════════════════════════════════════════════════════════════════════
# LCR — Lip Compression Ratio
# ══════════════════════════════════════════════════════════════════════════

def compute_lcr(landmarks):
    """
    Lip Compression Ratio — measures how compressed the lips are.
    Anxiety causes lip pressing and jaw tightening → lower LCR.

    Formula: LCR = lip_height / mouth_width

    Typical values:
        Relaxed    : ~0.15
        Compressed : < 0.08
    """
    upper_lip = landmarks[UPPER_LIP_CENTER]
    lower_lip = landmarks[LOWER_LIP_CENTER]
    left_corner  = landmarks[LEFT_MOUTH_CORNER]
    right_corner = landmarks[RIGHT_MOUTH_CORNER]

    lip_height   = _dist(upper_lip, lower_lip)
    mouth_width  = _dist(left_corner, right_corner)

    if mouth_width < 1e-6:
        return 0.0
    return float(lip_height / mouth_width)


# ══════════════════════════════════════════════════════════════════════════
# MAIN — compute all three at once
# ══════════════════════════════════════════════════════════════════════════

def compute_geometry(landmarks):
    """
    Compute all three geometric ratios in one call.

    Input  : landmarks — numpy array (478, 3)
    Output : dict with keys ear, bcr, lcr
    """
    return {
        "ear": compute_ear(landmarks),
        "bcr": compute_bcr(landmarks),
        "lcr": compute_lcr(landmarks)
    }


# ══════════════════════════════════════════════════════════════════════════
# QUICK TEST
# ══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import cv2
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from pipeline.face_detector import FaceDetector

    detector = FaceDetector()
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("[ERROR] Cannot open webcam")
        sys.exit(1)

    print("📷 Geometry test — press Q to quit")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        landmarks = detector.get_landmarks(frame)

        if landmarks is not None:
            geo = compute_geometry(landmarks)

            # Display values on screen
            cv2.putText(frame, f"EAR: {geo['ear']:.3f}",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, f"BCR: {geo['bcr']:.3f}",
                        (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, f"LCR: {geo['lcr']:.3f}",
                        (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            # Colour indicator
            ear = geo['ear']
            color = (0, 255, 0) if ear > 0.25 else (0, 165, 255) if ear > 0.20 else (0, 0, 255)
            cv2.putText(frame, "RELAXED" if ear > 0.25 else "TENSE",
                        (10, 130), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 3)
        else:
            cv2.putText(frame, "No face detected",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        cv2.imshow("Geometry Test — EAR / BCR / LCR", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    detector.close()
    print("✅ Geometry test complete")