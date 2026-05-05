# data/preprocess.py
# Phase 1 — Dataset Preprocessing Script
# Reads FER2013 from image folders (train/test) and CK+ from CK+48/ck folders

import os
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.utils import to_categorical

# ── Paths ──────────────────────────────────────────────────────────────────
RAW_DIR       = os.path.join(os.path.dirname(__file__), "raw")
PROCESSED_DIR = os.path.join(os.path.dirname(__file__), "processed")
os.makedirs(PROCESSED_DIR, exist_ok=True)

# ── Emotion labels ─────────────────────────────────────────────────────────
EMOTION_LABELS   = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]
EMOTION_TO_INDEX = {name: i for i, name in enumerate(EMOTION_LABELS)}

# ══════════════════════════════════════════════════════════════════════════
# HELPER — load all images from a folder tree like train/angry/*.jpg
# ══════════════════════════════════════════════════════════════════════════

def load_image_folder(root_dir):
    images, labels = [], []
    for emotion_name in EMOTION_LABELS:
        folder = os.path.join(root_dir, emotion_name)
        if not os.path.exists(folder):
            print(f"  [WARN] Folder not found, skipping: {folder}")
            continue
        files = [f for f in os.listdir(folder)
                 if f.lower().endswith((".jpg", ".jpeg", ".png"))]
        print(f"    [{EMOTION_TO_INDEX[emotion_name]}] {emotion_name:10s}: {len(files)} images")
        for fname in files:
            img_path = os.path.join(folder, fname)
            try:
                img = Image.open(img_path).convert("L").resize((48, 48))
                images.append(np.array(img, dtype=np.float32) / 255.0)
                labels.append(EMOTION_TO_INDEX[emotion_name])
            except Exception as e:
                print(f"  [SKIP] {fname}: {e}")
    return np.array(images), np.array(labels)


# ══════════════════════════════════════════════════════════════════════════
# PART 1 — FER2013
# ══════════════════════════════════════════════════════════════════════════

def process_fer2013():
    print("\n" + "="*55)
    print("  PART 1: Processing FER2013 Dataset")
    print("="*55)# data/preprocess.py
# Phase 1 — Dataset Preprocessing Script
# Reads FER2013 from image folders (train/test) and CK+ from CK+48/ck folders

import os
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.utils import to_categorical

# ── Paths ──────────────────────────────────────────────────────────────────
RAW_DIR       = os.path.join(os.path.dirname(__file__), "raw")
PROCESSED_DIR = os.path.join(os.path.dirname(__file__), "processed")
os.makedirs(PROCESSED_DIR, exist_ok=True)

# ── Emotion labels ─────────────────────────────────────────────────────────
EMOTION_LABELS   = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]
EMOTION_TO_INDEX = {name: i for i, name in enumerate(EMOTION_LABELS)}

# ══════════════════════════════════════════════════════════════════════════
# HELPER — load all images from a folder tree like train/angry/*.jpg
# ══════════════════════════════════════════════════════════════════════════

def load_image_folder(root_dir):
    images, labels = [], []
    for emotion_name in EMOTION_LABELS:
        folder = os.path.join(root_dir, emotion_name)
        if not os.path.exists(folder):
            print(f"  [WARN] Folder not found, skipping: {folder}")
            continue
        files = [f for f in os.listdir(folder)
                 if f.lower().endswith((".jpg", ".jpeg", ".png"))]
        print(f"    [{EMOTION_TO_INDEX[emotion_name]}] {emotion_name:10s}: {len(files)} images")
        for fname in files:
            img_path = os.path.join(folder, fname)
            try:
                img = Image.open(img_path).convert("L").resize((48, 48))
                images.append(np.array(img, dtype=np.float32) / 255.0)
                labels.append(EMOTION_TO_INDEX[emotion_name])
            except Exception as e:
                print(f"  [SKIP] {fname}: {e}")
    return np.array(images), np.array(labels)


# ══════════════════════════════════════════════════════════════════════════
# PART 1 — FER2013
# ══════════════════════════════════════════════════════════════════════════

def process_fer2013():
    print("\n" + "="*55)
    print("  PART 1: Processing FER2013 Dataset")
    print("="*55)

    train_dir = os.path.join(RAW_DIR, "train")
    test_dir  = os.path.join(RAW_DIR, "test")

    if not os.path.exists(train_dir):
        print(f"[ERROR] train/ folder not found at: {train_dir}")
        return

    print("[1/5] Loading TRAIN images...")
    X_all, y_all = load_image_folder(train_dir)
    print(f"      Total loaded: {len(X_all)} images")

    print("[2/5] Loading TEST images...")
    X_test_raw, y_test_raw = load_image_folder(test_dir)
    print(f"      Total loaded: {len(X_test_raw)} images")

    print("[3/5] Reshaping for CNN input (adding channel dimension)...")
    X_all       = X_all.reshape(-1, 48, 48, 1)
    X_test_raw  = X_test_raw.reshape(-1, 48, 48, 1)

    print("[4/5] One-hot encoding labels...")
    y_all      = to_categorical(y_all,      num_classes=7)
    y_test_raw = to_categorical(y_test_raw, num_classes=7)

    print("[5/5] Splitting train → train / val (90/10)...")
    X_train, X_val, y_train, y_val = train_test_split(
        X_all, y_all,
        test_size=0.10,
        random_state=42,
        stratify=np.argmax(y_all, axis=1)
    )
    X_test, y_test = X_test_raw, y_test_raw
    print(f"      Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")

    print("\n  Saving arrays to data/processed/ ...")
    np.save(os.path.join(PROCESSED_DIR, "X_train.npy"), X_train)
    np.save(os.path.join(PROCESSED_DIR, "y_train.npy"), y_train)
    np.save(os.path.join(PROCESSED_DIR, "X_val.npy"),   X_val)
    np.save(os.path.join(PROCESSED_DIR, "y_val.npy"),   y_val)
    np.save(os.path.join(PROCESSED_DIR, "X_test.npy"),  X_test)
    np.save(os.path.join(PROCESSED_DIR, "y_test.npy"),  y_test)

    print("\n  Computing class weights (to handle imbalance)...")
    raw_labels = np.argmax(y_train, axis=1)
    class_weights = compute_class_weight(
        class_weight="balanced",
        classes=np.unique(raw_labels),
        y=raw_labels
    )
    np.save(os.path.join(PROCESSED_DIR, "class_weights.npy"), class_weights)

    print("\n  Class weights:")
    for i, name in enumerate(EMOTION_LABELS):
        print(f"    [{i}] {name:10s}: weight = {class_weights[i]:.3f}")

    print("\n  ✅ FER2013 preprocessing complete!")


# ══════════════════════════════════════════════════════════════════════════
# PART 2 — CK+ Dataset
# ══════════════════════════════════════════════════════════════════════════

CK_LABEL_MAP = {
    0: 0,  # Anger    → angry
    1: 1,  # Contempt → disgust
    2: 1,  # Disgust  → disgust
    3: 2,  # Fear     → fear
    4: 3,  # Happy    → happy
    5: 5,  # Sadness  → sad
    6: 6,  # Surprise → surprise
}

def process_ckplus():
    print("\n" + "="*55)
    print("  PART 2: Processing CK+ Dataset")
    print("="*55)

    ck_dir = os.path.join(RAW_DIR, "ck", "CK+48")

    if not os.path.exists(ck_dir):
        print(f"  [SKIP] Folder not found: {ck_dir}")
        return

    # CK+ folder names → our emotion index
    CK_FOLDER_MAP = {
        "anger":    0,  # angry
        "contempt": 1,  # disgust (closest)
        "disgust":  1,  # disgust
        "fear":     2,  # fear
        "happy":    3,  # happy
        "sadness":  5,  # sad
        "surprise": 6,  # surprise
    }

    ck_images, ck_labels = [], []

    for folder_name, label_index in CK_FOLDER_MAP.items():
        folder_path = os.path.join(ck_dir, folder_name)
        if not os.path.exists(folder_path):
            continue
        files = [f for f in os.listdir(folder_path)
                 if f.lower().endswith((".jpg", ".jpeg", ".png"))]
        print(f"    {folder_name:10s}: {len(files)} images")
        for fname in files:
            try:
                img = Image.open(os.path.join(folder_path, fname)).convert("L").resize((48, 48))
                ck_images.append(np.array(img, dtype=np.float32) / 255.0)
                ck_labels.append(label_index)
            except Exception as e:
                print(f"  [SKIP] {fname}: {e}")

    if not ck_images:
        print("  [SKIP] No images loaded.")
        return

    ck_images = np.array(ck_images).reshape(-1, 48, 48, 1)
    ck_labels = np.array(ck_labels)

    np.save(os.path.join(PROCESSED_DIR, "ck_X.npy"), ck_images)
    np.save(os.path.join(PROCESSED_DIR, "ck_y.npy"), ck_labels)

    print(f"\n  ✅ Saved {len(ck_images)} CK+ images")
    print("  CK+ class distribution:")
    for i, name in enumerate(EMOTION_LABELS):
        count = np.sum(ck_labels == i)
        if count > 0:
            print(f"    [{i}] {name:10s}: {count} samples")
    print("\n  ✅ CK+ preprocessing complete!")


# ══════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("\n" + "█"*55)
    print("  ANXIETY DETECTION — DATA PREPROCESSING SCRIPT")
    print("█"*55)

    process_fer2013()
    process_ckplus()

    print("\n" + "="*55)
    print("  PREPROCESSING SUMMARY — Files Saved")
    print("="*55)
    for f in sorted(os.listdir(PROCESSED_DIR)):
        path = os.path.join(PROCESSED_DIR, f)
        size_mb = os.path.getsize(path) / (1024 * 1024)
        print(f"  ✅  {f:30s}  {size_mb:.1f} MB")

    print("\n  All done! Ready for Phase 2 — Colab Training.")
    print("="*55 + "\n")