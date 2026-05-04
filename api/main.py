# api/main.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import time
import os

# ── Pipeline imports ──────────────────────────────────────
from pipeline.face_detector import FaceDetector
from pipeline.geometry      import compute_geometry
from pipeline.cnn_inference import CNNInference
from pipeline.score_fusion  import compute_score

# ── Global pipeline objects (loaded once at startup) ──────
detector = None
cnn      = None
start_time = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global detector, cnn, start_time
    print("🚀 Loading pipeline...")

    # Download face landmarker model if not present
    model_path = os.path.join("models", "face_landmarker.task")
    if not os.path.exists(model_path):
        import urllib.request
        os.makedirs("models", exist_ok=True)
        print("Downloading face landmarker...")
        urllib.request.urlretrieve(
            "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task",
            model_path
        )
        print("✅ Downloaded")

    detector   = FaceDetector()
    cnn        = CNNInference()
    start_time = time.time()

    from db.database import create_db
    create_db()

    print("✅ Pipeline ready")
    print("✅ Database ready")
    yield
    print("🛑 Shutting down")
    if detector:
        detector.close()
    # Download CNN weights if not present
    if not os.path.exists(weights_path):
        print("Downloading CNN weights...")
        import gdown
        gdown.download(
            "https://drive.google.com/file/d/1bTGcFcV5MQBBlSXYWJ2u6Ues_IU_VbmQ/view?usp=drive_link",
            weights_path, quiet=False
        )
        print("✅ CNN weights downloaded")
    weights_path = os.path.join("models", "combined_weights.weights.h5")
    
# ── App ───────────────────────────────────────────────────
app = FastAPI(
    title       = "Anxiety Detection API",
    description = "Real-Time Facial Tension & Anxiety Detection — DIP Project",
    version     = "1.0.0",
    lifespan    = lifespan
)

# ── CORS (allow mobile apps to call this API) ─────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins     = ["*"],
    allow_credentials = True,
    allow_methods     = ["*"],
    allow_headers     = ["*"],
)

# ── Include routes ────────────────────────────────────────
from api.routes import router
app.include_router(router)