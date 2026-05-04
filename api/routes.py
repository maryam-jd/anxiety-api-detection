from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Header# api/routes.py

from fastapi import APIRouter, Depends, HTTPException, UploadFile, File
from pydantic import BaseModel
from typing import Optional
import numpy as np
import cv2
import base64
import time
import api.main as main_module

router = APIRouter()

# ── Request / Response Models ─────────────────────────────

class StreamRequest(BaseModel):
    frame_b64: str

class AnalysisResponse(BaseModel):
    anxiety_score     : float
    level             : str
    emotion           : str
    emotion_confidence: float
    components        : dict
    heatmap_b64       : Optional[str] = None
    processing_time_ms: float
    timestamp         : str

# ── Helper — decode base64 image ──────────────────────────

def decode_image(data: str):
    img_bytes = base64.b64decode(data)
    arr = np.frombuffer(img_bytes, np.uint8)
    frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if frame is None:
        raise HTTPException(status_code=400, detail="Invalid image data")
    return frame

def encode_image(frame) -> str:
    _, buf = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return base64.b64encode(buf).decode('utf-8')

def run_pipeline(frame):
    from pipeline.geometry import compute_geometry
    from pipeline.score_fusion import compute_score

    EMOTION_LABELS = ["angry","disgust","fear","happy","neutral","sad","surprise"]

    landmarks = main_module.detector.get_landmarks(frame)
    if landmarks is None:
        raise HTTPException(status_code=422, detail="No face detected in image")

    face_crop  = main_module.detector.get_face_crop(frame, landmarks)
    geo        = compute_geometry(landmarks)
    cnn_result = main_module.cnn.predict(face_crop)
    fusion     = compute_score(cnn_result, geo)

    top_emotion = max(EMOTION_LABELS, key=lambda e: cnn_result[e])
    top_conf    = cnn_result[top_emotion]

    return fusion, cnn_result, top_emotion, top_conf, landmarks

# ── Endpoints ─────────────────────────────────────────────

@router.get("/health", tags=["System"])
def health_check():
    uptime = round(time.time() - main_module.start_time, 1) if main_module.start_time else 0
    return {
        "status"  : "ok",
        "version" : "1.0.0",
        "uptime_s": uptime
    }

# ── Auth Endpoints ────────────────────────────────────────

class RegisterRequest(BaseModel):
    email   : str
    password: str

class LoginRequest(BaseModel):
    email   : str
    password: str

@router.post("/auth/register", tags=["Auth"], status_code=201)
def register(request: RegisterRequest):
    from db.database import create_db, create_user, get_user_by_email
    create_db()
    if get_user_by_email(request.email):
        raise HTTPException(status_code=409, detail="Email already registered")
    user = create_user(request.email, request.password)
    return {"user_id": user.id, "message": "Registered successfully"}

@router.post("/auth/login", tags=["Auth"])
def login(request: LoginRequest):
    from db.database import verify_password
    user = verify_password(request.email, request.password)
    if not user:
        raise HTTPException(status_code=401, detail="Invalid credentials")
    return {
        "api_key"    : user.api_key,
        "email"      : user.email,
        "message"    : "Login successful"
    }


@router.post("/analyse/stream", tags=["Analysis"])
def analyse_stream(request: StreamRequest):
    """Analyse a single base64-encoded frame — for mobile real-time use."""
    t0    = time.time()
    frame = decode_image(request.frame_b64)

    fusion, cnn_result, top_emotion, top_conf, landmarks = run_pipeline(frame)

    # Draw heatmap overlay
    comp = fusion["components"]
    color = fusion["color"]
    cv2.putText(frame, f"Anxiety: {fusion['anxiety_score']}/100",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    return {
        "anxiety_score"      : fusion["anxiety_score"],
        "level"              : fusion["level"],
        "emotion"            : top_emotion,
        "emotion_confidence" : round(top_conf * 100, 1),
        "components"         : comp,
        "heatmap_b64"        : encode_image(frame),
        "processing_time_ms" : round((time.time() - t0) * 1000, 1),
        "timestamp"          : time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    }

@router.post("/analyse/frame", tags=["Analysis"])
async def analyse_frame(file: UploadFile = File(...)):
    """Analyse an uploaded image file."""
    t0      = time.time()
    content = await file.read()
    arr     = np.frombuffer(content, np.uint8)
    frame   = cv2.imdecode(arr, cv2.IMREAD_COLOR)

    if frame is None:
        raise HTTPException(status_code=400, detail="Invalid image file")

    fusion, cnn_result, top_emotion, top_conf, landmarks = run_pipeline(frame)
    comp = fusion["components"]

    return {
        "anxiety_score"      : fusion["anxiety_score"],
        "level"              : fusion["level"],
        "emotion"            : top_emotion,
        "emotion_confidence" : round(top_conf * 100, 1),
        "components"         : comp,
        "heatmap_b64"        : encode_image(frame),
        "processing_time_ms" : round((time.time() - t0) * 1000, 1),
        "timestamp"          : time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    }

@router.post("/analyse/video", tags=["Analysis"])
async def analyse_video(file: UploadFile = File(...)):
    """Analyse an uploaded video file — returns per-frame scores and session summary."""
    import tempfile, os
    from pipeline.geometry import compute_geometry
    from pipeline.score_fusion import compute_score

    EMOTION_LABELS = ["angry","disgust","fear","happy","neutral","sad","surprise"]

    t0 = time.time()

    # Save uploaded video to temp file
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    cap = cv2.VideoCapture(tmp_path)
    if not cap.isOpened():
        raise HTTPException(status_code=400, detail="Invalid video file")

    frame_scores   = []
    processed      = 0
    total          = 0
    high_anxiety_frames = 0
    fps = cap.get(cv2.CAP_PROP_FPS) or 25

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        total += 1

        # Process every 3rd frame for speed
        if total % 3 != 0:
            continue

        landmarks = main_module.detector.get_landmarks(frame)
        if landmarks is None:
            continue

        face_crop = main_module.detector.get_face_crop(frame, landmarks)
        if face_crop is None:
            continue

        geo        = compute_geometry(landmarks)
        cnn_result = main_module.cnn.predict(face_crop)
        fusion     = compute_score(cnn_result, geo)

        frame_scores.append(fusion["anxiety_score"])
        processed += 1
        if fusion["anxiety_score"] > 66:
            high_anxiety_frames += 1

    cap.release()
    os.unlink(tmp_path)

    if not frame_scores:
        raise HTTPException(status_code=422, detail="No faces detected in video")

    high_anxiety_seconds = round(high_anxiety_frames * 3 / fps, 1)

    return {
        "total_frames"     : total,
        "processed_frames" : processed,
        "session_summary"  : {
            "mean_score"            : round(float(np.mean(frame_scores)), 1),
            "max_score"             : round(float(np.max(frame_scores)), 1),
            "min_score"             : round(float(np.min(frame_scores)), 1),
            "high_anxiety_seconds"  : high_anxiety_seconds
        },
        "frame_scores"         : [round(s, 1) for s in frame_scores],
        "processing_time_ms"   : round((time.time() - t0) * 1000, 1),
        "timestamp"            : time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    }



# ── Subscription Endpoints ────────────────────────────────

class SubscriptionRequest(BaseModel):
    plan       : str  # "monthly" or "annual"
    card_token : str  # Stripe token

@router.get("/subscription/status", tags=["Subscription"])
def subscription_status(x_api_key: str = Header(...)):
    from db.database import get_user_by_api_key
    user = get_user_by_api_key(x_api_key)
    if not user:
        raise HTTPException(status_code=401, detail="Invalid API key")
    return {
        "active"    : user["subscription_active"],
        "expires_at": str(user["subscription_expires_at"]),
        "plan"      : user["plan"]
    }

@router.post("/subscription/start", tags=["Subscription"])
def subscription_start(request: SubscriptionRequest, x_api_key: str = Header(...)):
    from db.database import get_user_by_api_key, engine
    from sqlmodel import Session, select
    from db.database import User
    from datetime import datetime, timedelta

    user_data = get_user_by_api_key(x_api_key)
    if not user_data:
        raise HTTPException(status_code=401, detail="Invalid API key")

    # For now simulate payment success (Stripe integration in Phase 5)
    with Session(engine) as session:
        user = session.exec(select(User).where(User.api_key == x_api_key)).first()
        user.subscription_active     = True
        user.subscription_expires_at = datetime.utcnow() + timedelta(days=30)
        user.plan                    = request.plan
        session.add(user)
        session.commit()

    return {
        "message"          : "Subscription activated",
        "plan"             : request.plan,
        "expires_at"       : str(datetime.utcnow() + timedelta(days=30))
    }

@router.post("/subscription/cancel", tags=["Subscription"])
def subscription_cancel(x_api_key: str = Header(...)):
    from db.database import get_user_by_api_key, engine
    from sqlmodel import Session, select
    from db.database import User

    user_data = get_user_by_api_key(x_api_key)
    if not user_data:
        raise HTTPException(status_code=401, detail="Invalid API key")

    with Session(engine) as session:
        user = session.exec(select(User).where(User.api_key == x_api_key)).first()
        user.subscription_active = False
        session.add(user)
        session.commit()

    return {"message": "Subscription cancelled"}