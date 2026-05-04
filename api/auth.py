# api/auth.py
from fastapi import Header, HTTPException
from db.database import get_user_by_api_key
import datetime

async def verify_api_key(x_api_key: str = Header(...)):
    user = get_user_by_api_key(x_api_key)
    if not user:
        raise HTTPException(status_code=401, detail="Invalid API key")
    if not user.get("subscription_active"):
        raise HTTPException(status_code=402, detail="Subscription required")
    expires = user.get("subscription_expires_at")
    if expires and datetime.datetime.utcnow() > expires:
        raise HTTPException(status_code=402, detail="Subscription expired")
    return user