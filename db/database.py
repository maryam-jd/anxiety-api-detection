# db/database.py
from sqlmodel import SQLModel, Field, Session, create_engine, select
from typing   import Optional
from datetime import datetime
import uuid, secrets, os, hashlib

DATABASE_URL = "sqlite:///./db/anxiety_api.db"
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})


class User(SQLModel, table=True):
    id                     : str      = Field(default_factory=lambda: str(uuid.uuid4()), primary_key=True)
    email                  : str      = Field(unique=True, index=True)
    password_hash          : str
    api_key                : str      = Field(unique=True, index=True)
    subscription_active    : bool     = Field(default=False)
    subscription_expires_at: Optional[datetime] = None
    stripe_customer_id     : Optional[str] = None
    stripe_subscription_id : Optional[str] = None
    plan                   : Optional[str] = None
    created_at             : datetime = Field(default_factory=datetime.utcnow)
    total_api_calls        : int      = Field(default=0)


# ── Password Hashing ──────────────────────────────────────
def hash_password(password: str) -> str:
    return hashlib.sha256(password.encode()).hexdigest()

def verify_password_hash(password: str, hashed: str) -> bool:
    return hashlib.sha256(password.encode()).hexdigest() == hashed


# ── DB Init ───────────────────────────────────────────────
def create_db():
    os.makedirs("db", exist_ok=True)
    SQLModel.metadata.create_all(engine)


# ── CRUD Operations ───────────────────────────────────────
def create_user(email: str, password: str):
    with Session(engine) as session:
        user = User(
            email         = email,
            password_hash = hash_password(password),
            api_key       = secrets.token_hex(32)
        )
        session.add(user)
        session.commit()
        session.refresh(user)
        return user

def verify_password(email: str, password: str):
    with Session(engine) as session:
        user = session.exec(
            select(User).where(User.email == email)
        ).first()
        if not user:
            return None
        if not verify_password_hash(password, user.password_hash):
            return None
        return user

def get_user_by_email(email: str):
    with Session(engine) as session:
        return session.exec(
            select(User).where(User.email == email)
        ).first()

def get_user_by_api_key(api_key: str):
    with Session(engine) as session:
        user = session.exec(
            select(User).where(User.api_key == api_key)
        ).first()
        if not user:
            return None
        return {
            "id"                     : user.id,
            "email"                  : user.email,
            "api_key"                : user.api_key,
            "subscription_active"    : user.subscription_active,
            "subscription_expires_at": user.subscription_expires_at,
            "plan"                   : user.plan
        }