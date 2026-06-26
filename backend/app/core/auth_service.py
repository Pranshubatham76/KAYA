"""
SentinelSite — Auth Service
JWT (RS256 → HS256 for simplicity here), bcrypt passwords, RBAC.
Roles: worker | supervisor | admin | system
Workers identified by SHA-256 worker_id hash (anonymizable).
"""
from __future__ import annotations

import hashlib
import logging
from datetime import datetime, timedelta
from uuid import uuid4

from jose import JWTError, jwt
from passlib.context import CryptContext
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.db.models import Site, User, UserRole

log = logging.getLogger(__name__)

_pwd_ctx = CryptContext(schemes=["bcrypt"], deprecated="auto")

# ── Token helpers ─────────────────────────────────────────────────────────────

def _hash_password(plain: str) -> str:
    return _pwd_ctx.hash(plain)

def _verify_password(plain: str, hashed: str) -> bool:
    return _pwd_ctx.verify(plain, hashed)

def _worker_id_hash(raw_worker_id: str) -> str:
    """SHA-256 of raw worker ID — stored in DB, sent by device."""
    return hashlib.sha256(raw_worker_id.encode()).hexdigest()

def _create_token(payload: dict, expires_delta: timedelta | None = None) -> str:
    data = payload.copy()
    expire = datetime.utcnow() + (expires_delta or timedelta(hours=settings.JWT_EXPIRY_HOURS))
    data["exp"] = expire
    data["iat"] = datetime.utcnow()
    return jwt.encode(data, settings.SECRET_KEY, algorithm=settings.JWT_ALGORITHM)

def decode_token(token: str) -> dict:
    """Raises JWTError on invalid/expired token."""
    return jwt.decode(token, settings.SECRET_KEY, algorithms=[settings.JWT_ALGORITHM])


# ── Service ───────────────────────────────────────────────────────────────────

class AuthService:

    # ── Registration ──────────────────────────────────────────────────────────

    async def register_supervisor(
        self,
        db: AsyncSession,
        site_id: str,
        email: str,
        password: str,
    ) -> User:
        """Register a supervisor/admin account with email+password."""
        existing = await db.execute(select(User).where(User.email == email))
        if existing.scalar_one_or_none():
            raise ValueError(f"Email already registered: {email}")

        user = User(
            id=str(uuid4()),
            site_id=site_id,
            email=email,
            hashed_password=_hash_password(password),
            role=UserRole.SUPERVISOR,
            is_active=True,
        )
        db.add(user)
        await db.commit()
        await db.refresh(user)
        log.info(f"Supervisor registered: {email} @ site={site_id}")
        return user

    async def register_worker(
        self,
        db: AsyncSession,
        site_id: str,
        raw_worker_id: str,
        device_id: str,
    ) -> tuple[User, str]:
        """
        Register/upsert a worker by raw worker ID (badge number etc.).
        Stores SHA-256 hash only — raw ID never persisted.
        Returns (user, access_token).
        """
        hashed_id = _worker_id_hash(raw_worker_id)

        result = await db.execute(
            select(User).where(
                User.worker_id == hashed_id,
                User.site_id == site_id,
            )
        )
        user = result.scalar_one_or_none()

        if not user:
            user = User(
                id=str(uuid4()),
                site_id=site_id,
                worker_id=hashed_id,
                device_id=device_id,
                role=UserRole.WORKER,
                is_active=True,
            )
            db.add(user)
            log.info(f"Worker registered: hash={hashed_id[:8]}... @ site={site_id}")
        else:
            # Update device pairing
            user.device_id = device_id

        await db.commit()
        await db.refresh(user)

        token = _create_token({
            "sub": str(user.id),
            "site_id": site_id,
            "role": user.role.value,
            "worker_id": hashed_id,
        })
        return user, token

    # ── Login ─────────────────────────────────────────────────────────────────

    async def login_supervisor(
        self,
        db: AsyncSession,
        email: str,
        password: str,
    ) -> tuple[User, str]:
        """Email + password login for supervisor/admin."""
        result = await db.execute(select(User).where(User.email == email))
        user = result.scalar_one_or_none()

        if not user or not user.hashed_password:
            raise ValueError("Invalid credentials")
        if not _verify_password(password, user.hashed_password):
            raise ValueError("Invalid credentials")
        if not user.is_active:
            raise ValueError("Account deactivated")

        token = _create_token({
            "sub": str(user.id),
            "site_id": str(user.site_id),
            "role": user.role.value,
            "email": user.email,
        })
        log.info(f"Login: {email} (role={user.role.value})")
        return user, token

    async def login_worker_device(
        self,
        db: AsyncSession,
        raw_worker_id: str,
        device_id: str,
        site_id: str,
    ) -> tuple[User, str]:
        """Device login — worker identified by raw ID + device pairing."""
        hashed_id = _worker_id_hash(raw_worker_id)
        result = await db.execute(
            select(User).where(
                User.worker_id == hashed_id,
                User.site_id == site_id,
            )
        )
        user = result.scalar_one_or_none()
        if not user:
            # Auto-register on first device login
            user, token = await self.register_worker(db, site_id, raw_worker_id, device_id)
            return user, token

        if not user.is_active:
            raise ValueError("Worker account deactivated")

        token = _create_token({
            "sub": str(user.id),
            "site_id": site_id,
            "role": user.role.value,
            "worker_id": hashed_id,
        })
        return user, token

    # ── Token verification ─────────────────────────────────────────────────────

    async def get_current_user(
        self,
        db: AsyncSession,
        token: str,
    ) -> User:
        """Decode JWT → fetch User. Raises on invalid token or inactive user."""
        try:
            payload = decode_token(token)
            user_id: str = payload.get("sub")
            if not user_id:
                raise ValueError("Token missing sub")
        except JWTError as e:
            raise ValueError(f"Invalid token: {e}")

        result = await db.execute(select(User).where(User.id == user_id))
        user = result.scalar_one_or_none()
        if not user or not user.is_active:
            raise ValueError("User not found or inactive")
        return user

    # ── RBAC helpers ──────────────────────────────────────────────────────────

    @staticmethod
    def require_role(user: User, *roles: UserRole) -> None:
        """Raise if user doesn't have one of the required roles."""
        if user.role not in roles:
            raise PermissionError(
                f"Role '{user.role.value}' not authorized. Required: {[r.value for r in roles]}"
            )

    @staticmethod
    def require_site_access(user: User, site_id: str) -> None:
        """Raise if user doesn't belong to the requested site."""
        if str(user.site_id) != str(site_id):
            raise PermissionError("Access denied: wrong site")

    # ── Password management ───────────────────────────────────────────────────

    async def change_password(
        self,
        db: AsyncSession,
        user_id: str,
        old_password: str,
        new_password: str,
    ) -> None:
        result = await db.execute(select(User).where(User.id == user_id))
        user = result.scalar_one_or_none()
        if not user or not user.hashed_password:
            raise ValueError("User not found")
        if not _verify_password(old_password, user.hashed_password):
            raise ValueError("Current password incorrect")
        if len(new_password) < 8:
            raise ValueError("Password must be at least 8 characters")
        user.hashed_password = _hash_password(new_password)
        await db.commit()

    # ── Admin: promote user role ──────────────────────────────────────────────

    async def set_user_role(
        self,
        db: AsyncSession,
        target_user_id: str,
        new_role: UserRole,
        acting_user: User,
    ) -> User:
        """Admin-only: change a user's role."""
        self.require_role(acting_user, UserRole.ADMIN)
        result = await db.execute(select(User).where(User.id == target_user_id))
        target = result.scalar_one_or_none()
        if not target:
            raise ValueError("User not found")
        target.role = new_role
        await db.commit()
        await db.refresh(target)
        log.info(f"Role changed: user={target_user_id} → {new_role.value} by admin={acting_user.id}")
        return target

    # ── List users ────────────────────────────────────────────────────────────

    async def list_site_users(
        self,
        db: AsyncSession,
        site_id: str,
        role: UserRole | None = None,
    ) -> list[User]:
        filters = [User.site_id == site_id, User.is_active == True]
        if role:
            filters.append(User.role == role)
        from sqlalchemy import and_
        result = await db.execute(
            select(User).where(and_(*filters)).order_by(User.role, User.email)
        )
        return list(result.scalars().all())

    async def deactivate_user(
        self,
        db: AsyncSession,
        target_user_id: str,
        acting_user: User,
    ) -> None:
        self.require_role(acting_user, UserRole.ADMIN, UserRole.SUPERVISOR)
        result = await db.execute(select(User).where(User.id == target_user_id))
        user = result.scalar_one_or_none()
        if not user:
            raise ValueError("User not found")
        user.is_active = False
        await db.commit()


# ── FastAPI dependency ────────────────────────────────────────────────────────

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

_bearer = HTTPBearer(auto_error=False)

async def get_current_user_dep(
    credentials: HTTPAuthorizationCredentials | None = Depends(_bearer),
    db: AsyncSession = None,  # injected via Depends(get_async_db) in routes
) -> User:
    """
    FastAPI dependency for protected routes.
    Usage in route:
        current_user: User = Depends(get_current_user_dep)
    Routes must use the full dependency — see api/ routers for pattern.
    """
    if not credentials:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Not authenticated")
    try:
        return await auth_service.get_current_user(db, credentials.credentials)
    except (ValueError, PermissionError) as e:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail=str(e))


# ── Singleton ─────────────────────────────────────────────────────────────────
auth_service = AuthService()
