"""Signup / login."""
from __future__ import annotations

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field

from api import auth_store
from api.deps import create_access_token

router = APIRouter(prefix="/auth", tags=["auth"])


class SignupBody(BaseModel):
    email: str = Field(..., min_length=3)
    password: str = Field(..., min_length=6)
    first_name: str = Field(..., min_length=1, max_length=120)
    last_name: str = Field(..., min_length=1, max_length=120)


class LoginBody(BaseModel):
    email: str
    password: str


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user_id: str
    first_name: str | None = None
    last_name: str | None = None


@router.post("/signup", response_model=TokenResponse)
def signup(body: SignupBody) -> TokenResponse:
    try:
        user_id = auth_store.create_user(
            body.email,
            body.password,
            body.first_name,
            body.last_name,
        )
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    token = create_access_token(user_id)
    fn = body.first_name.strip()
    ln = body.last_name.strip()
    return TokenResponse(
        access_token=token,
        user_id=user_id,
        first_name=fn,
        last_name=ln,
    )


@router.post("/login", response_model=TokenResponse)
def login(body: LoginBody) -> TokenResponse:
    user_id = auth_store.verify_user(body.email, body.password)
    if not user_id:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid email or password",
        )
    token = create_access_token(user_id)
    user = auth_store.get_user_by_id(user_id)
    fn = (user or {}).get("first_name")
    ln = (user or {}).get("last_name")
    return TokenResponse(
        access_token=token,
        user_id=user_id,
        first_name=str(fn) if fn is not None else None,
        last_name=str(ln) if ln is not None else None,
    )
