from __future__ import annotations
from dataclasses import dataclass
from typing import Literal


@dataclass
class User:
    id: str
    email: str
    hashed_pw: str | None
    role: Literal["admin", "editor", "viewer"]
    provider: Literal["local", "google", "microsoft", "aws", "saml"]
    provider_id: str | None
    invited_by: str | None
    created_at: str  # ISO string from SQLite

    @classmethod
    def from_row(cls, row) -> "User":
        return cls(
            id=row["id"], email=row["email"], hashed_pw=row["hashed_pw"],
            role=row["role"], provider=row["provider"],
            provider_id=row["provider_id"], invited_by=row["invited_by"],
            created_at=row["created_at"],
        )


@dataclass
class InviteToken:
    token: str
    email: str
    role: str
    created_by: str
    expires_at: str
    used_at: str | None


@dataclass
class Session:
    id: str
    user_id: str
    jwt_token: str
    expires_at: str
