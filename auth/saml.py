# auth/saml.py
"""SAML 2.0 stub — defined now, implemented later.

Role mapping: SAML group attribute → Admin / Editor / Viewer.
No invite token required for SAML users — assertion is the invite.
"""
from __future__ import annotations
from fastapi import Request
from auth.models import User


class SAMLProvider:
    """Stub — raise NotImplementedError until SAML is implemented."""

    def handle_callback(self, request: Request) -> User:
        raise NotImplementedError("SAML not yet implemented")
