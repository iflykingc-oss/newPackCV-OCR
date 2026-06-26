"""OIDC SSO 单点登录"""
import base64
import hashlib
import hmac
import json
import secrets
import time
import uuid
from typing import Dict, Optional
from urllib.parse import urlencode
from pydantic import BaseModel, Field


class SSOSession(BaseModel):
    """SSO 会话"""
    session_id: str = Field(default_factory=lambda: f"sess-{uuid.uuid4().hex[:16]}")
    user_id: str
    provider: str
    email: str
    access_token: str
    refresh_token: Optional[str] = None
    expires_at: float
    created_at: float = Field(default_factory=time.time)


class OIDCProvider:
    """OIDC Provider（OAuth 2.0 + Identity Token）"""

    def __init__(
        self,
        issuer: str,
        client_id: str,
        client_secret: str,
        redirect_uri: str,
        scopes: Optional[list] = None,
    ):
        self.issuer = issuer.rstrip("/")
        self.client_id = client_id
        self.client_secret = client_secret
        self.redirect_uri = redirect_uri
        self.scopes = scopes or ["openid", "profile", "email"]
        self._state_secrets: Dict[str, str] = {}
        self._sessions: Dict[str, SSOSession] = {}

    def get_authorization_url(self, state: Optional[str] = None) -> str:
        """生成授权 URL（PKCE flow）"""
        state = state or secrets.token_urlsafe(16)
        # 生成 PKCE
        code_verifier = secrets.token_urlsafe(32)
        code_challenge = base64.urlsafe_b64encode(
            hashlib.sha256(code_verifier.encode()).digest()
        ).rstrip(b"=").decode()
        self._state_secrets[state] = code_verifier

        params = {
            "response_type": "code",
            "client_id": self.client_id,
            "redirect_uri": self.redirect_uri,
            "scope": " ".join(self.scopes),
            "state": state,
            "code_challenge": code_challenge,
            "code_challenge_method": "S256",
        }
        return f"{self.issuer}/authorize?{urlencode(params)}"

    def exchange_code(self, code: str, state: str) -> SSOSession:
        """用授权码换 Token（PKCE 校验）"""
        code_verifier = self._state_secrets.pop(state, None)
        if not code_verifier:
            raise ValueError("无效的 state")

        # 实际场景：调用 provider token endpoint 换 token
        # 此处模拟返回
        access_token = secrets.token_urlsafe(32)
        refresh_token = secrets.token_urlsafe(32)
        expires_at = time.time() + 3600

        session = SSOSession(
            user_id=f"user-{uuid.uuid4().hex[:8]}",
            provider=self.issuer,
            email=f"user-{secrets.token_hex(4)}@example.com",
            access_token=access_token,
            refresh_token=refresh_token,
            expires_at=expires_at,
        )
        self._sessions[session.session_id] = session
        return session

    def get_session(self, session_id: str) -> Optional[SSOSession]:
        """获取会话"""
        sess = self._sessions.get(session_id)
        if sess and sess.expires_at > time.time():
            return sess
        return None

    def revoke_session(self, session_id: str) -> bool:
        """撤销会话"""
        return self._sessions.pop(session_id, None) is not None
