"""
Webhook 签名工具

使用 HMAC-SHA256 对事件载荷进行签名，确保投递过程中的完整性和真实性。
"""
import hashlib
import hmac
import time
from typing import Optional, Tuple


class WebhookSigner:
    """HMAC-SHA256 签名器

    签名格式: v1=<hex_digest>
    Header: X-PackCV-Signature: t=<timestamp>,v1=<hex_digest>
    """

    VERSION = "v1"

    def __init__(self, secret: str) -> None:
        if not secret or len(secret) < 8:
            raise ValueError("secret 必须至少 8 字符")
        self._secret = secret.encode("utf-8")

    def sign(self, payload: bytes, timestamp: Optional[int] = None) -> str:
        """对载荷进行签名

        Args:
            payload: 原始字节载荷
            timestamp: Unix 时间戳，默认当前时间

        Returns:
            形如 "t=1234567890,v1=abc..." 的签名字符串
        """
        ts = int(timestamp) if timestamp is not None else int(time.time())
        signed_payload = b"%d.%b" % (ts, payload)
        digest = hmac.new(self._secret, signed_payload, hashlib.sha256).hexdigest()
        return f"t={ts},{self.VERSION}={digest}"

    def sign_dict(self, data: dict) -> str:
        """对字典数据进行签名"""
        import json
        payload = json.dumps(data, sort_keys=True, separators=(",", ":")).encode("utf-8")
        return self.sign(payload)


def verify_signature(
    payload: bytes,
    signature_header: str,
    secret: str,
    tolerance_seconds: int = 300,
) -> bool:
    """验证 Webhook 签名

    Args:
        payload: 接收到的原始字节载荷
        signature_header: X-PackCV-Signature 头
        secret: 共享密钥
        tolerance_seconds: 时间戳容忍度（防重放）

    Returns:
        签名是否有效
    """
    if not signature_header:
        return False

    parts = dict(p.split("=", 1) for p in signature_header.split(",") if "=" in p)
    timestamp_str = parts.get("t")
    sig_version = parts.get("v1")
    if not timestamp_str or not sig_version:
        return False

    try:
        ts = int(timestamp_str)
    except ValueError:
        return False

    # 时间戳容忍度检查
    now = int(time.time())
    if abs(now - ts) > tolerance_seconds:
        return False

    signer = WebhookSigner(secret)
    expected = signer.sign(payload, timestamp=ts)

    expected_sig = dict(p.split("=", 1) for p in expected.split(",") if "=" in p)
    expected_v1 = expected_sig.get("v1", "")

    # 使用 hmac.compare_digest 防止时序攻击
    return hmac.compare_digest(sig_version, expected_v1)


def make_signature_for_test(
    secret: str, payload: bytes, timestamp: int
) -> Tuple[int, str]:
    """测试辅助函数"""
    signer = WebhookSigner(secret)
    return timestamp, signer.sign(payload, timestamp)
