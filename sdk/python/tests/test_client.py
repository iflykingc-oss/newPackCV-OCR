"""PackCV SDK 测试"""
import os
import sys
import base64
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
from packcv import (
    PackCVClient,
    PackCVError,
    AuthenticationError,
    QuotaExceededError,
    RateLimitError,
    ValidationError,
    ServerError,
    ExtractResult,
    Scenario,
    EngineTier,
)


class TestClientInit:
    def test_missing_api_key(self, monkeypatch):
        monkeypatch.delenv("PACKCV_API_KEY", raising=False)
        with pytest.raises(AuthenticationError):
            PackCVClient()

    def test_with_api_key(self):
        client = PackCVClient(api_key="pck_test_xxx")
        assert client.api_key == "pck_test_xxx"
        assert client.base_url.startswith("http")

    def test_from_env(self, monkeypatch):
        monkeypatch.setenv("PACKCV_API_KEY", "pck_env_yyy")
        client = PackCVClient()
        assert client.api_key == "pck_env_yyy"


class TestScenarios:
    def test_scenario_enum(self):
        assert Scenario.PACKAGING.value == "packaging"
        assert Scenario.AUTO.value == "auto"
        assert len(list(Scenario)) == 9

    def test_engine_tier_enum(self):
        assert EngineTier.CUSTOM.value == 0
        assert EngineTier.FALLBACK.value == 99


class TestExtractRequest:
    def test_to_dict(self):
        from packcv.models import ExtractRequest
        req = ExtractRequest(scenario=Scenario.PACKAGING, locale="en")
        d = req.to_dict()
        assert d["scenario"] == "packaging"
        assert d["locale"] == "en"
        assert d["engine_tier"] is None

    def test_with_engine_tier(self):
        from packcv.models import ExtractRequest
        req = ExtractRequest(scenario=Scenario.CONTRACT, engine_tier=EngineTier.PADDLEOCR_VL)
        assert req.to_dict()["engine_tier"] == 1


class TestExtractResult:
    def test_from_response(self):
        resp = {
            "request_id": "req_123",
            "scenario": "packaging",
            "fields": {"品牌": "ACME", "保质期": "2025-12-31"},
            "confidence": 0.95,
            "engine_used": "paddleocr_vl",
            "latency_ms": 1200,
        }
        result = ExtractResult.from_response(resp)
        assert result.request_id == "req_123"
        assert result.scenario == "packaging"
        assert result.fields["品牌"] == "ACME"
        assert result.confidence == 0.95
        assert result.get("品牌") == "ACME"
        assert result.get("不存在", "default") == "default"


class TestBatchResult:
    def test_success_rate(self):
        from packcv.models import BatchResult, ExtractResult
        results = [
            ExtractResult(request_id="1", scenario="x", fields={}, confidence=0.9, engine_used="x", latency_ms=100),
            Exception("err"),
            ExtractResult(request_id="3", scenario="x", fields={}, confidence=0.9, engine_used="x", latency_ms=100),
        ]
        br = BatchResult(total=3, succeeded=2, failed=1, results=results)
        assert br.success_rate == 2 / 3


class TestReadFile:
    def test_bytes_input(self, tmp_path):
        client = PackCVClient(api_key="pck_test")
        data = client._read_file(b"hello")
        assert data == b"hello"

    def test_local_file(self, tmp_path):
        f = tmp_path / "test.txt"
        f.write_bytes(b"hello world")
        client = PackCVClient(api_key="pck_test")
        data = client._read_file(str(f))
        assert data == b"hello world"

    def test_url_passthrough(self):
        client = PackCVClient(api_key="pck_test")
        data = client._read_file("https://example.com/img.jpg")
        assert data == "https://example.com/img.jpg"

    def test_missing_file_raises(self):
        client = PackCVClient(api_key="pck_test")
        with pytest.raises(ValidationError):
            client._read_file("/nonexistent/file.jpg")


class TestExceptions:
    def test_packcv_error_base(self):
        e = PackCVError("test", code="X001", status_code=500)
        assert e.message == "test"
        assert e.code == "X001"
        assert e.status_code == 500
        assert "PackCVError" in repr(e)

    def test_quota_error(self):
        e = QuotaExceededError("quota", quota=100, used=100)
        assert e.quota == 100
        assert e.used == 100

    def test_rate_limit_error(self):
        e = RateLimitError("rate", retry_after=60)
        assert e.retry_after == 60

    def test_server_error_is_retryable(self):
        e = ServerError("500")
        assert e.is_retryable is True
