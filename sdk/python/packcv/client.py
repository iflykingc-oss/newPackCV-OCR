"""
PackCV-OCR 同步客户端
"""
import os
import time
from typing import Any, Dict, List, Optional

import httpx

from packcv.exceptions import (
    APIError,
    AuthenticationError,
    PackCVError,
    QuotaExceededError,
    RateLimitError,
)
from packcv.types import (
    ExtractRequest,
    ExtractResponse,
    QAResponse,
    ScenarioInfo,
    TenantInfo,
    UsageInfo,
)


class PackCVClient:
    """PackCV-OCR 同步客户端

    Examples:
        >>> client = PackCVClient(api_key="pk_xxx")
        >>> result = client.extract(file_url="https://example.com/id.jpg")
        >>> print(result.scenario, result.structured_data)
    """

    DEFAULT_BASE_URL = "https://api.packcv-ocr.com"
    DEFAULT_TIMEOUT = 60.0
    MAX_RETRIES = 3

    def __init__(
        self,
        api_key: str,
        api_secret: Optional[str] = None,
        base_url: Optional[str] = None,
        timeout: float = DEFAULT_TIMEOUT,
        max_retries: int = MAX_RETRIES,
    ):
        """
        Args:
            api_key: API Key (pk_xxx)
            api_secret: API Secret (sk_xxx, 可选)
            base_url: API 基础 URL
            timeout: 请求超时（秒）
            max_retries: 最大重试次数
        """
        self.api_key = api_key
        self.api_secret = api_secret
        self.base_url = base_url or os.getenv("PACKCV_BASE_URL", self.DEFAULT_BASE_URL).rstrip("/")
        self.timeout = timeout
        self.max_retries = max_retries

        self._client = httpx.Client(
            base_url=self.base_url,
            timeout=timeout,
            headers=self._default_headers(),
        )

    def _default_headers(self) -> Dict[str, str]:
        h = {"User-Agent": f"packcv-ocr-python/1.0.0", "Content-Type": "application/json"}
        h["X-API-Key"] = self.api_key
        if self.api_secret:
            h["X-API-Secret"] = self.api_secret
        return h

    def _request(self, method: str, path: str, **kwargs) -> Dict[str, Any]:
        """底层 HTTP 请求（带重试）"""
        url = path if path.startswith("http") else f"{self.base_url}{path}"
        last_err: Optional[Exception] = None

        for attempt in range(self.max_retries + 1):
            try:
                resp = self._client.request(method, url, **kwargs)
                self._raise_for_status(resp)
                if resp.status_code == 204 or not resp.content:
                    return {}
                return resp.json()
            except (httpx.TimeoutException, httpx.NetworkError) as e:
                last_err = e
                if attempt < self.max_retries:
                    wait = 2 ** attempt
                    time.sleep(wait)
                    continue
            except RateLimitError as e:
                if e.retry_after and attempt < self.max_retries:
                    time.sleep(e.retry_after)
                    continue
                raise
        raise APIError(f"请求失败: {last_err}")

    def _raise_for_status(self, resp: httpx.Response):
        if resp.status_code < 400:
            return
        try:
            body = resp.json()
        except Exception:
            body = {"detail": resp.text}
        msg = body.get("detail") or body.get("message") or body.get("error") or resp.text
        if resp.status_code == 401:
            raise AuthenticationError(msg, status_code=401, response=body)
        if resp.status_code == 429:
            retry_after = int(resp.headers.get("Retry-After", "1"))
            raise RateLimitError(msg, retry_after=retry_after, status_code=429, response=body)
        if resp.status_code == 402:
            raise QuotaExceededError(msg, status_code=402, response=body)
        raise APIError(msg, status_code=resp.status_code, response=body)

    # ============= 业务接口 =============

    def health(self) -> Dict[str, Any]:
        """健康检查"""
        return self._request("GET", "/api/v1/health")

    def extract(self, request: ExtractRequest) -> ExtractResponse:
        """从图片/PDF 中提取结构化信息

        Args:
            request: 提取请求参数

        Returns:
            ExtractResponse: 提取结果
        """
        data = self._request("POST", "/api/v1/extract", json=request.model_dump(mode="json"))
        return ExtractResponse(**data)

    def extract_simple(
        self,
        file_url: str,
        scenario: Optional[str] = None,
        user_question: str = "",
    ) -> ExtractResponse:
        """简化版提取接口"""
        return self.extract(ExtractRequest(file_url=file_url, scenario=scenario, user_question=user_question))

    def qa(self, question: str, file_url: Optional[str] = None) -> QAResponse:
        """文档问答

        Args:
            question: 问题
            file_url: 文档URL（可选）
        """
        payload: Dict[str, Any] = {"question": question}
        if file_url:
            payload["file_url"] = file_url
        data = self._request("POST", "/api/v1/qa", json=payload)
        return QAResponse(**data)

    def batch_extract(self, file_urls: List[str], scenario: Optional[str] = None) -> List[ExtractResponse]:
        """批量提取"""
        data = self._request(
            "POST", "/api/v1/batch",
            json={"file_urls": file_urls, "scenario": scenario},
        )
        results = data.get("results", [])
        return [ExtractResponse(**r) for r in results]

    def list_scenarios(self) -> List[ScenarioInfo]:
        """列出所有支持的场景"""
        data = self._request("GET", "/api/v1/scenarios")
        if isinstance(data, list):
            return [ScenarioInfo(**s) for s in data]
        return [ScenarioInfo(**s) for s in data.get("scenarios", data.get("data", []))]

    def get_usage(self, year_month: Optional[str] = None) -> UsageInfo:
        """查询当前租户本月用量"""
        params = {"year_month": year_month} if year_month else {}
        data = self._request("GET", "/api/v1/usage", params=params)
        return UsageInfo(**data)

    def me(self) -> TenantInfo:
        """获取当前租户信息"""
        data = self._request("GET", "/api/v1/me")
        return TenantInfo(**data)

    def close(self):
        """关闭客户端"""
        self._client.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
