"""
PackCV-OCR 异步客户端（asyncio + httpx）
"""
import asyncio
import os
from typing import Any, Dict, List, Optional

import httpx

from packcv.client import PackCVClient
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


class AsyncPackCVClient:
    """PackCV-OCR 异步客户端

    Examples:
        >>> async with AsyncPackCVClient(api_key="pk_xxx") as client:
        ...     result = await client.extract_simple("https://example.com/id.jpg")
    """

    def __init__(
        self,
        api_key: str,
        api_secret: Optional[str] = None,
        base_url: Optional[str] = None,
        timeout: float = PackCVClient.DEFAULT_TIMEOUT,
        max_retries: int = PackCVClient.MAX_RETRIES,
        max_concurrency: int = 10,
    ):
        self.api_key = api_key
        self.api_secret = api_secret
        self.base_url = base_url or os.getenv("PACKCV_BASE_URL", PackCVClient.DEFAULT_BASE_URL).rstrip("/")
        self.timeout = timeout
        self.max_retries = max_retries
        self._semaphore = asyncio.Semaphore(max_concurrency)
        self._client: Optional[httpx.AsyncClient] = None

    def _default_headers(self) -> Dict[str, str]:
        h = {"User-Agent": "packcv-ocr-python-async/1.0.0", "Content-Type": "application/json"}
        h["X-API-Key"] = self.api_key
        if self.api_secret:
            h["X-API-Secret"] = self.api_secret
        return h

    async def __aenter__(self):
        self._client = httpx.AsyncClient(
            base_url=self.base_url,
            timeout=self.timeout,
            headers=self._default_headers(),
        )
        return self

    async def __aexit__(self, *args):
        if self._client:
            await self._client.aclose()

    async def _request(self, method: str, path: str, **kwargs) -> Dict[str, Any]:
        if not self._client:
            raise PackCVError("客户端未初始化，请使用 async with 语法")
        url = path if path.startswith("http") else f"{self.base_url}{path}"
        last_err: Optional[Exception] = None

        for attempt in range(self.max_retries + 1):
            try:
                async with self._semaphore:
                    resp = await self._client.request(method, url, **kwargs)
                self._raise_for_status(resp)
                if resp.status_code == 204 or not resp.content:
                    return {}
                return resp.json()
            except (httpx.TimeoutException, httpx.NetworkError) as e:
                last_err = e
                if attempt < self.max_retries:
                    await asyncio.sleep(2 ** attempt)
                    continue
            except RateLimitError as e:
                if e.retry_after and attempt < self.max_retries:
                    await asyncio.sleep(e.retry_after)
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

    async def health(self) -> Dict[str, Any]:
        return await self._request("GET", "/api/v1/health")

    async def extract(self, request: ExtractRequest) -> ExtractResponse:
        data = await self._request("POST", "/api/v1/extract", json=request.model_dump(mode="json"))
        return ExtractResponse(**data)

    async def extract_simple(self, file_url: str, scenario: Optional[str] = None, user_question: str = "") -> ExtractResponse:
        return await self.extract(ExtractRequest(file_url=file_url, scenario=scenario, user_question=user_question))

    async def qa(self, question: str, file_url: Optional[str] = None) -> QAResponse:
        payload: Dict[str, Any] = {"question": question}
        if file_url:
            payload["file_url"] = file_url
        data = await self._request("POST", "/api/v1/qa", json=payload)
        return QAResponse(**data)

    async def batch_extract(self, file_urls: List[str], scenario: Optional[str] = None) -> List[ExtractResponse]:
        data = await self._request("POST", "/api/v1/batch", json={"file_urls": file_urls, "scenario": scenario})
        return [ExtractResponse(**r) for r in data.get("results", [])]

    async def list_scenarios(self) -> List[ScenarioInfo]:
        data = await self._request("GET", "/api/v1/scenarios")
        if isinstance(data, list):
            return [ScenarioInfo(**s) for s in data]
        return [ScenarioInfo(**s) for s in data.get("scenarios", data.get("data", []))]

    async def get_usage(self, year_month: Optional[str] = None) -> UsageInfo:
        params = {"year_month": year_month} if year_month else {}
        data = await self._request("GET", "/api/v1/usage", params=params)
        return UsageInfo(**data)

    async def me(self) -> TenantInfo:
        data = await self._request("GET", "/api/v1/me")
        return TenantInfo(**data)
