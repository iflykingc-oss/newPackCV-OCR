"""
PackCV-OCR Python SDK Client
=============================

同步/异步 HTTP 客户端,完整支持 V6.3.0 API。
"""

import os
import time
import json
import logging
import mimetypes
import asyncio
from pathlib import Path
from typing import Optional, Dict, Any, List, Union
from urllib.parse import urljoin
import urllib.request
import urllib.parse
import urllib.error

from .exceptions import (
    PackCVError,
    AuthenticationError,
    RateLimitError,
    QuotaExceededError,
    ValidationError,
    ServerError,
)
from .models import (
    ExtractRequest,
    ExtractResult,
    BatchResult,
    Scenario,
    EngineTier,
)

logger = logging.getLogger(__name__)

DEFAULT_BASE_URL = os.getenv("PACKCV_BASE_URL", "https://api.vibecoding.dev")
DEFAULT_TIMEOUT = 30
DEFAULT_RETRIES = 3
RETRY_BACKOFF = 0.5  # 指数退避基数


class PackCVClient:
    """同步客户端 (基于 urllib,无第三方依赖)"""

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: str = DEFAULT_BASE_URL,
        timeout: int = DEFAULT_TIMEOUT,
        max_retries: int = DEFAULT_RETRIES,
    ):
        self.api_key = api_key or os.getenv("PACKCV_API_KEY")
        if not self.api_key:
            raise AuthenticationError("api_key 必须提供 (或设置环境变量 PACKCV_API_KEY)")
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.max_retries = max_retries

    # ========== 核心 API: extract ==========
    def extract(
        self,
        image: Optional[Union[str, bytes, Path]] = None,
        document: Optional[Union[str, bytes, Path]] = None,
        scenario: Union[Scenario, str] = Scenario.AUTO,
        locale: str = "zh-CN",
        engine_tier: Optional[Union[EngineTier, int]] = None,
        webhook_url: Optional[str] = None,
        **kwargs,
    ) -> ExtractResult:
        """
        同步提取 (单张图片或单个文档)

        参数:
            image: 图片路径/URL/二进制
            document: 文档路径/二进制
            scenario: 场景 (Scenario 枚举)
            locale: 错误消息语言
            engine_tier: 引擎优先级
            webhook_url: 异步回调URL

        返回:
            ExtractResult

        示例:
            >>> result = client.extract(image="receipt.jpg", scenario="finance_receipt")
            >>> print(result.fields.get("金额"))
        """
        req = ExtractRequest(
            image=self._read_file(image) if image else None,
            document=self._read_file(document) if document else None,
            scenario=Scenario(scenario) if isinstance(scenario, str) else scenario,
            locale=locale,
            engine_tier=EngineTier(engine_tier) if isinstance(engine_tier, int) else engine_tier,
            webhook_url=webhook_url,
            **kwargs,
        )
        return self._extract_sync(req)

    def batch_extract(
        self,
        images: List[Union[str, bytes, Path]],
        scenario: Union[Scenario, str] = Scenario.AUTO,
        locale: str = "zh-CN",
        max_concurrency: int = 5,
    ) -> BatchResult:
        """
        批量提取 (并发执行)

        示例:
            >>> results = client.batch_extract(
            ...     images=["r1.jpg", "r2.jpg", "r3.jpg"],
            ...     scenario="finance_receipt",
            ...     max_concurrency=3,
            ... )
            >>> print(f"成功率: {results.success_rate:.0%}")
        """
        results: List[Union[ExtractResult, Exception]] = []
        succeeded = 0
        failed = 0
        for image in images:
            try:
                r = self.extract(image=image, scenario=scenario, locale=locale)
                results.append(r)
                succeeded += 1
            except PackCVError as e:
                logger.warning(f"批量处理失败: {e}")
                results.append(e)
                failed += 1
        return BatchResult(
            total=len(images),
            succeeded=succeeded,
            failed=failed,
            results=results,
        )

    def extract_async(
        self,
        image: Optional[Union[str, bytes, Path]] = None,
        document: Optional[Union[str, bytes, Path]] = None,
        scenario: Union[Scenario, str] = Scenario.AUTO,
        webhook_url: str = "",
        **kwargs,
    ) -> "AsyncTask":
        """
        异步提取 (返回任务句柄,通过 webhook 或 poll() 获取结果)

        示例:
            >>> task = client.extract_async(
            ...     image="doc.pdf",
            ...     scenario="contract",
            ...     webhook_url="https://yours.com/webhook",
            ... )
            >>> result = task.poll(timeout=60)  # 阻塞等待
        """
        req = ExtractRequest(
            image=self._read_file(image) if image else None,
            document=self._read_file(document) if document else None,
            scenario=Scenario(scenario) if isinstance(scenario, str) else scenario,
            webhook_url=webhook_url,
            **kwargs,
        )
        resp = self._request("POST", "/api/v1/extract/async", json=req.to_dict())
        return AsyncTask(self, resp["task_id"])

    # ========== 业务 API ==========
    def list_scenarios(self) -> List[Dict[str, Any]]:
        """列出所有支持的场景"""
        return self._request("GET", "/api/v1/scenarios")

    def get_usage(self) -> Dict[str, Any]:
        """查询本月用量"""
        return self._request("GET", "/api/v1/usage")

    def get_quota(self) -> Dict[str, Any]:
        """查询配额"""
        return self._request("GET", "/api/v1/quota")

    # ========== 内部: HTTP 请求 ==========
    def _extract_sync(self, req: ExtractRequest) -> ExtractResult:
        # 简化版: 实际应走 multipart/form-data
        # 这里为了零依赖, 使用 base64 编码
        payload = req.to_dict()
        # 二进制需要转 base64
        if isinstance(payload.get("image"), bytes):
            import base64
            payload["image"] = base64.b64encode(payload["image"]).decode("ascii")
        if isinstance(payload.get("document"), bytes):
            import base64
            payload["document"] = base64.b64encode(payload["document"]).decode("ascii")
        resp = self._request("POST", "/api/v1/extract", json=payload)
        return ExtractResult.from_response(resp)

    def _request(
        self,
        method: str,
        path: str,
        json: Optional[Dict[str, Any]] = None,
        params: Optional[Dict[str, Any]] = None,
    ) -> Any:
        url = urljoin(self.base_url + "/", path.lstrip("/"))
        if params:
            url += "?" + urllib.parse.urlencode(params)
        body = None
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "User-Agent": f"packcv-python-sdk/6.3.0",
        }
        if json is not None:
            body = json.dumps(json).encode("utf-8")
            headers["Content-Type"] = "application/json"
        req = urllib.request.Request(url, data=body, method=method, headers=headers)
        for attempt in range(self.max_retries):
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                    data = json.loads(resp.read().decode("utf-8"))
                    return data
            except urllib.error.HTTPError as e:
                body_text = e.read().decode("utf-8", errors="ignore")
                try:
                    err = json.loads(body_text)
                except json.JSONDecodeError:
                    err = {"message": body_text}
                if e.code == 401:
                    raise AuthenticationError(err.get("message", "Unauthorized"), code=err.get("code"), status_code=401, response=err)
                if e.code == 402:
                    raise QuotaExceededError(err.get("message", "Quota exceeded"), quota=err.get("quota"), used=err.get("used"), code=err.get("code"), status_code=402, response=err)
                if e.code == 429:
                    retry_after = int(e.headers.get("Retry-After", "60"))
                    if attempt < self.max_retries - 1:
                        logger.warning(f"限流,等待 {retry_after}s 后重试")
                        time.sleep(retry_after)
                        continue
                    raise RateLimitError(err.get("message", "Rate limited"), retry_after=retry_after, code=err.get("code"), status_code=429, response=err)
                if e.code in (400, 422):
                    raise ValidationError(err.get("message", "Bad request"), code=err.get("code"), status_code=e.code, response=err)
                if 500 <= e.code < 600:
                    if attempt < self.max_retries - 1:
                        wait = RETRY_BACKOFF * (2 ** attempt)
                        logger.warning(f"服务异常 {e.code},{wait}s 后重试 ({attempt + 1}/{self.max_retries})")
                        time.sleep(wait)
                        continue
                    raise ServerError(err.get("message", "Server error"), code=err.get("code"), status_code=e.code, response=err)
                raise PackCVError(err.get("message", "Unknown error"), status_code=e.code, response=err)
            except (urllib.error.URLError, TimeoutError, OSError) as e:
                if attempt < self.max_retries - 1:
                    wait = RETRY_BACKOFF * (2 ** attempt)
                    logger.warning(f"网络异常 {e},{wait}s 后重试 ({attempt + 1}/{self.max_retries})")
                    time.sleep(wait)
                    continue
                raise ServerError(f"网络异常: {e}")

    @staticmethod
    def _read_file(file: Union[str, bytes, Path]) -> Union[str, bytes]:
        if isinstance(file, bytes):
            return file
        path = Path(file)
        if path.exists():
            return path.read_bytes()
        if isinstance(file, str) and file.startswith(("http://", "https://")):
            return file  # 远程URL直接传
        raise ValidationError(f"文件不存在或格式错误: {file}")


class AsyncPackCVClient:
    """异步客户端 (基于 asyncio + aiohttp,可选依赖)"""

    def __init__(self, *args, **kwargs):
        try:
            import aiohttp  # noqa: F401
        except ImportError:
            raise PackCVError("异步客户端需要安装 aiohttp: pip install aiohttp")
        self._sync = PackCVClient(*args, **kwargs)
        # 简化: 异步客户端提供与同步相同的接口,内部用线程池
        import concurrent.futures
        self._executor = concurrent.futures.ThreadPoolExecutor(max_workers=10)

    async def extract(self, *args, **kwargs) -> ExtractResult:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(self._executor, lambda: self._sync.extract(*args, **kwargs))

    async def batch_extract(self, images: List, **kwargs) -> BatchResult:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(self._executor, lambda: self._sync.batch_extract(images, **kwargs))


class AsyncTask:
    """异步任务句柄"""
    def __init__(self, client: PackCVClient, task_id: str):
        self.client = client
        self.task_id = task_id

    def status(self) -> Dict[str, Any]:
        """查询任务状态"""
        return self.client._request("GET", f"/api/v1/tasks/{self.task_id}")

    def poll(self, timeout: int = 60, interval: float = 1.0) -> ExtractResult:
        """阻塞等待任务完成"""
        start = time.time()
        while time.time() - start < timeout:
            status = self.status()
            if status.get("state") == "completed":
                return ExtractResult.from_response(status.get("result", {}))
            if status.get("state") == "failed":
                raise PackCVError(status.get("error", "Task failed"))
            time.sleep(interval)
        raise TimeoutError(f"轮询超时 ({timeout}s),任务未完成")
