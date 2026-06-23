"""MinerU 文档解析引擎适配器

支持格式: PDF, DOCX, PPTX, XLSX, Images
调用方式: REST API (生产) / CLI (开发测试)
"""

import json
import logging
import os
import subprocess
import tempfile
import time
import requests
from typing import Optional, Dict, Any, List
from urllib.parse import urlparse

from utils.document_engines.base import BaseDocumentEngine, DocumentParseResult
from utils.file.file import FileOps

logger = logging.getLogger(__name__)


class MinerUDocumentEngine(BaseDocumentEngine):
    """MinerU 文档解析引擎"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self._config = config or {}
        self._mode = self._config.get("mode", "cli")  # cli | api
        self._api_endpoint = self._config.get("endpoint", "http://localhost:8000")
        self._api_key = self._config.get("api_key", "")
        self._backend = self._config.get("backend", "pipeline")  # pipeline | vlm | hybrid

    @property
    def name(self) -> str:
        return f"MinerU[{self._mode}/{self._backend}]"

    @property
    def supported_formats(self) -> List[str]:
        return ["pdf", "docx", "pptx", "xlsx", "image", "png", "jpg", "jpeg", "tiff", "bmp"]

    def is_available(self) -> bool:
        if self._mode == "api":
            try:
                resp = requests.get(f"{self._api_endpoint}/health", timeout=5)
                return resp.status_code == 200
            except Exception:
                return False
        # CLI mode: check if mineru command exists
        try:
            import mineru
            return True
        except ImportError:
            return False

    def parse(self, file_url: str, file_type: str,
              options: Optional[Dict[str, Any]] = None) -> DocumentParseResult:
        start = time.time()
        opts = {**(options or {})}

        try:
            downloaded_path = self._download_file(file_url, file_type)
            result = self._run_mineru(downloaded_path, opts)
            elapsed = (time.time() - start) * 1000
            result.processing_time_ms = elapsed
            return result
        except Exception as e:
            elapsed = (time.time() - start) * 1000
            logger.error(f"MinerU parse failed: {e}")
            return DocumentParseResult(
                success=False,
                error=str(e),
                engine_name=self.name,
                processing_time_ms=elapsed,
            )

    def _download_file(self, file_url: str, file_type: str) -> str:
        """下载文件到临时目录"""
        if file_url.startswith("http"):
            ext = self._guess_extension(file_url, file_type)
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=ext)
            resp = requests.get(file_url, timeout=120)
            resp.raise_for_status()
            with open(tmp.name, "wb") as f:
                f.write(resp.content)
            logger.info(f"  downloaded {file_url} → {tmp.name} ({len(resp.content)} bytes)")
            return tmp.name
        elif os.path.exists(file_url):
            return file_url
        else:
            raise FileNotFoundError(f"Cannot access file: {file_url}")

    def _guess_extension(self, url: str, file_type: str) -> str:
        ext_map = {
            "pdf": ".pdf", "docx": ".docx", "pptx": ".pptx",
            "xlsx": ".xlsx", "image": ".png", "png": ".png",
            "jpg": ".jpg", "jpeg": ".jpg", "tiff": ".tiff",
        }
        return ext_map.get(file_type, ".pdf")

    def _run_mineru(self, file_path: str, options: Dict[str, Any]) -> DocumentParseResult:
        """调用 MinerU 解析"""
        # Strategy: use Python API if available, fallback to CLI
        output_dir = tempfile.mkdtemp(prefix="mineru_")

        if self._mode == "api":
            return self._call_api(file_path, output_dir)
        else:
            return self._call_cli(file_path, output_dir)

    def _call_cli(self, file_path: str, output_dir: str) -> DocumentParseResult:
        """通过 CLI 调用 MinerU"""
        import mineru
        from mineru import cli_entry

        try:
            # MinerU CLI equivalent: mineru -p <path> -o <output>
            cmd = ["mineru", "-p", file_path, "-o", output_dir, "-b", self._backend]
            logger.info(f"  running: {' '.join(cmd)}")
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            if result.returncode != 0:
                logger.error(f"  mineru stderr: {result.stderr[:500]}")
                # Try fallback with pipeline backend
                cmd[-1] = "pipeline"
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)

            if result.returncode != 0:
                raise RuntimeError(f"MinerU CLI failed: {result.stderr[:500]}")

            # Parse mineru output - it outputs JSON/Markdown files
            return self._collect_output(output_dir)

        except subprocess.TimeoutExpired:
            raise TimeoutError("MinerU execution timed out (>300s)")
        except Exception as e:
            raise RuntimeError(f"MinerU CLI error: {e}")

    def _call_api(self, file_path: str, output_dir: str) -> DocumentParseResult:
        """通过 REST API 调用 MinerU 服务"""
        with open(file_path, "rb") as f:
            files = {"file": f}
            headers = {}
            if self._api_key:
                headers["Authorization"] = f"Bearer {self._api_key}"

            resp = requests.post(
                f"{self._api_endpoint}/file_parse",
                files=files,
                headers=headers,
                timeout=600,
            )
            resp.raise_for_status()
            data = resp.json()

            return DocumentParseResult(
                markdown=data.get("markdown", ""),
                tables=[
                    {"html": t.get("html", ""), "caption": t.get("caption", "")}
                    for t in data.get("tables", [])
                ],
                images=data.get("images", []),
                metadata={
                    "pages": data.get("pages", 0),
                    "has_tables": len(data.get("tables", [])) > 0,
                    "has_formulas": data.get("has_formulas", False),
                    "has_stamps": data.get("has_stamps", False),
                    "language": data.get("language", ""),
                },
                engine_name=self.name,
                success=True,
            )

    def _collect_output(self, output_dir: str) -> DocumentParseResult:
        """收集 MinerU CLI 输出结果"""
        markdown = ""
        tables = []
        metadata = {
            "pages": 0, "has_tables": False,
            "has_formulas": False, "has_stamps": False,
            "language": ""
        }

        # MinerU outputs multiple files in output_dir
        for root, dirs, files in os.walk(output_dir):
            for fname in files:
                fpath = os.path.join(root, fname)
                try:
                    if fname.endswith(".md") and os.path.getsize(fpath) > 0:
                        with open(fpath, "r", encoding="utf-8") as f:
                            content = f.read()
                            if len(content) > len(markdown):
                                markdown = content
                    elif fname.endswith(".json"):
                        with open(fpath, "r", encoding="utf-8") as f:
                            data = json.load(f)
                            if "tables" in data:
                                tables = data.get("tables", [])
                                metadata["has_tables"] = len(tables) > 0
                            if "pages" in data:
                                metadata["pages"] = data.get("pages", 0)
                            if "language" in data:
                                metadata["language"] = data.get("language", "")
                except Exception as e:
                    logger.warning(f"  skip {fname}: {e}")

        # Cleanup
        try:
            import shutil
            shutil.rmtree(output_dir, ignore_errors=True)
        except Exception:
            pass

        return DocumentParseResult(
            markdown=markdown,
            tables=tables,
            metadata=metadata,
            engine_name=self.name,
            success=len(markdown) > 0,
            error="" if len(markdown) > 0 else "Empty MinerU output",
        )