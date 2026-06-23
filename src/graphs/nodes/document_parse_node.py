# -*- coding: utf-8 -*-
"""
MinerU文档解析节点
调用MinerU引擎解析PDF/DOCX/PPTX/XLSX文档，输出Markdown/JSON
V5.9 新增
"""

import os
import json
import tempfile
import subprocess
import logging
from typing import Optional
from langchain_core.runnables import RunnableConfig
from langgraph.runtime import Runtime
from coze_coding_utils.runtime_ctx.context import Context
from graphs.state import DocumentParseInput, DocumentParseOutput
from utils.file.file import FileOps

logger = logging.getLogger(__name__)

# MinerU CLI路径
MINERU_CMD = "mineru"


def _parse_with_mineru(input_path: str, output_dir: str, backend: str = "pipeline") -> Optional[dict]:
    """调用MinerU CLI解析文档

    Args:
        input_path: 输入文件路径
        output_dir: 输出目录
        backend: 解析后端 (pipeline / auto)

    Returns:
        解析结果字典或None
    """
    try:
        cmd = [
            MINERU_CMD,
            "-p", input_path,
            "-o", output_dir,
        ]
        if backend == "pipeline":
            cmd.extend(["-b", "pipeline"])

        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=300
        )

        if result.returncode != 0:
            logger.error("MinerU解析失败: %s", result.stderr[:500])
            return None

        # MinerU输出：output_dir/<文件名>/ 下的 auto.md / auto.json
        base_name = os.path.splitext(os.path.basename(input_path))[0]
        result_dir = os.path.join(output_dir, base_name)

        parsed = {
            "markdown": "",
            "tables": [],
            "metadata": {},
            "total_pages": 0,
        }

        # 读取Markdown输出
        md_path = os.path.join(result_dir, "auto.md")
        if os.path.exists(md_path):
            with open(md_path, "r", encoding="utf-8") as f:
                parsed["markdown"] = f.read()

        # 读取JSON输出
        json_path = os.path.join(result_dir, "auto.json")
        if os.path.exists(json_path):
            with open(json_path, "r", encoding="utf-8") as f:
                json_data = json.load(f)
                tables = json_data.get("tables", []) if isinstance(json_data, dict) else []
                parsed["tables"] = tables
                parsed["metadata"] = json_data.get("metadata", {}) if isinstance(json_data, dict) else {}
                parsed["total_pages"] = json_data.get("page_count", 0) if isinstance(json_data, dict) else 0

        logger.info("MinerU解析成功: pages=%d, md_len=%d, tables=%d",
                     parsed["total_pages"], len(parsed["markdown"]), len(parsed["tables"]))

        return parsed

    except subprocess.TimeoutExpired:
        logger.error("MinerU解析超时(>300s)")
        return None
    except FileNotFoundError:
        logger.error("MinerU未安装，请执行: uv pip install mineru")
        return None
    except Exception as e:
        logger.error("MinerU解析异常: %s", e)
        return None


def document_parse_node(
    state: DocumentParseInput,
    config: RunnableConfig,
    runtime: Runtime[Context],
) -> DocumentParseOutput:
    """
    title: MinerU文档解析
    desc: 使用MinerU引擎解析PDF/DOCX/PPTX/XLSX文档，输出含表格/版面/阅读顺序的Markdown结构
    integrations: 
    """
    ctx = runtime.context

    # 下载文档到临时目录
    tmp_dir = tempfile.mkdtemp(prefix="mineru_")
    output_dir = tempfile.mkdtemp(prefix="mineru_out_")

    try:
        # 确定文件扩展名
        file_url = state.file_url
        file_ext = state.file_type or os.path.splitext(file_url.split("?")[0])[1].lower()

        # 下载文件
        local_path = FileOps.save_to_local(file_url)
        if local_path is None:
            raise ValueError(f"下载文件失败: {file_url}")
        local_path = str(local_path)

        # 引擎模式
        backend = "pipeline"
        if state.engine_mode == "auto":
            # 自动检测 - 大文件用pipeline，小文件尝试vlm
            file_size = os.path.getsize(local_path)
            backend = "pipeline" if file_size > 10 * 1024 * 1024 else "pipeline"

        # 调用MinerU
        engine_used = "mineru"
        result = _parse_with_mineru(local_path, output_dir, backend)

        if result is None:
            # 降级：返回文件URL作为文本
            logger.warning("MinerU解析失败，降级返回原始URL")
            return DocumentParseOutput(
                markdown_output=f"文件解析失败，原始URL: {file_url}",
                tables=[],
                metadata={},
                reading_order_text="",
                layout_info=[],
                total_pages=0,
                parse_time=0.0,
                engine_used=f"{engine_used}_failed",
            )

        # 提取阅读顺序文本（去除Markdown标记）
        import re
        reading_order_text = re.sub(r'[#*_~`>|\[\]()\-]', '', result["markdown"])
        reading_order_text = re.sub(r'\n{3,}', '\n\n', reading_order_text).strip()

        logger.info("文档解析完成: pages=%d, tables=%d, text_len=%d",
                     result["total_pages"], len(result["tables"]), len(result["markdown"]))

        return DocumentParseOutput(
            markdown_output=result["markdown"],
            tables=result["tables"],
            metadata=result["metadata"],
            reading_order_text=reading_order_text,
            layout_info=[],
            total_pages=result["total_pages"],
            parse_time=0.0,
            engine_used=engine_used,
        )

    except Exception as e:
        logger.error("文档解析异常: %s", e)
        return DocumentParseOutput(
            markdown_output=f"文档解析失败: {e}",
            tables=[],
            metadata={},
            reading_order_text="",
            layout_info=[],
            total_pages=0,
            parse_time=0.0,
            engine_used="error",
        )
    finally:
        # 清理临时文件
        try:
            import shutil
            shutil.rmtree(tmp_dir, ignore_errors=True)
            shutil.rmtree(output_dir, ignore_errors=True)
        except Exception:
            pass