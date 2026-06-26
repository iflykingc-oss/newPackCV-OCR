# -*- coding: utf-8 -*-
"""
HTTP API接口层
提供RESTful API接口
"""

import os
import json
import base64
import logging
from typing import Optional, Dict, Any
from datetime import datetime

logger = logging.getLogger(__name__)


class PackCVAPI:
    """PackCV RESTful API封装"""

    def __init__(self, base_url: str = None, api_key: str = None):
        """
        初始化API客户端

        Args:
            base_url: API服务地址
            api_key: API密钥
        """
        self.base_url = base_url or os.getenv("PACKCV_API_URL", "http://localhost:8000")
        self.api_key = api_key or os.getenv("PACKCV_API_KEY", "")
        self._session = None

    def _get_session(self):
        """获取HTTP会话"""
        if self._session is None:
            import requests
            self._session = requests.Session()
            if self.api_key:
                self._session.headers.update({"Authorization": f"Bearer {self.api_key}"})
        return self._session

    def recognize(
        self,
        image: str = None,
        image_base64: str = None,
        image_url: str = None,
        enhance: bool = True,
        detect: bool = False
    ) -> Dict[str, Any]:
        """
        识别图片

        Args:
            image: 图片路径（本地）
            image_base64: 图片Base64编码
            image_url: 图片URL
            enhance: 是否预处理增强
            detect: 是否目标检测

        Returns:
            识别结果
        """
        session = self._get_session()

        # 准备图片数据
        if image:
            with open(image, 'rb') as f:
                image_base64 = base64.b64encode(f.read()).decode()
        elif image_base64:
            pass
        else:
            raise ValueError("必须提供 image, image_base64 或 image_url")

        payload = {
            "image_base64": image_base64,
            "enhance": enhance,
            "detect": detect
        }

        try:
            response = session.post(
                f"{self.base_url}/api/v1/recognize",
                json=payload,
                timeout=60
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"API调用失败: {e}")
            raise

    def recognize_batch(
        self,
        images: list,
        enhance: bool = True
    ) -> Dict[str, Any]:
        """
        批量识别

        Args:
            images: 图片列表
            enhance: 是否预处理增强

        Returns:
            任务ID
        """
        session = self._get_session()

        payload = {
            "images": images,
            "enhance": enhance
        }

        try:
            response = session.post(
                f"{self.base_url}/api/v1/recognize/batch",
                json=payload,
                timeout=300
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"API调用失败: {e}")
            raise

    def get_task_status(self, task_id: str) -> Dict[str, Any]:
        """获取任务状态"""
        session = self._get_session()

        try:
            response = session.get(
                f"{self.base_url}/api/v1/tasks/{task_id}",
                timeout=10
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"获取任务状态失败: {e}")
            raise

    def get_result(self, task_id: str) -> Dict[str, Any]:
        """获取任务结果"""
        session = self._get_session()

        try:
            response = session.get(
                f"{self.base_url}/api/v1/tasks/{task_id}/result",
                timeout=10
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"获取任务结果失败: {e}")
            raise

    def get_alerts(
        self,
        status: str = None,
        alert_level: str = None,
        limit: int = 100
    ) -> Dict[str, Any]:
        """获取告警列表"""
        session = self._get_session()

        params = {"limit": limit}
        if status:
            params["status"] = status
        if alert_level:
            params["alert_level"] = alert_level

        try:
            response = session.get(
                f"{self.base_url}/api/v1/alerts",
                params=params,
                timeout=10
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"获取告警列表失败: {e}")
            raise

    def acknowledge_alert(
        self,
        alert_id: str,
        acknowledged_by: str = None
    ) -> Dict[str, Any]:
        """确认告警"""
        session = self._get_session()

        payload = {"acknowledged_by": acknowledged_by}

        try:
            response = session.post(
                f"{self.base_url}/api/v1/alerts/{alert_id}/acknowledge",
                json=payload,
                timeout=10
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"确认告警失败: {e}")
            raise

    def generate_report(
        self,
        report_type: str = "expiry",
        period_start: str = None,
        period_end: str = None,
        format: str = "json"
    ) -> Dict[str, Any]:
        """生成报表"""
        session = self._get_session()

        payload = {
            "report_type": report_type,
            "format": format
        }
        if period_start:
            payload["period_start"] = period_start
        if period_end:
            payload["period_end"] = period_end

        try:
            response = session.post(
                f"{self.base_url}/api/v1/reports/generate",
                json=payload,
                timeout=300
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"生成报表失败: {e}")
            raise

    def get_config(self) -> Dict[str, Any]:
        """获取配置"""
        session = self._get_session()

        try:
            response = session.get(
                f"{self.base_url}/api/v1/config",
                timeout=10
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"获取配置失败: {e}")
            raise

    def update_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """更新配置"""
        session = self._get_session()

        try:
            response = session.put(
                f"{self.base_url}/api/v1/config",
                json=config,
                timeout=10
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"更新配置失败: {e}")
            raise


# ============ API服务端实现 ============

class PackCVAPIServer:
    """PackCV API服务器（用于集成到Flask/FastAPI）"""

    @staticmethod
    def create_routes(app, workflow_runner):
        """
        创建API路由

        Args:
            app: Flask/FastAPI应用实例
            workflow_runner: 工作流运行器
        """
        from flask import request, jsonify

        @app.route("/api/v1/recognize", methods=["POST"])
        def recognize():
            """单图识别接口"""
            data = request.get_json()
            image_base64 = data.get("image_base64")
            enhance = data.get("enhance", True)
            detect = data.get("detect", False)

            if not image_base64:
                return jsonify({"error": "缺少图片数据"}), 400

            try:
                result = workflow_runner.recognize(
                    image_base64=image_base64,
                    enhance=enhance,
                    detect=detect
                )
                return jsonify(result)
            except Exception as e:
                logger.error(f"识别失败: {e}")
                return jsonify({"error": str(e)}), 500

        @app.route("/api/v1/recognize/batch", methods=["POST"])
        def recognize_batch():
            """批量识别接口"""
            data = request.get_json()
            images = data.get("images", [])
            enhance = data.get("enhance", True)

            if not images:
                return jsonify({"error": "缺少图片数据"}), 400

            try:
                task_id = workflow_runner.recognize_batch(
                    images=images,
                    enhance=enhance
                )
                return jsonify({"task_id": task_id})
            except Exception as e:
                logger.error(f"批量识别失败: {e}")
                return jsonify({"error": str(e)}), 500

        @app.route("/api/v1/tasks/<task_id>", methods=["GET"])
        def get_task_status(task_id):
            """获取任务状态"""
            status = workflow_runner.get_task_status(task_id)
            return jsonify(status)

        @app.route("/api/v1/tasks/<task_id>/result", methods=["GET"])
        def get_task_result(task_id):
            """获取任务结果"""
            result = workflow_runner.get_task_result(task_id)
            return jsonify(result)

        @app.route("/api/v1/alerts", methods=["GET"])
        def get_alerts():
            """获取告警列表"""
            status = request.args.get("status")
            alert_level = request.args.get("alert_level")
            limit = int(request.args.get("limit", 100))

            alerts = workflow_runner.get_alerts(
                status=status,
                alert_level=alert_level,
                limit=limit
            )
            return jsonify(alerts)

        @app.route("/api/v1/alerts/<alert_id>/acknowledge", methods=["POST"])
        def acknowledge_alert(alert_id):
            """确认告警"""
            data = request.get_json() or {}
            acknowledged_by = data.get("acknowledged_by")

            result = workflow_runner.acknowledge_alert(
                alert_id=alert_id,
                acknowledged_by=acknowledged_by
            )
            return jsonify(result)

        @app.route("/api/v1/reports/generate", methods=["POST"])
        def generate_report():
            """生成报表"""
            data = request.get_json() or {}
            report_type = data.get("report_type", "expiry")
            period_start = data.get("period_start")
            period_end = data.get("period_end")
            format = data.get("format", "json")

            result = workflow_runner.generate_report(
                report_type=report_type,
                period_start=period_start,
                period_end=period_end,
                format=format
            )
            return jsonify(result)

        @app.route("/api/v1/config", methods=["GET"])
        def get_config():
            """获取配置"""
            config = workflow_runner.get_config()
            return jsonify(config)

        @app.route("/api/v1/config", methods=["PUT"])
        def update_config():
            """更新配置"""
            data = request.get_json()
            config = workflow_runner.update_config(data)
            return jsonify(config)

        @app.route("/api/v1/health", methods=["GET"])
        def health():
            """健康检查"""
            return jsonify({"status": "healthy", "timestamp": datetime.now().isoformat()})
