"""
PackCV 压测脚本 (Locust)

使用:
    pip install locust
    locust -f scripts/loadtest/locustfile.py --host=http://localhost:9001

Web UI: http://localhost:8089
"""
import random
import time

from locust import HttpUser, task, between, events


# 测试图片 URL 池（mock）
SAMPLE_IMAGES = [
    "https://coze-coding-mockdata.tos-cn-beijing.volces.com/img_ekyv3.png",
    "https://example.com/sample1.jpg",
    "https://example.com/sample2.jpg",
]

SCENARIOS = ["id_card", "business_license", "invoice", "contract", "auto"]
QUESTIONS = [
    "提取姓名和身份证号",
    "法人是谁？",
    "提取金额",
    "提取有效期",
    "提取所有信息",
]


class PackCVUser(HttpUser):
    """模拟真实用户负载"""

    wait_time = between(1, 3)  # 请求间隔 1-3 秒

    def on_start(self):
        """每个用户启动时执行一次"""
        self.api_key = "pk_loadtest_" + str(random.randint(1000, 9999))
        self.client.headers.update({
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        })

    @task(10)
    def extract_document(self):
        """结构化提取（最高频）"""
        payload = {
            "input_file": {
                "url": random.choice(SAMPLE_IMAGES),
                "file_type": "image",
            },
            "scenario": random.choice(SCENARIOS),
            "user_question": random.choice(QUESTIONS),
        }
        with self.client.post(
            "/api/v1/extract",
            json=payload,
            catch_response=True,
            name="/api/v1/extract",
        ) as resp:
            if resp.status_code == 200:
                data = resp.json()
                if "structured_data" in data and "confidence" in data.get("structured_data", {}):
                    resp.success()
                else:
                    resp.failure("Missing structured_data")
            elif resp.status_code == 429:
                resp.failure("Rate limited")
            else:
                resp.failure(f"HTTP {resp.status_code}")

    @task(5)
    def qa(self):
        """问答（中频）"""
        payload = {
            "input_file": {
                "url": random.choice(SAMPLE_IMAGES),
                "file_type": "image",
            },
            "user_question": random.choice(QUESTIONS),
        }
        self.client.post(
            "/api/v1/qa",
            json=payload,
            name="/api/v1/qa",
        )

    @task(3)
    def list_scenarios(self):
        """列出场景（低频）"""
        self.client.get("/api/v1/scenarios", name="/api/v1/scenarios")

    @task(2)
    def health_check(self):
        """健康检查（低频）"""
        self.client.get("/api/v1/system/health", name="/api/v1/system/health")

    @task(1)
    def list_providers(self):
        """列出 LLM Provider"""
        self.client.get("/providers", name="/providers")


class WebhookUser(HttpUser):
    """模拟 WebHook 订阅者"""

    wait_time = between(5, 10)

    @task
    def subscribe_and_dispatch(self):
        sub = self.client.post(
            "/webhooks/subscribe",
            json={
                "tenant_id": "loadtest",
                "url": "https://httpbin.org/post",
                "events": ["task.completed"],
                "secret": "loadtestsecret123",
            },
            catch_response=True,
        ).json()

        if "subscription_id" in sub:
            self.client.post(
                "/webhooks/dispatch",
                json={
                    "event_type": "task.completed",
                    "tenant_id": "loadtest",
                    "data": {"task_id": "t-001"},
                },
                name="/webhooks/dispatch",
            )


# 自定义统计
stats_print_interval = 10.0  # 秒


@events.report_to_master.add_listener
def on_report_to_master(client_id, data):
    """向 master 报告自定义数据"""
    data["custom"] = {
        "test_start": time.time(),
    }


@events.test_start.add_listener
def on_test_start(environment, **kwargs):
    print("=" * 60)
    print("PackCV 负载测试启动")
    print(f"目标: {environment.host}")
    print("=" * 60)


@events.test_stop.add_listener
def on_test_stop(environment, **kwargs):
    print("=" * 60)
    print("PackCV 负载测试结束")
    print("=" * 60)
