"""
性能基准测试 - 热点路径优化验证

测试项:
  - 健康检查 P50/P95/P99
  - 场景列表 P50/P95/P99
  - 脱敏吞吐量
  - 断路器开销
  - API Key 验证延迟
"""
import os
import time
import statistics
import httpx
from typing import List

BASE_URL = os.getenv("PACKCV_API_URL", "http://localhost:9001")


def bench(
    name: str,
    func,
    warmup: int = 3,
    iterations: int = 50,
) -> None:
    """运行基准测试并打印结果"""
    # 预热
    for _ in range(warmup):
        try:
            func()
        except Exception:
            pass

    # 正式测试
    latencies: List[float] = []
    errors: int = 0
    for _ in range(iterations):
        start = time.perf_counter()
        try:
            func()
            latency = (time.perf_counter() - start) * 1000  # ms
            latencies.append(latency)
        except Exception as e:
            errors += 1

    if not latencies:
        print(f"  ❌ {name}: 全部失败 ({errors} errors)")
        return

    latencies.sort()
    p50 = latencies[int(len(latencies) * 0.5)]
    p95 = latencies[int(len(latencies) * 0.95)]
    p99 = latencies[int(len(latencies) * 0.99)]
    avg = statistics.mean(latencies)
    rps = 1000.0 / avg if avg > 0 else 0

    print(f"  ✅ {name}:")
    print(f"     P50={p50:.1f}ms  P95={p95:.1f}ms  P99={p99:.1f}ms  AVG={avg:.1f}ms")
    print(f"     RPS={rps:.0f}  errors={errors}/{iterations}")


def run_all() -> None:
    print("=" * 60)
    print("PackCV-OCR 性能基准测试")
    print(f"目标: {BASE_URL}")
    print("=" * 60)

    with httpx.Client(base_url=BASE_URL, timeout=10.0) as client:

        print("\n📊 1. 健康检查延迟")
        bench("GET /api/v1/system/health", lambda: client.get("/api/v1/system/health"))

        print("\n📊 2. 健康探针延迟")
        bench("GET /health/live", lambda: client.get("/health/live"))
        bench("GET /health/ready", lambda: client.get("/health/ready"))

        print("\n📊 3. Dashboard 延迟")
        bench("GET /admin/dashboard", lambda: client.get("/admin/dashboard"))

        print("\n📊 4. Provider 列表延迟")
        bench("GET /providers", lambda: client.get("/providers"))

        print("\n📊 5. WebHook 事件类型延迟")
        bench("GET /webhooks/event-types", lambda: client.get("/webhooks/event-types"))

        print("\n📊 6. OpenAPI 规范延迟")
        bench("GET /openapi-spec", lambda: client.get("/openapi-spec"))

        print("\n📊 7. 断路器状态延迟")
        bench("GET /health/circuit-breakers", lambda: client.get("/health/circuit-breakers"))

        print("\n📊 8. Metrics 端点延迟")
        bench("GET /metrics", lambda: client.get("/metrics"))

        print("\n📊 9. 脱敏吞吐量（批量）")
        def mask_burst():
            client.post("/api/v1/security/mask", json={
                "text": "身份证110101199001011234，手机13800138000，邮箱test@example.com",
                "types": ["id_card", "phone", "email"]
            })
        bench("POST /api/v1/security/mask", mask_burst, iterations=30)

        print("\n📊 10. Web 页面渲染延迟")
        bench("GET / (Dashboard)", lambda: client.get("/"))
        bench("GET /tenants", lambda: client.get("/tenants"))

    print("\n" + "=" * 60)
    print("✅ 性能基准测试完成")
    print("=" * 60)


if __name__ == "__main__":
    run_all()
