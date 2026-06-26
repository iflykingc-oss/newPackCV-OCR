#!/usr/bin/env python3
"""
PackCV CLI - 开发者命令行工具

Usage:
    packcv health              # 健康检查
    packcv scenarios           # 列出场景
    packcv extract <file>      # 提取信息
    packcv tenant list         # 列出租户
    packcv tenant create       # 创建租户
    packcv key generate <tid>  # 生成 API Key
    packcv billing <tid>       # 查看用量
    packcv mask <text>         # 脱敏测试
    packcv webhook list <tid>  # WebHook 列表
    packcv circuit             # 断路器状态
    packcv version             # 版本信息
"""
import os
import sys
import json
import argparse
import httpx
from typing import Optional

BASE_URL = os.getenv("PACKCV_API_URL", "http://localhost:9001")
API_KEY = os.getenv("PACKCV_API_KEY", "")


def _client() -> httpx.Client:
    headers = {}
    if API_KEY:
        headers["Authorization"] = f"Bearer {API_KEY}"
    return httpx.Client(base_url=BASE_URL, headers=headers, timeout=30.0)


def cmd_health(args: argparse.Namespace) -> None:
    """健康检查（三探针）"""
    with _client() as c:
        for probe in ("live", "ready", "startup"):
            r = c.get(f"/health/{probe}")
            status = r.json().get("status", "unknown")
            icon = "✅" if status == "ok" else "❌"
            print(f"  {icon} {probe}: {status}")

        # 断路器
        r = c.get("/health/circuit-breakers")
        cbs = r.json().get("circuit_breakers", {})
        if cbs:
            print("\n断路器:")
            for name, stats in cbs.items():
                icon = "🟢" if stats["state"] == "closed" else "🔴" if stats["state"] == "open" else "🟡"
                print(f"  {icon} {name}: {stats['state']} (failures: {stats['failure_count']})")


def cmd_scenarios(args: argparse.Namespace) -> None:
    """列出场景"""
    with _client() as c:
        r = c.get("/api/v1/workflow/scenarios")
        if r.status_code != 200:
            print(f"❌ 请求失败: {r.status_code}")
            return
        data = r.json()
        scenarios = data if isinstance(data, list) else data.get("scenarios", [])
        print(f"共 {len(scenarios)} 个场景:")
        for s in scenarios:
            name = s.get("name", s.get("scenario_id", "?"))
            desc = s.get("description", "")
            print(f"  • {name}: {desc[:60]}")


def cmd_extract(args: argparse.Namespace) -> None:
    """提取信息"""
    file_url = args.file
    scenario = args.scenario or "auto"
    question = args.question or ""

    with _client() as c:
        payload = {"file_url": file_url, "scenario": scenario}
        if question:
            payload["user_question"] = question
        r = c.post("/api/v1/extract", json=payload)
        if r.status_code != 200:
            print(f"❌ 提取失败: {r.status_code}")
            print(r.text[:500])
            return
        print(json.dumps(r.json(), indent=2, ensure_ascii=False))


def cmd_tenant_list(args: argparse.Namespace) -> None:
    """列出租户"""
    with _client() as c:
        r = c.get("/api/v1/admin/tenants")
        if r.status_code != 200:
            print(f"❌ 请求失败: {r.status_code}")
            return
        data = r.json()
        tenants = data if isinstance(data, list) else data.get("tenants", [])
        print(f"共 {len(tenants)} 个租户:")
        for t in tenants[:20]:
            tid = t.get("tenant_id", "?")
            name = t.get("name", "?")
            tier = t.get("tier", "?")
            print(f"  • {tid}: {name} ({tier})")


def cmd_tenant_create(args: argparse.Namespace) -> None:
    """创建租户"""
    with _client() as c:
        r = c.post("/api/v1/admin/tenants", json={
            "tenant_id": args.id,
            "name": args.name or args.id,
            "tier": args.tier or "BASIC",
            "contact_email": args.email or "",
        })
        print(f"状态码: {r.status_code}")
        print(json.dumps(r.json(), indent=2, ensure_ascii=False))


def cmd_key_generate(args: argparse.Namespace) -> None:
    """生成 API Key"""
    with _client() as c:
        r = c.post(f"/api/v1/admin/tenants/{args.tenant_id}/api-keys", json={
            "name": args.name or "cli-generated",
            "expires_days": args.days or 365,
        })
        print(f"状态码: {r.status_code}")
        data = r.json()
        if "api_key" in data:
            print(f"API Key: {data['api_key']}")
            print("⚠️  请妥善保存，此 Key 仅显示一次！")
        else:
            print(json.dumps(data, indent=2, ensure_ascii=False))


def cmd_billing(args: argparse.Namespace) -> None:
    """查看用量"""
    with _client() as c:
        r = c.get(f"/api/v1/billing/usage/{args.tenant_id}")
        if r.status_code != 200:
            print(f"❌ 请求失败: {r.status_code}")
            return
        print(json.dumps(r.json(), indent=2, ensure_ascii=False))


def cmd_mask(args: argparse.Namespace) -> None:
    """脱敏测试"""
    with _client() as c:
        r = c.post("/api/v1/security/mask", json={
            "text": args.text,
            "types": ["id_card", "phone", "bank_card", "email", "name"],
        })
        if r.status_code != 200:
            print(f"❌ 请求失败: {r.status_code}")
            return
        data = r.json()
        print(f"原文: {args.text}")
        if "masked_text" in data:
            print(f"脱敏: {data['masked_text']}")
        else:
            print(json.dumps(data, indent=2, ensure_ascii=False))


def cmd_webhook_list(args: argparse.Namespace) -> None:
    """WebHook 列表"""
    with _client() as c:
        r = c.get(f"/webhooks/list/{args.tenant_id}")
        if r.status_code != 200:
            print(f"❌ 请求失败: {r.status_code}")
            return
        data = r.json()
        count = data.get("count", 0)
        print(f"共 {count} 个 WebHook 订阅:")
        for s in data.get("subscriptions", []):
            sid = s.get("subscription_id", "?")[:8]
            events = ", ".join(s.get("events", []))
            url = s.get("url", "?")
            print(f"  • {sid}..: {events} → {url[:50]}")


def cmd_circuit(args: argparse.Namespace) -> None:
    """断路器状态"""
    with _client() as c:
        r = c.get("/health/circuit-breakers")
        data = r.json()
        cbs = data.get("circuit_breakers", {})
        if not cbs:
            print("暂无断路器")
            return
        for name, stats in cbs.items():
            icon = "🟢" if stats["state"] == "closed" else "🔴" if stats["state"] == "open" else "🟡"
            print(f"{icon} {name}:")
            print(f"    状态: {stats['state']}")
            print(f"    失败数: {stats['failure_count']}/{stats['failure_threshold']}")
            print(f"    失败率: {stats['failure_rate']*100:.1f}%")
            print(f"    连续成功: {stats['consecutive_successes']}")


def cmd_version(args: argparse.Namespace) -> None:
    """版本信息"""
    print("PackCV-OCR CLI v7.0.0")
    print(f"API: {BASE_URL}")
    print(f"Python: {sys.version.split()[0]}")


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="packcv",
        description="PackCV-OCR 开发者命令行工具",
    )
    sub = parser.add_subparsers(dest="command", help="子命令")

    # health
    p = sub.add_parser("health", help="健康检查")
    p.set_defaults(func=cmd_health)

    # scenarios
    p = sub.add_parser("scenarios", help="列出场景")
    p.set_defaults(func=cmd_scenarios)

    # extract
    p = sub.add_parser("extract", help="提取信息")
    p.add_argument("file", help="文件URL")
    p.add_argument("--scenario", "-s", default="auto", help="场景")
    p.add_argument("--question", "-q", help="用户问题")
    p.set_defaults(func=cmd_extract)

    # tenant
    p_tenant = sub.add_parser("tenant", help="租户管理")
    tp = p_tenant.add_subparsers(dest="subcommand")
    p = tp.add_parser("list", help="列出租户")
    p.set_defaults(func=cmd_tenant_list)
    p = tp.add_parser("create", help="创建租户")
    p.add_argument("--id", required=True, help="租户ID")
    p.add_argument("--name", help="租户名称")
    p.add_argument("--tier", default="BASIC", help="套餐等级")
    p.add_argument("--email", help="联系邮箱")
    p.set_defaults(func=cmd_tenant_create)

    # key
    p = sub.add_parser("key", help="API Key 管理")
    p.add_argument("action", choices=["generate"], help="操作")
    p.add_argument("--tenant-id", "-t", required=True, help="租户ID")
    p.add_argument("--name", default="cli-key", help="Key名称")
    p.add_argument("--days", type=int, default=365, help="有效天数")
    p.set_defaults(func=cmd_key_generate)

    # billing
    p = sub.add_parser("billing", help="查看用量")
    p.add_argument("tenant_id", help="租户ID")
    p.set_defaults(func=cmd_billing)

    # mask
    p = sub.add_parser("mask", help="脱敏测试")
    p.add_argument("text", help="待脱敏文本")
    p.set_defaults(func=cmd_mask)

    # webhook
    p = sub.add_parser("webhook", help="WebHook 管理")
    p.add_argument("action", choices=["list"], help="操作")
    p.add_argument("--tenant-id", "-t", required=True, help="租户ID")
    p.set_defaults(func=cmd_webhook_list)

    # circuit
    p = sub.add_parser("circuit", help="断路器状态")
    p.set_defaults(func=cmd_circuit)

    # version
    p = sub.add_parser("version", help="版本信息")
    p.set_defaults(func=cmd_version)

    args = parser.parse_args()
    if not hasattr(args, "func"):
        parser.print_help()
        sys.exit(1)

    args.func(args)


if __name__ == "__main__":
    main()
