#!/usr/bin/env python3
"""
自动标注基准数据集（V5.4 SP + LLM）
用途：
  1. 遍历 benchmark 图片 → 用 V5.4 SP + LLM 生成结构化标注
  2. 保存为 ground_truth JSON（作为回归测试黄金标准）
  3. 支持增量标注（跳过已有标注的图片）

用法：
  python scripts/auto_label_benchmark.py [--sample N] [--category all|categories...]
"""

import os, sys, json, time, argparse, glob
from datetime import datetime
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from jinja2 import Template
from langchain_core.runnables import RunnableConfig
from langgraph.runtime import Runtime
from coze_coding_utils.runtime_ctx.context import Context
from coze_coding_dev_sdk import LLMClient
from langchain_core.messages import SystemMessage, HumanMessage

# V5.4 SP（从配置文件读取）
SP_FILE = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'config', 'model_extract_llm_cfg.json')

BENCHMARK_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'assets', 'benchmark')
GROUND_TRUTH_DIR = os.path.join(BENCHMARK_DIR, 'ground_truth')
REPORT_DIR = os.path.join(BENCHMARK_DIR, 'reports')

# 支持的图片后缀
IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.webp', '.bmp'}


def load_sp():
    """加载V5.4 SP配置文件"""
    with open(SP_FILE, 'r', encoding='utf-8') as f:
        cfg = json.load(f)
    return cfg.get('sp', ''), cfg.get('up', ''), cfg.get('config', {})


def find_images(category: str = 'all', sample: int = 0):
    """扫描 benchmark 目录中的图片"""
    images = []
    
    # GroceryStoreDataset
    gs_dir = os.path.join(BENCHMARK_DIR, 'GroceryStoreDataset', 'dataset')
    if os.path.exists(gs_dir):
        for root, dirs, files in os.walk(gs_dir):
            for f in files:
                ext = os.path.splitext(f)[1].lower()
                if ext in IMAGE_EXTENSIONS:
                    rel_path = os.path.relpath(os.path.join(root, f), BENCHMARK_DIR)
                    dir_parts = rel_path.split(os.sep)
                    cat = dir_parts[1] if len(dir_parts) > 1 else 'unknown'
                    images.append({
                        'path': os.path.join(root, f),
                        'category': cat,
                        'source': 'GroceryStore'
                    })
    
    if category != 'all':
        target_cats = set(category.split(','))
        images = [img for img in images if img['category'] in target_cats]
    
    if sample > 0 and len(images) > sample:
        import random
        random.shuffle(images)
        images = images[:sample]
    
    return images


def auto_label_image(img_path: str, sp: str, up: str, llm_cfg: dict, ctx: Context) -> dict:
    """用V5.4 SP+LLM为单张图生成结构化标注"""
    from utils.file.file import File, FileOps
    
    # 读取图片内容
    img_file = File(url=img_path, file_type='image')
    text_content = FileOps.extract_text(img_file)
    
    if not text_content or len(text_content.strip()) < 5:
        return {"error": "图片内容为空或无法读取", "image": img_path}
    
    # 渲染UP模板
    up_template = Template(up)
    user_prompt = up_template.render(ocr_text=text_content)
    
    # 准备消息
    system_msg = SystemMessage(content=sp)
    human_msg = HumanMessage(content=user_prompt)
    
    # 调用LLM
    client = LLMClient(ctx=ctx)
    response = client.invoke(
        messages=[system_msg, human_msg],
        model=llm_cfg.get('model', 'doubao-seed-2-0-pro-260215'),
        temperature=0.1,
        max_tokens=2000
    )
    
    content = response.content if hasattr(response, 'content') else str(response)
    
    # 解析JSON
    content = content.strip()
    if content.startswith('```json'):
        content = content[7:]
    if content.startswith('```'):
        content = content[3:]
    if content.endswith('```'):
        content = content[:-3]
    content = content.strip()
    
    try:
        result = json.loads(content)
        return result
    except json.JSONDecodeError as e:
        return {"error": f"JSON解析失败: {e}", "raw_response": content[:500], "image": img_path}


def main():
    parser = argparse.ArgumentParser(description='自动标注基准数据集')
    parser.add_argument('--sample', type=int, default=200, help='采样数量（默认200，0=全部）')
    parser.add_argument('--category', type=str, default='all', help='品类筛选（逗号分隔）')
    parser.add_argument('--force', action='store_true', help='强制重新标注已存在的')
    parser.add_argument('--resume', action='store_true', help='断点续标（跳过已有）')
    args = parser.parse_args()
    
    os.makedirs(GROUND_TRUTH_DIR, exist_ok=True)
    os.makedirs(REPORT_DIR, exist_ok=True)
    
    # 加载SP
    sp, up, llm_cfg = load_sp()
    print(f"📋 V5.4 SP加载完成 | model={llm_cfg.get('model')}")
    
    # 查找图片
    images = find_images(args.category, args.sample)
    print(f"🔍 找到 {len(images)} 张图片")
    
    if args.resume:
        existing = set(os.listdir(GROUND_TRUTH_DIR))
        before = len(images)
        images = [img for img in images if f"{os.path.splitext(os.path.basename(img['path']))[0]}.json" not in existing]
        print(f"⏩ 跳过已标注 {before - len(images)} 张，剩余 {len(images)} 张")
    
    # 初始化Context
    ctx = Context()
    
    # 标注统计
    stats = {"total": len(images), "success": 0, "failed": 0, "errors": []}
    start_time = time.time()
    
    for i, img in enumerate(images):
        img_name = os.path.basename(img['path'])
        gt_path = os.path.join(GROUND_TRUTH_DIR, f"{os.path.splitext(img_name)[0]}.json")
        
        if os.path.exists(gt_path) and not args.force:
            print(f"  [{i+1}/{len(images)}] ⏩ 跳过已有: {img_name}")
            stats['success'] += 1
            continue
        
        print(f"  [{i+1}/{len(images)}] 📷 标注: {img_name} ({img['category']})", end='')
        sys.stdout.flush()
        
        try:
            result = auto_label_image(img['path'], sp, up, llm_cfg, ctx)
            
            # 添加元数据
            result['_meta'] = {
                'source': img['source'],
                'category': img['category'],
                'image_name': img_name,
                'image_path': img['path'],
                'labeled_at': datetime.now().isoformat(),
                'labeler': 'V5.4_SP_Auto',
                'field_count': len([k for k in result.keys() if not k.startswith('_')]),
            }
            
            # 保存标注
            with open(gt_path, 'w', encoding='utf-8') as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
            
            if 'error' in result:
                stats['failed'] += 1
                stats['errors'].append(img_name)
                print(f" ❌ {result['error'][:50]}")
            else:
                stats['success'] += 1
                print(f" ✅ {result['_meta']['field_count']}字段")
        
        except Exception as e:
            stats['failed'] += 1
            stats['errors'].append(img_name)
            print(f" ❌ 异常: {str(e)[:60]}")
        
        # 每10张保存一次进度报告
        if (i + 1) % 10 == 0:
            elapsed = time.time() - start_time
            speed = (i + 1) / elapsed if elapsed > 0 else 0
            print(f"\n📊 进度: {i+1}/{len(images)} | {speed:.1f}张/秒 | {stats['success']}成功/{stats['failed']}失败")
    
    # 最终报告
    elapsed = time.time() - start_time
    report = {
        'timestamp': datetime.now().isoformat(),
        'total_images': stats['total'],
        'success': stats['success'],
        'failed': stats['failed'],
        'errors': stats['errors'],
        'elapsed_seconds': round(elapsed, 1),
        'speed_per_second': round(stats['total'] / elapsed, 1) if elapsed > 0 else 0,
        'config': {'sp_file': SP_FILE, 'sample': args.sample, 'category': args.category}
    }
    
    report_path = os.path.join(REPORT_DIR, f'auto_label_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json')
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    
    print(f"\n{'='*50}")
    print(f"✅ 自动标注完成")
    print(f"   总图: {stats['total']} | 成功: {stats['success']} | 失败: {stats['failed']}")
    print(f"   耗时: {elapsed:.1f}s | 速度: {report['speed_per_second']:.1f}张/秒")
    print(f"   报告: {report_path}")
    if stats['errors']:
        print(f"   失败列表: {stats['errors'][:5]}...")
    print(f"{'='*50}")


if __name__ == '__main__':
    main()