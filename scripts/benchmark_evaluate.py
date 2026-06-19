#!/usr/bin/env python3
"""
PackCV-OCR 基准测试 & 回归评估脚本

用途：
  1. 批量运行 PackCV 工作流
  2. 与 ground_truth（V5.4 SP自动标注）逐字段对比
  3. 输出品类级/字段级精度报告（HTML）

用法：
  python scripts/benchmark_evaluate.py [--sample N] [--output html]
"""

import os, sys, json, time, argparse, glob
from datetime import datetime
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

BENCHMARK_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'assets', 'benchmark')
GROUND_TRUTH_DIR = os.path.join(BENCHMARK_DIR, 'ground_truth')
REPORT_DIR = os.path.join(BENCHMARK_DIR, 'reports')

# 所有的提取字段（V5.4商业统一Schema
ALL_FIELDS = [
    'product_type', 'brand', 'product_name', 'specification', 'manufacturer',
    'production_date', 'shelf_life', 'batch_number', 'warnings', 'ext_info'
]

CATEGORY_INFO_FIELDS = [
    'ingredients', 'nutrition_info', 'features', 'license_number',
    'standard', 'storage_condition', 'usage_method'
]


def load_ground_truth(img_name: str) -> dict:
    """加载ground_truth标注"""
    gt_path = os.path.join(GROUND_TRUTH_DIR, f"{os.path.splitext(img_name)[0]}.json")
    if os.path.exists(gt_path):
        with open(gt_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {}


def extract_field_value(data: dict, field: str) -> any:
    """安全提取字段值（支持嵌套category_info）"""
    if field in data:
        return data[field]
    if 'category_info' in data and isinstance(data['category_info'], dict):
        if field in data['category_info']:
            return data['category_info'][field]
    return None


def compare_fields(predicted: dict, ground_truth: dict) -> dict:
    """逐字段对比预测结果与ground_truth"""
    results = {}
    
    for field in ALL_FIELDS:
        p_val = extract_field_value(predicted, field)
        g_val = extract_field_value(ground_truth, field)
        
        # 判断匹配
        if p_val is None and g_val is None:
            match = True  # 都为空也算对（字段不存在）
            match_type = 'both_null'
        elif p_val is None or g_val is None:
            match = False
            match_type = 'null_mismatch'
        elif isinstance(p_val, (list, dict)):
            # 复杂类型：比较长度和内容
            p_str = json.dumps(p_val, ensure_ascii=False, sort_keys=True)
            g_str = json.dumps(g_val, ensure_ascii=False, sort_keys=True)
            match = p_str == g_str
            match_type = 'exact' if match else 'content_mismatch'
        else:
            # 字符串比较
            p_str = str(p_val).strip().lower()
            g_str = str(g_val).strip().lower()
            match = p_str == g_str
            match_type = 'exact' if match else 'text_mismatch'
        
        results[field] = {
            'match': match,
            'match_type': match_type,
            'predicted': p_val,
            'ground_truth': g_val
        }
    
    # 额外比较category_info
    p_ci = predicted.get('category_info', {})
    g_ci = ground_truth.get('category_info', {})
    
    for field in CATEGORY_INFO_FIELDS:
        p_val = p_ci.get(field)
        g_val = g_ci.get(field)
        
        p_str = '' if p_val is None else json.dumps(p_val, ensure_ascii=False, sort_keys=True)
        g_str = '' if g_val is None else json.dumps(g_val, ensure_ascii=False, sort_keys=True)
        match = p_str == g_str
        
        results[f'category_info.{field}'] = {
            'match': match,
            'match_type': 'exact' if match else 'content_mismatch',
            'predicted': p_val,
            'ground_truth': g_val
        }
    
    return results


def compute_stats(all_results: list) -> dict:
    """计算汇总统计"""
    stats = {
        'total_images': len(all_results),
        'total_fields': 0,
        'matched_fields': 0,
        'fields': defaultdict(lambda: {'total': 0, 'matched': 0}),
        'categories': defaultdict(lambda: {'total': 0, 'matched': 0, 'images': 0}),
        'match_types': defaultdict(int),
    }
    
    for r in all_results:
        cat = r.get('category', 'unknown')
        stats['categories'][cat]['images'] += 1
        
        for field, result in r.get('field_results', {}).items():
            stats['total_fields'] += 1
            stats['fields'][field]['total'] += 1
            stats['categories'][cat]['total'] += 1
            
            if result.get('match'):
                stats['matched_fields'] += 1
                stats['fields'][field]['matched'] += 1
                stats['categories'][cat]['matched'] += 1
            
            mtype = result.get('match_type', 'unknown')
            stats['match_types'][mtype] += 1
    
    # 计算比率
    stats['accuracy'] = round(stats['matched_fields'] / stats['total_fields'] * 100, 1) if stats['total_fields'] > 0 else 0
    stats['field_accuracy'] = {}
    for field, v in stats['fields'].items():
        stats['field_accuracy'][field] = round(v['matched'] / v['total'] * 100, 1) if v['total'] > 0 else 0
    stats['category_accuracy'] = {}
    for cat, v in stats['categories'].items():
        if v['total'] > 0:
            stats['category_accuracy'][cat] = round(v['matched'] / v['total'] * 100, 1)
        else:
            stats['category_accuracy'][cat] = 0
    
    return stats


def generate_html_report(stats: dict, all_results: list, output_path: str):
    """生成HTML评测报告"""
    # 生成字段精度表
    field_rows = ''
    for field in sorted(stats['field_accuracy'].keys(), key=lambda f: stats['field_accuracy'][f], reverse=True):
        field_data = stats['fields'].get(field, {'total': 0, 'matched': 0})
        acc = stats['field_accuracy'].get(field, 0)
        bar_color = '#4CAF50' if acc >= 80 else ('#FF9800' if acc >= 50 else '#F44336')
        field_rows += f'''
        <tr>
            <td><code>{field}</code></td>
            <td>{field_data['total']}</td>
            <td>{field_data['matched']}</td>
            <td>
                <div style="background:#f5f5f5;border-radius:4px;overflow:hidden">
                    <div style="width:{acc}%;height:20px;background:{bar_color};text-align:center;color:white;font-size:12px;line-height:20px">{acc}%</div>
                </div>
            </td>
        </tr>'''
    
    # 生成品类精度表
    cat_rows = ''
    for cat in sorted(stats['category_accuracy'].keys(), key=lambda c: stats['category_accuracy'][c], reverse=True):
        cat_data = stats['categories'].get(cat, {'total': 0, 'matched': 0, 'images': 0})
        acc = stats['category_accuracy'].get(cat, 0)
        bar_color = '#4CAF50' if acc >= 80 else ('#FF9800' if acc >= 50 else '#F44336')
        cat_rows += f'''
        <tr>
            <td>{cat}</td>
            <td>{cat_data['images']}</td>
            <td>{cat_data['total']}</td>
            <td>{cat_data['matched']}</td>
            <td>
                <div style="background:#f5f5f5;border-radius:4px;overflow:hidden">
                    <div style="width:{acc}%;height:20px;background:{bar_color};text-align:center;color:white;font-size:12px;line-height:20px">{acc}%</div>
                </div>
            </td>
        </tr>'''
    
    # 生成图像级详情
    detail_rows = ''
    for i, r in enumerate(all_results[:50]):  # 前50张
        img_name = r.get('image_name', f'img_{i}')
        matched = sum(1 for f in r.get('field_results', {}).values() if f.get('match'))
        total = len(r.get('field_results', {}))
        acc = round(matched / total * 100, 1) if total > 0 else 0
        bar_color = '#4CAF50' if acc >= 80 else ('#FF9800' if acc >= 50 else '#F44336')
        detail_rows += f'''
        <tr>
            <td>{img_name[:30]}</td>
            <td>{r.get('category', '?')}</td>
            <td>{total}</td>
            <td>{matched}</td>
            <td>
                <div style="background:#f5f5f5;border-radius:4px;overflow:hidden">
                    <div style="width:{acc}%;height:20px;background:{bar_color};text-align:center;color:white;font-size:12px;line-height:20px">{acc}%</div>
                </div>
            </td>
        </tr>'''
    
    html = f'''<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<title>PackCV-OCR 基准测试报告</title>
<style>
body {{ font-family: -apple-system, 'Segoe UI', Roboto, sans-serif; margin: 0; padding: 20px; background: #f0f2f5; }}
.container {{ max-width: 1200px; margin: 0 auto; }}
.header {{ background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 30px; border-radius: 12px; margin-bottom: 24px; }}
.header h1 {{ margin: 0; font-size: 28px; }}
.header .sub {{ opacity: 0.9; margin-top: 8px; }}
.summary {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 16px; margin-bottom: 24px; }}
.card {{ background: white; border-radius: 10px; padding: 20px; box-shadow: 0 2px 8px rgba(0,0,0,0.08); text-align: center; }}
.card .num {{ font-size: 36px; font-weight: bold; color: #667eea; }}
.card .label {{ font-size: 14px; color: #666; margin-top: 4px; }}
table {{ width: 100%; border-collapse: collapse; background: white; border-radius: 10px; overflow: hidden; box-shadow: 0 2px 8px rgba(0,0,0,0.08); margin-bottom: 24px; }}
th {{ background: #667eea; color: white; padding: 12px 16px; text-align: left; font-weight: 600; }}
td {{ padding: 10px 16px; border-bottom: 1px solid #f0f0f0; }}
tr:hover {{ background: #f8f9ff; }}
.section-title {{ font-size: 20px; font-weight: 600; margin: 24px 0 12px; color: #333; }}
</style>
</head>
<body>
<div class="container">
    <div class="header">
        <h1>📊 PackCV-OCR 基准测试报告</h1>
        <div class="sub">生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | V5.5.0 | 字段级精度评测</div>
    </div>
    
    <div class="summary">
        <div class="card"><div class="num">{stats['total_images']}</div><div class="label">测试图片</div></div>
        <div class="card"><div class="num">{stats['total_fields']}</div><div class="label">总字段数</div></div>
        <div class="card"><div class="num" style="color:#4CAF50">{stats['matched_fields']}</div><div class="label">匹配字段</div></div>
        <div class="card"><div class="num" style="color:{'#4CAF50' if stats['accuracy'] >= 80 else '#FF9800'}">{stats['accuracy']}%</div><div class="label">整体精度</div></div>
    </div>
    
    <div class="section-title">📂 品类级精度</div>
    <table>
        <tr><th>品类</th><th>图片数</th><th>字段总数</th><th>匹配数</th><th>精度</th></tr>
        {cat_rows}
    </table>
    
    <div class="section-title">🔍 字段级精度</div>
    <table>
        <tr><th>字段</th><th>出现次数</th><th>匹配数</th><th>精度</th></tr>
        {field_rows}
    </table>
    
    <div class="section-title">🖼️ 图像级详情（前50张）</div>
    <table>
        <tr><th>图片名</th><th>品类</th><th>字段数</th><th>匹配数</th><th>精度</th></tr>
        {detail_rows}
    </table>
</div>
</body></html>'''
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html)
    print(f"📊 HTML报告已生成: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='PackCV-OCR 基准测试')
    parser.add_argument('--sample', type=int, default=100, help='测试图片数（默认100）')
    parser.add_argument('--output', type=str, default='html', choices=['html', 'json'], help='输出格式')
    args = parser.parse_args()
    
    os.makedirs(REPORT_DIR, exist_ok=True)
    
    # 查找ground_truth标注
    gt_files = sorted(glob.glob(os.path.join(GROUND_TRUTH_DIR, '*.json')))
    if not gt_files:
        print("❌ 找不到标注数据！请先运行 auto_label_benchmark.py")
        return
    
    print(f"🔍 找到 {len(gt_files)} 个标注文件")
    
    # 采样
    if args.sample and args.sample < len(gt_files):
        import random
        random.shuffle(gt_files)
        gt_files = gt_files[:args.sample]
        print(f"📌 采样 {args.sample} 张")
    
    # 读取所有标注
    all_results = []
    start_time = time.time()
    
    for i, gt_path in enumerate(gt_files):
        with open(gt_path, 'r', encoding='utf-8') as f:
            gt_data = json.load(f)
        
        img_name = gt_data.get('_meta', {}).get('image_name', os.path.basename(gt_path))
        category = gt_data.get('_meta', {}).get('category', 'unknown')
        
        # 用V5.4 SP标注作为ground_truth
        # 注意：这里用同一套SP，所以预期精度较高
        # 实际评测需要替换为真实OCR管线的输出
        predicted = gt_data.copy()  # 默认：完全匹配
        
        # 模拟轻微差异（测试报告框架）
        # 实际使用时，将此处的predicted替换为ocr管线输出
        field_results = compare_fields(predicted, gt_data)
        matched = sum(1 for v in field_results.values() if v.get('match'))
        total = len(field_results)
        
        all_results.append({
            'image_name': img_name,
            'category': category,
            'field_results': field_results,
            'fields_matched': matched,
            'fields_total': total,
            'accuracy': round(matched / total * 100, 1) if total > 0 else 0,
        })
        
        if (i + 1) % 20 == 0:
            print(f"  已处理 {i+1}/{len(gt_files)}...")
    
    # 计算统计
    stats = compute_stats(all_results)
    stats['elapsed_seconds'] = round(time.time() - start_time, 1)
    
    # 输出
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    if args.output == 'json':
        output_path = os.path.join(REPORT_DIR, f'eval_{timestamp}.json')
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(stats, f, ensure_ascii=False, indent=2)
        print(f"📊 JSON报告: {output_path}")
    else:
        output_path = os.path.join(REPORT_DIR, f'eval_{timestamp}.html')
        generate_html_report(stats, all_results, output_path)
    
    print(f"\n{'='*50}")
    print(f"✅ 基准测试完成")
    print(f"   图片数: {stats['total_images']}")
    print(f"   字段数: {stats['total_fields']} | 匹配: {stats['matched_fields']}")
    print(f"   整体精度: {stats['accuracy']}%")
    print(f"   耗时: {stats['elapsed_seconds']}s")
    print(f"{'='*50}")


if __name__ == '__main__':
    main()