"""
PackCV-OCR V5.0 批量评测 - 简化版
使用现有真实图片 + Python退化生成，无需AI生成
"""
import os, sys, json, time, hashlib
from typing import Dict
from PIL import Image, ImageFilter, ImageEnhance, ImageDraw
import numpy as np

sys.path.insert(0, os.path.join(os.getenv("COZE_WORKSPACE_PATH", "/workspace/projects"), "src"))
from graphs.graph import main_graph
from coze_coding_dev_sdk.s3 import S3SyncStorage

ASSETS_DIR = os.path.join(os.getenv("COZE_WORKSPACE_PATH", "/workspace/projects"), "assets")
MOCK_DIR = os.path.join(ASSETS_DIR, "mock")
os.makedirs(MOCK_DIR, exist_ok=True)

STORAGE = S3SyncStorage(
    endpoint_url=os.getenv("COZE_BUCKET_ENDPOINT_URL"),
    access_key="", secret_key="",
    bucket_name=os.getenv("COZE_BUCKET_NAME"),
    region="cn-beijing",
)

RESAMPLE = Image.Resampling.LANCZOS if hasattr(Image.Resampling, 'LANCZOS') else Image.BICUBIC

def upload(img: Image.Image, name: str) -> str:
    local = os.path.join(MOCK_DIR, name)
    img.save(local, quality=92)
    with open(local, "rb") as f:
        key = STORAGE.upload_file(file_content=f.read(), file_name=name, content_type="image/jpeg")
    return STORAGE.generate_presigned_url(key=key, expire_time=86400)

def run_test(url: str, case: str) -> Dict:
    t0 = time.time()
    try:
        r = main_graph.invoke({
            "package_image": {"url": url, "file_type": "image"},
            "platform": "none",
            "user_question": "请提取这张包装图片中的所有产品信息"
        }, {"recursion_limit": 50})
        t = time.time() - t0
        data = r.get("structured_data", {})
        fcnt = sum(1 for v in data.values() if v and v != "N/A" and v != [])
        ncnt = sum(1 for v in data.values() if not v or v == "N/A")
        tcnt = len(data)
        return {"case": case, "status": "ok", "t": round(t,2), "fcnt": fcnt, "ncnt": ncnt, "tcnt": tcnt,
                "rate": round(fcnt/max(tcnt,1)*100,1), "raw_len": len(r.get("raw_text","")),
                "error": ""}
    except Exception as e:
        t = time.time() - t0
        return {"case": case, "status": "err", "t": round(t,2), "fcnt": 0, "ncnt": 0, "tcnt": 0,
                "rate": 0.0, "raw_len": 0, "error": str(e)[:200]}

# 退化函数
QUALITY_DEGRADATIONS = {
    "原始_清晰": lambda img: img,
    "轻度模糊_r=2": lambda img: img.filter(ImageFilter.GaussianBlur(2)),
    "中度模糊_r=4": lambda img: img.filter(ImageFilter.GaussianBlur(4)),
    "重度模糊_r=6": lambda img: img.filter(ImageFilter.GaussianBlur(6)),
    "暗光_x0.6": lambda img: ImageEnhance.Brightness(img).enhance(0.6),
    "暗光_x0.3": lambda img: ImageEnhance.Brightness(img).enhance(0.3),
    "过曝_x1.6": lambda img: ImageEnhance.Brightness(img).enhance(1.6),
    "轻度噪点_2%": lambda img: _noise(img, 0.02),
    "重度噪点_5%": lambda img: _noise(img, 0.05),
    "倾斜10度": lambda img: img.rotate(10, expand=True, fillcolor=(255,255,255)),
    "倾斜25度": lambda img: img.rotate(25, expand=True, fillcolor=(255,255,255)),
    "透视20%": lambda img: _perspective(img, 0.2),
    "遮挡20%": lambda img: _occlude(img, 0.2),
    "遮挡40%": lambda img: _occlude(img, 0.4),
    "小字0.5x": lambda img: _down_up(img, 0.5),
    "小字0.3x": lambda img: _down_up(img, 0.3),
    "暗光+模糊": lambda img: ImageEnhance.Brightness(img.filter(ImageFilter.GaussianBlur(3))).enhance(0.5),
    "模糊+倾斜": lambda img: img.filter(ImageFilter.GaussianBlur(3)).rotate(15, expand=True, fillcolor=(255,255,255)),
    "全退化": lambda img: _noise(ImageEnhance.Brightness(img.filter(ImageFilter.GaussianBlur(4))).enhance(0.5), 0.03).rotate(10, expand=True, fillcolor=(255,255,255)),
}

def _noise(img, intensity):
    arr = np.array(img)
    mask = np.random.random(arr.shape[:2]) < intensity
    for c in range(min(3, arr.shape[2] if len(arr.shape)==3 else 1)):
        channel = arr[:,:,c] if len(arr.shape)==3 else arr
        channel[mask & (np.random.random(mask.shape)<0.5)] = 255
        channel[mask & ~(np.random.random(mask.shape)<0.5)] = 0
    return Image.fromarray(arr.astype('uint8'))

def _perspective(img, skew):
    w,h=img.size; off=int(w*skew)
    return img.transform(img.size, Image.PERSPECTIVE, 
        [1,0,-off/2,0,1,0,off/w,0,1], Image.Resampling.BICUBIC)

def _occlude(img, cov):
    draw=ImageDraw.Draw(img); w,h=img.size
    for _ in range(max(1,int(cov*5))):
        bw,bh=w//8,h//8
        x0=np.random.randint(0,max(1,w-bw))
        y0=np.random.randint(0,max(1,h-bh))
        x1=np.random.randint(x0+1,min(w,x0+bw*3))
        y1=np.random.randint(y0+1,min(h,y0+bh*3))
        draw.rectangle([x0,y0,x1,y1],fill=(200,200,200))
    return img

def _down_up(img, scale):
    w,h=img.size
    s=img.resize((int(w*scale),int(h*scale)),RESAMPLE)
    return s.resize((w,h),RESAMPLE)

def main():
    results = []
    
    # 真实图片：紫薯雪饼 + 答菲湿巾
    real_images = {
        "紫薯雪饼": os.path.join(ASSETS_DIR, "1111_20260618183731267.jpg"),
        "答菲湿巾": os.path.join(ASSETS_DIR, "测试识别图.jpg"),
    }
    
    total_expected = len(real_images) * len(QUALITY_DEGRADATIONS)
    print(f"开始评测: {len(real_images)}张真实图 × {len(QUALITY_DEGRADATIONS)}种退化 = {total_expected}个测试用例")
    
    idx = 0
    for real_name, real_path in real_images.items():
        if not os.path.exists(real_path):
            print(f"[SKIP] {real_name}: 文件不存在")
            continue
        base = Image.open(real_path).convert("RGB")
        
        for deg_name, deg_func in QUALITY_DEGRADATIONS.items():
            idx += 1
            try:
                variant = deg_func(base.copy())
                fname = f"eval_{hashlib.md5(f'{real_name}_{deg_name}'.encode()).hexdigest()[:8]}.jpg"
                url = upload(variant, fname)
                r = run_test(url, f"{real_name}-{deg_name}")
                results.append(r)
                bar = "█" * int(r["rate"]/5) + "░" * (20-int(r["rate"]/5))
                print(f"[{idx:3d}/{total_expected}] {real_name:8s} | {deg_name:14s} | {bar} {r['rate']:5.1f}% | {r['t']:.1f}s | {r['error'] or 'OK'}")
            except Exception as e:
                print(f"[{idx:3d}/{total_expected}] {real_name:8s} | {deg_name:14s} | {'❌ ERROR':>20s} | {str(e)[:60]}")
    
    # ===== 汇总 =====
    print("\n" + "=" * 65)
    print("📊 PackCV-OCR V5.0 批量评测报告")
    print("=" * 65)
    
    success = [r for r in results if r["status"] == "ok"]
    errors = [r for r in results if r["status"] == "err"]
    
    rates = [r["rate"] for r in success]
    times = [r["t"] for r in success]
    
    print(f"\n📈 总测试: {len(results)} | 成功: {len(success)} | 失败: {len(errors)}")
    print(f"📈 平均提取率: {np.mean(rates):.1f}%")
    print(f"📈 中位提取率: {np.median(rates):.1f}%")
    print(f"📈 平均处理时间: {np.mean(times):.2f}s")
    
    # 按场景分组
    scenarios = {}
    for r in results:
        seg = "模糊" if "模糊" in r["case"] else \
              "暗光" if "暗光" in r["case"] else \
              "过曝" if "过曝" in r["case"] else \
              "噪点" if "噪点" in r["case"] else \
              "倾斜" if "倾斜" in r["case"] else \
              "透视" if "透视" in r["case"] else \
              "遮挡" if "遮挡" in r["case"] else \
              "小字" if "小字" in r["case"] else \
              "组合" if "+" in r["case"] or "全" in r["case"] else \
              "原始清晰" if "原始" in r["case"] else "其他"
        scenarios.setdefault(seg, []).append(r)
    
    print("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print("各场景分类评测")
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    for sc in ["原始清晰", "模糊", "暗光", "过曝", "噪点", "倾斜", "透视", "遮挡", "小字", "组合"]:
        sg = scenarios.get(sc, [])
        if not sg: continue
        sr = [r for r in sg if r["status"]=="ok"]
        srates = [r["rate"] for r in sr]
        stimes = [r["t"] for r in sr]
        print(f"\n{sc:10s} | {len(sg):2d}例 | 提取率: {np.mean(srates):.1f}% | 最快: {min(stimes):.1f}s | 最慢: {max(stimes):.1f}s")
        if sr:
            best = max(sr, key=lambda x:x["rate"])
            worst = min(sr, key=lambda x:x["rate"])
            print(f"         最佳: {best['case'][:25]} ({best['rate']}%)")
            print(f"         最差: {worst['case'][:25]} ({worst['rate']}%)")
    
    # 保存
    report = {
        "total": len(results), "success": len(success), "error": len(errors),
        "avg_rate": round(float(np.mean(rates)), 1),
        "median_rate": round(float(np.median(rates)), 1),
        "avg_time_s": round(float(np.mean(times)), 2),
        "scenarios": {k: {
            "count": len(v), "avg_rate": round(float(np.mean([r["rate"] for r in v if r["status"]=="ok"])), 1)
        } for k,v in scenarios.items()},
        "results": results,
    }
    rpath = os.path.join(MOCK_DIR, "eval_report.json")
    with open(rpath, "w") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"\n✅ 报告已保存: {rpath}")

if __name__ == "__main__":
    main()