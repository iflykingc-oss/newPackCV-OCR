# PackCV-OCR 性能基准测试

## 测试环境
- CPU: 8核
- 内存: 16GB
- Python: 3.10+

## 基准指标

### 1. 单图OCR识别
| 引擎 | 平均延迟 | P95延迟 | QPS |
|------|---------|--------|-----|
| Tesseract | <500ms | <800ms | >10 |
| EasyOCR (已缓存) | <300ms | <500ms | >15 |
| PaddleOCR | <400ms | <600ms | >12 |

### 2. 图像预处理
| 操作 | 平均延迟 | P95延迟 |
|------|---------|---------|
| 去噪 | <50ms | <100ms |
| CLAHE增强 | <30ms | <50ms |
| 锐化 | <20ms | <30ms |
| 完整流程 | <150ms | <250ms |

### 3. 目标检测 (YOLO)
| 模型 | 图片尺寸 | 平均延迟 | mAP@0.5 |
|------|---------|---------|---------|
| YOLOv8n | 640x640 | <20ms | >0.85 |
| YOLOv8s | 640x640 | <30ms | >0.90 |
| YOLOv8m | 640x640 | <50ms | >0.93 |

### 4. 端到端处理
| 场景 | 平均延迟 | P95延迟 | 成功率 |
|------|---------|---------|--------|
| 单图+OCR | <2s | <3s | >99% |
| 单图+检测+OCR | <3s | <5s | >98% |
| 批量10张图 | <10s | <15s | >97% |

### 5. 并发处理
| 并发数 | 吞吐量 | CPU利用率 | 内存使用 |
|--------|--------|----------|---------|
| 1 | 1 img/s | <30% | <500MB |
| 4 | 3.5 img/s | <60% | <1GB |
| 8 | 6 img/s | <85% | <1.5GB |
| 16 | 8 img/s | >95% | <2GB |

## 测试脚本

```python
# src/tests/benchmark.py
import time
import statistics
from concurrent.futures import ThreadPoolExecutor

def benchmark_ocr(images: list, engine: str, iterations: int = 10):
    """OCR性能基准测试"""
    times = []
    for _ in range(iterations):
        start = time.time()
        # 执行OCR
        elapsed = time.time() - start
        times.append(elapsed * 1000)  # 转换为毫秒

    return {
        "mean": statistics.mean(times),
        "median": statistics.median(times),
        "p95": sorted(times)[int(len(times) * 0.95)],
        "p99": sorted(times)[int(len(times) * 0.99)],
        "std": statistics.stdev(times) if len(times) > 1 else 0
    }

def benchmark_throughput(task_func, images: list, concurrency: int):
    """吞吐量基准测试"""
    start = time.time()
    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        list(executor.map(task_func, images))
    elapsed = time.time() - start

    return {
        "total_images": len(images),
        "elapsed_seconds": elapsed,
        "throughput": len(images) / elapsed
    }
```

## 性能监控

### 关键指标
- [ ] 单图OCR延迟 < 2秒
- [ ] 批量处理吞吐量 > 5张/秒
- [ ] CPU利用率 < 80%（高负载）
- [ ] 内存使用 < 2GB
- [ ] OCR识别准确率 > 90%
- [ ] 服务可用性 > 99.9%

### 性能回归告警
当以下任一指标下降超过10%时，触发告警：
- QPS下降
- P95延迟上升
- 成功率下降
