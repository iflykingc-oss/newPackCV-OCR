# -*- coding: utf-8 -*-
"""
命令行工具
提供一键识别、批量处理、报表生成等功能
"""

import os
import sys
import json
import argparse
import logging
from pathlib import Path
from typing import List, Optional

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.core.ocr.ocr_scheduler import OCRScheduler
from src.core.cv.preprocessor import ImagePreprocessor
from src.core.cv.detector import ObjectDetector
from src.core.rule_engine.validator import ExpiryValidator
from src.core.rule_engine.alert import AlertManager
from src.storage.oss import upload_image
from utils.file.file import FileOps

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class PackCVCLI:
    """PackCV命令行工具"""

    def __init__(self):
        self.ocr_scheduler = OCRScheduler()
        self.preprocessor = ImagePreprocessor()
        self.detector = ObjectDetector()
        self.validator = ExpiryValidator()
        self.alert_manager = AlertManager()

    def recognize_single(
        self,
        image_path: str,
        enhance: bool = True,
        detect: bool = False,
        output: str = None
    ) -> dict:
        """
        识别单张图片

        Args:
            image_path: 图片路径（本地或URL）
            enhance: 是否预处理增强
            detect: 是否进行目标检测
            output: 输出文件路径

        Returns:
            识别结果
        """
        logger.info(f"识别图片: {image_path}")

        # 读取图片
        if image_path.startswith('http'):
            import requests
            response = requests.get(image_path)
            image_bytes = response.content
        else:
            with open(image_path, 'rb') as f:
                image_bytes = f.read()

        # 预处理
        if enhance:
            logger.info("预处理增强...")
            # 预处理逻辑

        # OCR识别
        logger.info("OCR识别...")
        ocr_result = self.ocr_scheduler.recognize(image_bytes)

        # 目标检测
        if detect:
            logger.info("目标检测...")
            # 检测逻辑

        # 校验
        logger.info("效期校验...")
        validation_result = self.validator.validate({
            "production_date": ocr_result.structured_data.get("production_date"),
            "shelf_life": ocr_result.structured_data.get("shelf_life")
        })

        result = {
            "image": image_path,
            "raw_text": ocr_result.raw_text,
            "corrected_text": ocr_result.corrected_text,
            "structured_data": ocr_result.structured_data,
            "confidence": ocr_result.overall_confidence,
            "validation": validation_result
        }

        # 输出
        if output:
            with open(output, 'w', encoding='utf-8') as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
            logger.info(f"结果已保存到: {output}")
        else:
            print(json.dumps(result, ensure_ascii=False, indent=2))

        return result

    def recognize_batch(
        self,
        image_dir: str,
        output_dir: str = "./output",
        pattern: str = "*.{jpg,jpeg,png}",
        parallel: int = 4
    ) -> List[dict]:
        """
        批量识别图片

        Args:
            image_dir: 图片目录
            output_dir: 输出目录
            pattern: 文件匹配模式
            parallel: 并行数

        Returns:
            识别结果列表
        """
        from concurrent.futures import ThreadPoolExecutor, as_completed

        logger.info(f"批量识别: {image_dir}")

        # 收集图片文件
        image_dir_path = Path(image_dir)
        image_files = list(image_dir_path.glob(pattern))
        logger.info(f"找到 {len(image_files)} 张图片")

        if not image_files:
            logger.warning(f"未找到匹配 {pattern} 的图片")
            return []

        # 确保输出目录存在
        os.makedirs(output_dir, exist_ok=True)

        results = []
        with ThreadPoolExecutor(max_workers=parallel) as executor:
            futures = {}
            for img_file in image_files:
                output_file = os.path.join(
                    output_dir,
                    f"{img_file.stem}_result.json"
                )
                future = executor.submit(
                    self.recognize_single,
                    str(img_file),
                    output=output_file
                )
                futures[future] = img_file

            for future in as_completed(futures):
                img_file = futures[future]
                try:
                    result = future.result()
                    results.append(result)
                    logger.info(f"✓ 完成: {img_file.name}")
                except Exception as e:
                    logger.error(f"✗ 失败: {img_file.name} - {e}")

        # 生成汇总报告
        summary_file = os.path.join(output_dir, "summary.json")
        summary = {
            "total": len(image_files),
            "success": len(results),
            "failed": len(image_files) - len(results),
            "results": results
        }
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)

        logger.info(f"批量识别完成: {len(results)}/{len(image_files)} 成功")
        logger.info(f"汇总报告: {summary_file}")

        return results

    def generate_report(
        self,
        data_dir: str,
        report_type: str = "expiry",
        output: str = "./report.xlsx"
    ) -> str:
        """
        生成报表

        Args:
            data_dir: 数据目录（包含识别结果JSON）
            report_type: 报表类型（expiry/stock/compliance）
            output: 输出文件路径

        Returns:
            报表文件路径
        """
        logger.info(f"生成报表: {report_type}")

        # 收集数据
        data_dir_path = Path(data_dir)
        result_files = list(data_dir_path.glob("*_result.json"))

        all_data = []
        for result_file in result_files:
            with open(result_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                all_data.append(data)

        # 生成报表
        if report_type == "expiry":
            from src.storage.db import ReportModel
            report = self._generate_expiry_report(all_data)
        elif report_type == "stock":
            report = self._generate_stock_report(all_data)
        else:
            report = self._generate_compliance_report(all_data)

        # 保存报表
        with open(output.replace('.xlsx', '.json'), 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)

        logger.info(f"报表已生成: {output}")
        return output

    def _generate_expiry_report(self, data: List[dict]) -> dict:
        """生成效期报表"""
        expiring_soon = []
        expired = []
        normal = []

        for item in data:
            structured = item.get("structured_data", {})
            validation = item.get("validation", {})

            product_info = {
                "name": structured.get("product_name", "未知"),
                "production_date": structured.get("production_date", ""),
                "shelf_life": structured.get("shelf_life", ""),
                "days_until_expiry": validation.get("days_until_expiry")
            }

            days = validation.get("days_until_expiry", 999)
            if days < 0:
                expired.append(product_info)
            elif days <= 30:
                expiring_soon.append(product_info)
            else:
                normal.append(product_info)

        return {
            "report_type": "expiry",
            "summary": {
                "total": len(data),
                "expired": len(expired),
                "expiring_soon": len(expiring_soon),
                "normal": len(normal)
            },
            "expired_products": expired,
            "expiring_soon_products": expiring_soon,
            "normal_products": normal
        }

    def _generate_stock_report(self, data: List[dict]) -> dict:
        """生成库存报表"""
        return {
            "report_type": "stock",
            "total_products": len(data)
        }

    def _generate_compliance_report(self, data: List[dict]) -> dict:
        """生成合规报表"""
        return {
            "report_type": "compliance",
            "total_products": len(data),
            "compliant": len([d for d in data if d.get("validation", {}).get("is_valid")]),
            "non_compliant": len([d for d in data if not d.get("validation", {}).get("is_valid")])
        }


def main():
    """命令行入口"""
    parser = argparse.ArgumentParser(
        description="PackCV - 包装/货架OCR识别工具",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    subparsers = parser.add_subparsers(dest="command", help="子命令")

    # 单图识别
    recognize_parser = subparsers.add_parser(
        "recognize",
        help="识别单张图片"
    )
    recognize_parser.add_argument("image", help="图片路径或URL")
    recognize_parser.add_argument("--no-enhance", action="store_true", help="禁用预处理增强")
    recognize_parser.add_argument("--detect", action="store_true", help="启用目标检测")
    recognize_parser.add_argument("-o", "--output", help="输出文件路径")

    # 批量识别
    batch_parser = subparsers.add_parser(
        "batch",
        help="批量识别图片"
    )
    batch_parser.add_argument("directory", help="图片目录")
    batch_parser.add_argument("-o", "--output-dir", default="./output", help="输出目录")
    batch_parser.add_argument("-p", "--parallel", type=int, default=4, help="并行数")

    # 生成报表
    report_parser = subparsers.add_parser(
        "report",
        help="生成报表"
    )
    report_parser.add_argument("data_dir", help="数据目录")
    report_parser.add_argument("-t", "--type", default="expiry", choices=["expiry", "stock", "compliance"])
    report_parser.add_argument("-o", "--output", default="./report.xlsx")

    args = parser.parse_args()

    cli = PackCVCLI()

    if args.command == "recognize":
        cli.recognize_single(
            args.image,
            enhance=not args.no_enhance,
            detect=args.detect,
            output=args.output
        )
    elif args.command == "batch":
        cli.recognize_batch(
            args.directory,
            output_dir=args.output_dir,
            parallel=args.parallel
        )
    elif args.command == "report":
        cli.generate_report(
            args.data_dir,
            report_type=args.type,
            output=args.output
        )
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
