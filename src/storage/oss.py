# -*- coding: utf-8 -*-
"""
对象存储封装
支持S3兼容存储（MinIO/阿里云OSS/火山引擎等）
"""

import os
import json
import logging
from typing import Optional, List, Dict, Any
from urllib.parse import urlparse

logger = logging.getLogger(__name__)


class OSSStorage:
    """S3兼容对象存储封装"""

    def __init__(
        self,
        endpoint: str = None,
        access_key: str = None,
        secret_key: str = None,
        bucket_name: str = None,
        region: str = "cn-beijing"
    ):
        """
        初始化OSS存储

        Args:
            endpoint: OSS endpoint地址
            access_key: 访问密钥ID
            secret_key: 访问密钥Secret
            bucket_name: 存储桶名称
            region: 区域
        """
        self.endpoint = endpoint or os.getenv("OSS_ENDPOINT", "")
        self.access_key = access_key or os.getenv("OSS_ACCESS_KEY", "")
        self.secret_key = secret_key or os.getenv("OSS_SECRET_KEY", "")
        self.bucket_name = bucket_name or os.getenv("OSS_BUCKET", "")
        self.region = region

        # 优先使用项目封装的S3SyncStorage
        self._client = None
        self._init_client()

    def _init_client(self):
        """初始化S3客户端"""
        try:
            from src.tools.s3_tool import S3SyncStorage
            self._client = S3SyncStorage(
                endpoint=self.endpoint,
                access_key=self.access_key,
                secret_key=self.secret_key,
                bucket_name=self.bucket_name
            )
            logger.info(f"OSS存储初始化成功: {self.bucket_name}")
        except ImportError:
            logger.warning("S3SyncStorage未找到，使用boto3兼容模式")
            try:
                import boto3
                self._s3 = boto3.client(
                    's3',
                    endpoint_url=self.endpoint,
                    aws_access_key_id=self.access_key,
                    aws_secret_access_key=self.secret_key,
                    region_name=self.region
                )
            except ImportError:
                logger.error("请安装boto3: pip install boto3")
                self._s3 = None

    def upload_file(
        self,
        file_content: bytes,
        file_name: str,
        content_type: str = "application/octet-stream"
    ) -> str:
        """
        上传文件到OSS

        Args:
            file_content: 文件内容
            file_name: 文件名
            content_type: MIME类型

        Returns:
            OSS上的文件路径/URL
        """
        if self._client:
            return self._client.upload_file(file_content, file_name, content_type)
        elif self._s3:
            self._s3.put_object(
                Bucket=self.bucket_name,
                Key=file_name,
                Body=file_content,
                ContentType=content_type
            )
            return f"{self.endpoint}/{self.bucket_name}/{file_name}"
        else:
            raise RuntimeError("OSS客户端未初始化")

    def download_file(self, key: str) -> bytes:
        """下载文件"""
        if self._client:
            return self._client.download_file(key)
        elif self._s3:
            response = self._s3.get_object(Bucket=self.bucket_name, Key=key)
            return response['Body'].read()
        else:
            raise RuntimeError("OSS客户端未初始化")

    def generate_presigned_url(
        self,
        key: str,
        expire_time: int = 3600
    ) -> str:
        """
        生成预签名URL

        Args:
            key: 文件路径
            expire_time: 过期时间（秒）

        Returns:
            预签名访问URL
        """
        if self._client:
            return self._client.generate_presigned_url(key, expire_time)
        elif self._s3:
            return self._s3.generate_presigned_url(
                'get_object',
                Params={'Bucket': self.bucket_name, 'Key': key},
                ExpiresIn=expire_time
            )
        else:
            raise RuntimeError("OSS客户端未初始化")

    def delete_file(self, key: str):
        """删除文件"""
        if self._client:
            return self._client.delete_file(key)
        elif self._s3:
            self._s3.delete_object(Bucket=self.bucket_name, Key=key)

    def list_files(self, prefix: str = "") -> List[str]:
        """列出文件"""
        if self._s3:
            response = self._s3.list_objects_v2(
                Bucket=self.bucket_name,
                Prefix=prefix
            )
            return [obj['Key'] for obj in response.get('Contents', [])]
        return []


# 全局单例
_oss_storage: Optional[OSSStorage] = None


def get_oss_storage() -> OSSStorage:
    """获取OSS存储单例"""
    global _oss_storage
    if _oss_storage is None:
        _oss_storage = OSSStorage()
    return _oss_storage


def upload_image(image_content: bytes, filename: str) -> str:
    """
    上传图片的便捷函数

    Args:
        image_content: 图片二进制内容
        filename: 文件名

    Returns:
        预签名访问URL
    """
    storage = get_oss_storage()
    # 根据扩展名确定content-type
    ext = filename.split('.')[-1].lower()
    content_type_map = {
        'jpg': 'image/jpeg',
        'jpeg': 'image/jpeg',
        'png': 'image/png',
        'gif': 'image/gif',
        'webp': 'image/webp'
    }
    content_type = content_type_map.get(ext, 'image/jpeg')

    key = storage.upload_file(image_content, filename, content_type)
    return storage.generate_presigned_url(key)
