'''
@Author  :61022
@Time    :2026/8/3
@Desc    :
'''
import json

from minio import Minio

from atguigu.config.config import KBImportConfig
from atguigu.tool.logger import logger

# 定义MinIO的客户端和权限
client = None

def get_minio_client():
    global client
    if not client:
        try:
            client = Minio(
                endpoint=KBImportConfig.MINIO_ENDPOINT,
                access_key=KBImportConfig.MINIO_ACCESS_KEY,
                secret_key=KBImportConfig.MINIO_SECRET_KEY,
                secure=False
            )
            # bucket不存在就创建
            if not client.bucket_exists(KBImportConfig.MINIO_BUCKET_NAME):
                client.make_bucket(KBImportConfig.MINIO_BUCKET_NAME)

                # 读取minio 免ak sk
                policy = {
                    "Version": "2012-10-17",
                    "Statement": [
                        {
                            "Effect": "Allow",
                            "Principal": {"AWS": "*"},
                            "Action": "s3:GetObject",
                            "Resource": f"arn:aws:s3:::{KBImportConfig.MINIO_BUCKET_NAME}/*",
                        },
                    ],
                }

                # 设置bucket的权限策略
                client.set_bucket_policy(KBImportConfig.MINIO_BUCKET_NAME, json.dumps(policy))
                logger.warning("桶不存在, 创建桶成功")
        except Exception as e:
            logger.error("MinIO初始化失败")
            raise e
    return client


if __name__ == '__main__':
    client=get_minio_client()
    client=get_minio_client()