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
except Exception as e:
    logger.error(f"MinIO初始化失败 {e}")

# Upload the file, renaming it in the process
# client.fput_object(
#     bucket_name, destination_file, source_file,
# )


# # Remove list of objects.
# errors = client.remove_objects(
#     "my-bucket",
#     [
#         DeleteObject("my-object1"),
#         DeleteObject("my-object2"),
#         DeleteObject("my-object3", "13f88b18-8dcd-4c83-88f2-8631fdb6250c"),
#     ],
# )
# for error in errors:
#     print("error occurred when deleting object", error)
#
# # Remove a prefix recursively.
# delete_object_list = map(
#     lambda x: DeleteObject(x.object_name),
#     client.list_objects("my-bucket", "my/prefix/", recursive=True),
# )
# errors = client.remove_objects("my-bucket", delete_object_list)
# for error in errors:
#     print("error occurred when deleting object", error)

def get_minio_client():
    return client


if __name__ == '__main__':
    get_minio_client()