'''
@Author  :61022
@Time    :2026/8/8
@Desc    :
'''
from pymilvus import MilvusClient

from atguigu.config.config import KBImportConfig

milvus_client = None

def get_milvus_client():
    global milvus_client
    if milvus_client:
        return milvus_client

    milvus_client = MilvusClient(
        uri=KBImportConfig.MILVUS_URL
    )

    return milvus_client

if __name__ == '__main__':
    print(get_milvus_client())