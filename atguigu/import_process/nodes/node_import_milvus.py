'''
@Author  :61022
@Time    :2026/7/31
@Desc    :
'''
import json
import re
from typing import Tuple, List, Dict

from pymilvus import DataType

from atguigu.config.config import KBImportConfig
from atguigu.import_process.base import NodeBase
from atguigu.import_process.state import ImportGraphState
from atguigu.tool.logger import logger
from atguigu.tool.milvus_utils import get_milvus_client


class NodeImportMilvus(NodeBase):
    """
    导入向量库节点：数据持久化
    """

    @property
    def name(self) -> str:
        return "node_import_milvus"

    def process(self, state: ImportGraphState):
        # 初始化参数
        chunks, file_content_sha256, file_title = self.init_params(state)

        # 创建milvus表
        self.create_milvus_collection()

        # 按file_content_sha256幂等删除milvus表
        self.delete_milvus_data(file_content_sha256)

        # 写入milvus表
        self.insert_milvus_data(chunks, file_content_sha256, file_title)

        return {"chunks":chunks}

    def init_params(self, state: ImportGraphState) -> Tuple[List[Dict], str, str]:
        chunks = state.get("chunks")
        if not chunks:
            logger.error("chunks 不能为空")
            raise ValueError("chunks 不能为空")

        file_content_sha256 = state.get("file_content_sha256")
        if not file_content_sha256:
            logger.error("file_content_sha256 不能为空")
            raise ValueError("file_content_sha256 不能为空")

        file_title = state.get("file_title")
        if not file_title:
            logger.error("file_title 不能为空")
            raise ValueError("file_title 不能为空")

        # 校验2：切片包含dense_vector字段
        first_chunk = chunks[0]
        if 'dense' not in first_chunk:
            raise ValueError("错误: 数据中缺失dense字段")

        # 校验3：切片包含 sparse_vector 字段
        if 'sparse' not in first_chunk:
            raise ValueError("错误: 数据中缺失sparse字段")

        return chunks, file_content_sha256, file_title

    def create_milvus_collection(self):
        milvus_client = get_milvus_client()
        if not milvus_client.has_collection(collection_name=KBImportConfig.CHUNKS_COLLECTION):
            # 创建列
            schema = milvus_client.create_schema(auto_id=True, enable_dynamic_field=True)
            schema.add_field(field_name="chunk_id", datatype=DataType.INT64, is_primary=True, auto_id=True)
            schema.add_field(field_name="content", datatype=DataType.VARCHAR, max_length=65535)
            schema.add_field(field_name="file_title", datatype=DataType.VARCHAR, max_length=100)
            schema.add_field(field_name="file_content_sha256", datatype=DataType.VARCHAR, max_length=64)
            schema.add_field(field_name="item_name", datatype=DataType.VARCHAR, max_length=100)
            schema.add_field(field_name="title", datatype=DataType.VARCHAR, max_length=100)
            schema.add_field(field_name="parent_title", datatype=DataType.VARCHAR, max_length=100)
            schema.add_field(field_name="part", datatype=DataType.INT8)
            schema.add_field(field_name="sparse_vector", datatype=DataType.SPARSE_FLOAT_VECTOR)
            schema.add_field(field_name="dense_vector", datatype=DataType.FLOAT_VECTOR,
                             dim=KBImportConfig.BGE_DENSE_DIM)
            # 创建索引
            index_params = milvus_client.prepare_index_params()
            index_params.add_index(
                field_name="sparse_vector",
                index_type="SPARSE_INVERTED_INDEX",
                metric_type="IP",
                params={"inverted_index_algo": "DAAT_MAXSCORE", "normalize": True, "quantization": "none"}
            )

            index_params.add_index(
                field_name="dense_vector",
                index_type="AUTOINDEX",
                metric_type="COSINE"
            )

            index_params.add_index(
                field_name="file_content_sha256",
                index_type="INVERTED"
            )

            milvus_client.create_collection(collection_name=KBImportConfig.CHUNKS_COLLECTION, schema=schema, index_params=index_params)

            logger.info(f"milvus collection {KBImportConfig.CHUNKS_COLLECTION} 创建成功")

    def delete_milvus_data(self, file_content_sha256):
        milvus_client = get_milvus_client()

        # SHA256只包含64个十六进制字符，也能避免直接拼接过滤条件时出现异常。
        if not re.fullmatch(r"[0-9a-fA-F]{64}", file_content_sha256):
            raise ValueError("file_content_sha256不是合法的SHA256")

        delete_result = milvus_client.delete(collection_name=KBImportConfig.CHUNKS_COLLECTION,
                                             filter=f'file_content_sha256 == "{file_content_sha256}"')
        deleted_count = delete_result["delete_count"]
        logger.info(f"milvus {KBImportConfig.CHUNKS_COLLECTION} 成功删除 {deleted_count} 条记录")

    def insert_milvus_data(self, chunks: List[Dict], file_content_sha256: str, file_title:str):

        milvus_client = get_milvus_client()

        insert_data = [{
            "content": chunk.get("chunk"),
            "file_content_sha256": file_content_sha256,
            "title": chunk.get("h2"),
            "parent_title": chunk.get("nearest_heading_title"),
            "part": chunk.get("nearest_heading_position"),
            "file_title": file_title,
            "item_name": chunk.get("item_name"),
            "sparse_vector": chunk.get("sparse"),
            "dense_vector": chunk.get("dense")
        } for chunk in chunks]

        insert_result=milvus_client.insert(collection_name=KBImportConfig.CHUNKS_COLLECTION, data=insert_data)

        logger.info(f"milvus {KBImportConfig.CHUNKS_COLLECTION} 成功写入 {insert_result.get('insert_count')} 条数据")

        for idx, chunk in enumerate(chunks):
            chunk["milvus_chunk_id"] =insert_result["ids"][idx]

if __name__ == '__main__':
    node = NodeImportMilvus()

    with open(r"E:\output\hak180产品安全手册\hak180产品安全手册_chunks.json", "r", encoding="utf-8") as f:
        chunks = json.loads(f.read())
    init_state = {
        "file_content_sha256": "dff3aaa8133427992d0e0693391287f936a81b4b3806bc1ebc3b047609026765",
        "chunks": chunks,
        "file_title": "hak180产品安全手册",
    }

    print(node(init_state)['chunks'][0]['milvus_chunk_id'])
