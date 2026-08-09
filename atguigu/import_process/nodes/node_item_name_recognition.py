'''
@Author  :61022
@Time    :2026/7/31
@Desc    :
'''
import json
from pathlib import Path
from typing import Tuple, List, Any, Dict

from langchain_core.messages import SystemMessage, HumanMessage
from pymilvus import DataType, MilvusClient

from atguigu.config.config import KBImportConfig
from atguigu.import_process.base import NodeBase
from atguigu.import_process.prompt import NAME_RECOGNITION
from atguigu.import_process.state import ImportGraphState
from atguigu.tool.bge_m3_utils import get_embedding_for_milvus
from atguigu.tool.json_format_util import parse_json
from atguigu.tool.llm_util import get_llm_model
from atguigu.tool.logger import logger
from atguigu.tool.milvus_utils import get_milvus_client


class NodeItemNameRecognition(NodeBase):
    """
    主体识别节点：主体识别与标签提取, 用于检索时加速检索和提高检索精确度
    """

    @property
    def name(self) -> str:
        return "node_item_name_recognition"

    def process(self, state: ImportGraphState):

        # 初始化参数
        file_title, chunks, file_content_sha256 = self.init_param(state)

        # 获取top-K个chunks, 组成提示词
        top_k_chunks = self.get_top_k_chunks(chunks, file_title)

        # 调用LLM, 输入提示词, 进行主体识别
        item_name = self.get_item_name(top_k_chunks, file_title)

        # 主体信息回写chunks
        for chunk in chunks:
            chunk["item_name"] = item_name

        # 使用BGE-M3进行混合向量化
        dense, sparse = self.get_embedding_vector(item_name)

        # 创建milvus collection
        milvus_client = self.create_collection()

        # 将混合向量写入milvus, 幂等写入
        self.upsert_milvus(milvus_client, dense, sparse, file_content_sha256, file_title, item_name)

        # 落盘json测试用
        with open(rf"E:\output\{file_title}\{file_title}_item_name.json", mode="w", encoding='utf-8') as f:
            f.write(parse_json(chunks))

        return {"chunks": chunks}

    def get_embedding_vector(self, item_name: str) -> tuple[Any, Any]:
        """
        用BGE-M3 embedding模型, 为主体生成稠密向量和稀疏向量
        :param item_name:
        :return:
        """
        embedding_dict = get_embedding_for_milvus([item_name])
        dense = embedding_dict["dense"][0]
        sparse = embedding_dict["sparse"][0]
        return dense, sparse

    def init_param(self, state: ImportGraphState) -> Tuple[str, List[dict], str]:

        chunks = state.get("chunks")
        if not chunks:
            logger.error("chunks 不能为空")
            raise ValueError("chunks 不能为空")

        file_title = state.get("file_title")
        if not file_title:
            logger.error("file_title 不能为空")
            raise ValueError("file_title 不能为空")

        file_content_sha256 = state.get("file_content_sha256")
        if not file_content_sha256:
            logger.error("file_content_sha256 不能为空")
            raise ValueError("file_content_sha256 不能为空")

        return file_title, chunks, file_content_sha256

    def get_top_k_chunks(self, chunks: List[dict], file_title: str) -> str:

        TOP_K_CHUNKS = 10
        MAX_PROMPT_LENGTH = 50000
        current_length = 0
        chunk_list = [file_title, "\n"]

        for chunk in chunks[:TOP_K_CHUNKS]:
            chunk_content = chunk.get("raw_chunk")

            chunk_list.append(chunk_content)

            current_length += len(chunk_content)

            if current_length >= MAX_PROMPT_LENGTH:
                break

        prompts = "\n".join(chunk_list)[:MAX_PROMPT_LENGTH]

        return prompts

    def get_item_name(self, top_k_chunks: str, file_title: str):
        """如果调用大模型出错或者输出没有东西, 用file_title作为item_name"""
        try:
            llm = get_llm_model()

            messages = [
                SystemMessage(content="你是商品识别专家，只输出识别的字符串即可！"),
                HumanMessage(content=NAME_RECOGNITION.format(file_title=file_title, context=top_k_chunks))
            ]

            item_name = llm.invoke(messages).content

            if item_name:
                return item_name.strip().replace("\n", "").replace("\r", "").replace(" ", "")
            else:
                return file_title

        except Exception as e:
            logger.exception(f"调用大模型出错, 返回file_title, {e}")
            return file_title

    def create_collection(self) -> MilvusClient:

        milvus_client = get_milvus_client()

        if not milvus_client.has_collection(collection_name=KBImportConfig.ITEM_NAME_COLLECTION):
            # 创建字段
            schema = milvus_client.create_schema()
            # 用file_content_sha256作为主键, 文件内容级别唯一
            schema.add_field(
                field_name="id",
                datatype=DataType.VARCHAR,
                max_length=64,
                is_primary=True,
                auto_id=False,
                description="用file_content_sha256作为主键, 文件内容级别唯一"
            ).add_field(
                field_name="file_title",
                datatype=DataType.VARCHAR,
                max_length=100,
                description="文件名"
            ).add_field(
                field_name="item_name",
                datatype=DataType.VARCHAR,
                max_length=100,
                description="主体名称"
            ).add_field(
                field_name="dense_vector",
                datatype=DataType.FLOAT_VECTOR,
                dim=KBImportConfig.BGE_DENSE_DIM
            ).add_field(
                field_name="sparse_vector",
                datatype=DataType.SPARSE_FLOAT_VECTOR
            )
            # 创建索引
            index_params = milvus_client.prepare_index_params()
            """
            全部向量
               ↓
            使用 K-Means 分成 nlist=128 个聚类
               ↓
            查询向量与128个聚类中心比较
               ↓
            选择最近的 nprobe=10 个聚类
               ↓
            在这10个聚类内部进行精确向量比较
               ↓
            返回最相似的 TopK 结果
            """
            index_params.add_index(
                field_name='dense_vector',
                index_type='IVF_FLAT',
                metric_type='COSINE',
                params={"nlist": 128, "nprobe": 10}
            )

            index_params.add_index(
                field_name='sparse_vector',
                index_type='SPARSE_INVERTED_INDEX',
                metric_type='IP',
                params={
                    "inverted_index_algo": "DAAT_MAXSCORE",
                    # 高效的稀疏检索算法
                    "normalize": True,
                    # ↑ L2 归一化，让内积 (IP) 等价于余弦相似度
                    "quantization": "none"
                    # ↑ 关闭量化，保持原始精度：模型生成的向量已经压缩的一半的精度了（BGE_FP16=1），这里就不再压缩了
                    # "quantization": "none" → 存储原始向量，不压缩
                    # "quantization": "sq8" → 存储压缩后的向量（8-bit 量化
                }
            )

            milvus_client.create_collection(
                collection_name=KBImportConfig.ITEM_NAME_COLLECTION,
                schema=schema,
                index_params=index_params
            )
        return milvus_client

    def upsert_milvus(self, milvus_client: MilvusClient, dense, sparse, file_content_sha256: str,
                      file_title: str, item_name: str) -> Dict:

        data = {
            "id": file_content_sha256,
            "file_title": file_title,
            "item_name": item_name,
            "dense_vector": dense,
            "sparse_vector": sparse
        }

        result = milvus_client.upsert(collection_name=KBImportConfig.ITEM_NAME_COLLECTION, data=data)
        logger.info(f"Milvus成功upsert数据: {result}")


if __name__ == '__main__':
    node = NodeItemNameRecognition()

    with open(r"E:\output\hak180产品安全手册\hak180产品安全手册_new.json", "r", encoding="utf-8") as f:
        chunks = json.loads(f.read())

    init_state = {
        "chunks": chunks,
        "file_title": "hak180产品安全手册",
        "file_content_sha256": "dff3aaa8133427992d0e0693391287f936a81b4b3806bc1ebc3b047609026765"
    }

    print(node(init_state))
