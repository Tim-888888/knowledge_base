'''
@Author  :61022
@Time    :2026/7/31
@Desc    :
'''
import json
from typing import Tuple, List, Any

from langchain_core.messages import SystemMessage, HumanMessage

from atguigu.import_process.base import NodeBase
from atguigu.import_process.prompt import NAME_RECOGNITION
from atguigu.import_process.state import ImportGraphState
from atguigu.tool.bge_m3_utils import get_embedding_for_milvus
from atguigu.tool.llm_util import get_llm_model
from atguigu.tool.logger import logger


class NodeItemNameRecognition(NodeBase):
    """
    主体识别节点：主体识别与标签提取, 用于检索时加速检索和提高检索精确度
    """

    @property
    def name(self) -> str:
        return "node_item_name_recognition"

    def process(self, state: ImportGraphState):

        # 初始化参数
        file_title, chunks = self.init_param(state)

        # 获取top-K个chunks, 组成提示词
        top_k_chunks = self.get_top_k_chunks(chunks, file_title)

        # 调用LLM, 输入提示词, 进行主体识别
        item_name = self.get_item_name(top_k_chunks, file_title)
        print(item_name)

        # 主体信息回写chunks
        for chunk in chunks:
            chunk["item_name"] = item_name

        # 使用BGE-M3进行混合向量化
        dense, sparse = self.get_embedding_vector(item_name)

        # 混合向量写入milvus

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

    def init_param(self, state: ImportGraphState) -> Tuple[str, List[dict]]:

        chunks = state.get("chunks")
        if not chunks:
            logger.error("chunks 不能为空")
            raise ValueError("chunks 不能为空")

        file_title = state.get("file_title")
        if not file_title:
            logger.error("file_title 不能为空")
            raise ValueError("file_title 不能为空")

        return file_title, chunks

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


if __name__ == '__main__':
    node = NodeItemNameRecognition()

    with open(r"E:\output\hak180产品安全手册\hak180产品安全手册_new.json", "r", encoding="utf-8") as f:
        chunks = json.loads(f.read())

    init_state = {
        "chunks": chunks,
        "file_title": "hak180产品安全手册"
    }

    print(node(init_state))
