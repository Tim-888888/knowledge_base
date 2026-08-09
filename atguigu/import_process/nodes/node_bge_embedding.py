'''
@Author  :61022
@Time    :2026/7/31
@Desc    :
'''
import json
from typing import Dict, List, Tuple

from atguigu.import_process.base import NodeBase
from atguigu.import_process.state import ImportGraphState
from atguigu.tool.bge_m3_utils import get_embedding_for_milvus
from atguigu.tool.json_format_util import parse_json
from atguigu.tool.logger import logger


class NodeBGEEmbedding(NodeBase):
    """
    混合向量化节点：使用 BGE-M3 模型将文本转换为向量
    """

    @property
    def name(self) -> str:
        return "node_bge_embedding"

    def process(self, state: ImportGraphState):
        chunks, file_content_sha256, file_title = self.init_params(state)

        # chunk前面拼装item_name, 并进行混合向量化
        self.get_chunk_vetor(chunks)

        # 落盘json测试用
        with open(rf"E:\output\{file_title}\{file_title}_chunks.json", mode="w", encoding='utf-8') as f:
            f.write(parse_json(chunks))

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

        return chunks, file_content_sha256, file_title

    def get_chunk_vetor(self, chunks: List[Dict]):
        # chunk前面拼装item_name
        chunks_content = [chunk.get("item_name") + "\n\n" + chunk.get("chunk") for chunk in chunks]
        hybrid_embedding = get_embedding_for_milvus(chunks_content)
        dense = hybrid_embedding.get("dense")
        sparse = hybrid_embedding.get("sparse")
        for idx, chunk in enumerate(chunks):
            chunk["dense"]=dense[idx]
            chunk["sparse"]=sparse[idx]


if __name__ == '__main__':
    node = NodeBGEEmbedding()
    with open(r"E:\output\hak180产品安全手册\hak180产品安全手册_item_name.json", "r", encoding="utf-8") as f:
        chunks = json.loads(f.read())
    init_state = {
        "file_content_sha256": "dff3aaa8133427992d0e0693391287f936a81b4b3806bc1ebc3b047609026765",
        "chunks": chunks,
        "file_title": "hak180产品安全手册",
    }
    print(node(init_state))
