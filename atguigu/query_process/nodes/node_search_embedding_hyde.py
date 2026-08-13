'''
@Author  :61022
@Time    :2026/8/9
@Desc    :
'''
import json
from typing import List

from langchain_core.messages import HumanMessage
from pymilvus import AnnSearchRequest, WeightedRanker

from atguigu.config.config import KBImportConfig
from atguigu.query_process.base import NodeBase
from atguigu.query_process.prompt import HYDE_PROMPT
from atguigu.query_process.state import QueryGraphState
from atguigu.tool.bge_m3_utils import get_embedding_for_milvus
from atguigu.tool.json_format_util import parse_json
from atguigu.tool.llm_util import get_llm_model
from atguigu.tool.logger import logger
from atguigu.tool.milvus_utils import get_milvus_client


class NodeSearchEmbeddingHyde(NodeBase):
    """
    节点功能：HyDE (Hypothetical Document Embedding) 假设文档嵌入
    先让 LLM 生成假设性答案，再对答案进行Milvus向量检索，提高召回率。
    """

    # 覆盖基类的 name 属性，标识节点名称
    @property
    def name(self) -> str:
        return "node_search_embedding_hyde"

    def process(self, state: QueryGraphState) -> QueryGraphState:
        """
        节点逻辑
        :param state: 工作流状态对象
        :return: 更新后的状态对象
        """

        # 参数校验
        item_names = state.get("item_names")
        if not item_names:
            raise ValueError("item_names 不能为空")

        rewritten_query = state.get("rewritten_query")
        if not rewritten_query:
            raise ValueError("rewritten_query 不能为空")

        # 假设性文档生成
        llm = get_llm_model()

        messages = [
            HumanMessage(HYDE_PROMPT.format(rewritten_query=rewritten_query))
        ]

        hyde_doc = llm.invoke(messages).content

        final_message=f"{rewritten_query}  {hyde_doc}"

        embedding_contents = get_embedding_for_milvus([final_message])
        dense = embedding_contents.get('dense')[0]
        sparse = embedding_contents.get('sparse')[0]
        expr = f"item_name in {json.dumps(item_names)}"
        output_fields = ["chunk_id", "title", "file_title", "content", "item_name"]
        match_chunks = self.retrieval_milvus(dense, sparse, expr, output_fields)

        # 取出milvus里匹配出来的多个chunks ,组装数据返回
        hyde_embedding_chunks = [
            {
                **match_chunk.get("entity"),
                "score": match_chunk.get("distance"),
                "source": "local"
            }
            for match_chunk in match_chunks
        ]

        # return state
        return {"hyde_embedding_chunks": hyde_embedding_chunks}

    def retrieval_milvus(self, item_name_dense, item_name_sparse, expr: str = None, output_fields=None, limit=10) -> list[dict]:
        # 混合查询milvus
        dense_request = AnnSearchRequest(
            data=[item_name_dense],
            anns_field="dense_vector",
            param={"metric_type": "COSINE"},
            limit=limit,
            expr=expr
        )

        sparse_request = AnnSearchRequest(
            data=[item_name_sparse],
            anns_field="sparse_vector",
            param={"metric_type": "IP"},
            limit=limit,
            expr=expr
        )

        ranker = WeightedRanker(0.8, 0.3, norm_score=True)

        milvus_client = get_milvus_client()
        search_result: List[List[dict]] = milvus_client.hybrid_search(
            KBImportConfig.CHUNKS_COLLECTION,
            [dense_request, sparse_request],
            ranker=ranker,
            limit=limit,
            output_fields=output_fields
        )
        match_items = search_result[0]
        return match_items



if __name__ == "__main__":
    init_state = {
        "item_names": ["BrotherHAK180烫金机"],
        "rewritten_query": "请问hak180烫金机怎么使用？",
    }
    node_search_embedding_hyde = NodeSearchEmbeddingHyde()
    result = node_search_embedding_hyde(init_state)
    logger.info(parse_json(result))
