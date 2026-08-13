import json
from typing import List

from pymilvus import AnnSearchRequest, WeightedRanker

from atguigu.config.config import KBImportConfig
from atguigu.query_process.base import NodeBase
from atguigu.query_process.state import QueryGraphState
from atguigu.tool.bge_m3_utils import get_embedding_for_milvus
from atguigu.tool.json_format_util import parse_json
from atguigu.tool.logger import logger
from atguigu.tool.milvus_utils import get_milvus_client


class NodeSearchEmbedding(NodeBase):
    """
   节点功能：基于已确认主体名+改写后的用户问题，执行Milvus向量数据库混合检索
   """

    # 覆盖基类的 name 属性，标识节点名称
    @property
    def name(self) -> str:
        return "node_search_embedding"

    def process(self, state: QueryGraphState) -> QueryGraphState:
        """
        节点逻辑
        :param state: 工作流状态对象
        :return: 更新后的状态对象
        """

        item_names = state.get("item_names")
        if not item_names:
            raise ValueError("item_names 不能为空")

        rewritten_query = state.get("rewritten_query")
        if not rewritten_query:
            raise ValueError("rewritten_query 不能为空")

        embedding_contents = get_embedding_for_milvus([rewritten_query])
        dense = embedding_contents.get('dense')[0]
        sparse = embedding_contents.get('sparse')[0]
        expr = f"item_name in {json.dumps(item_names)}"
        output_fields = ["chunk_id", "title", "file_title", "content", "item_name"]
        match_chunks = self.retrieval_milvus(dense, sparse, expr, output_fields)

        # 取出milvus里匹配出来的多个chunks ,组装数据返回
        embedding_chunks = [
            {
                **match_chunk.get("entity"),
                "score": match_chunk.get("distance"),
                "source": "local"
            }
            for match_chunk in match_chunks
        ]

        # return state
        return {"embedding_chunks": embedding_chunks}

    def retrieval_milvus(self, item_name_dense, item_name_sparse, expr: str = None, output_fields=None, limit=10) -> list[dict]:
        # 混合查询milvus
        dense_request = AnnSearchRequest(
            data=[item_name_dense],
            anns_field="dense_vector",
            param={"metric_type": "COSINE"},
            limit=limit,
            expr=expr #这里会优先进行标量过滤, 再进行向量检索
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

    node_search_embedding = NodeSearchEmbedding()
    result = node_search_embedding(init_state)
    logger.info(parse_json(result))
