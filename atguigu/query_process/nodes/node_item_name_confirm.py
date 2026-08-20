'''
@Author  :61022
@Time    :2026/8/9
@Desc    :
'''
from typing import Tuple, Dict

from langchain_core.messages import SystemMessage, HumanMessage
from pymilvus import AnnSearchRequest, WeightedRanker

from atguigu.query_process.base import NodeBase
from atguigu.query_process.prompt import ITEM_NAME_EXTRACT_SYSTEM_PROMPT, ITEM_NAME_EXTRACT_TEMPLATE
from atguigu.query_process.state import QueryGraphState
from atguigu.tool.bge_m3_utils import get_embedding_for_milvus
from atguigu.tool.llm_util import get_llm_model
from atguigu.tool.milvus_utils import get_milvus_client
from atguigu.tool.mongo_history_utils import *


class NodeItemNameConfirm(NodeBase):
    """
    节点功能：确认用户问题中的核心商品名称。(意图识别)
    """

    # 覆盖基类的 name 属性，标识节点名称
    @property
    def name(self) -> str:
        return "node_item_name_confirm"

    def process(self, state: QueryGraphState) -> QueryGraphState:
        """
        节点逻辑
        :param state: 工作流状态对象
        :return: 更新后的状态对象
        	1.参数校验
			2.获取该session的历史对话
			3.存储用户本轮对话
			4.把历史对话+本轮对话传给LLM, 进行query改写和商品名识别
			5.商品名 混合向量化+查询milvus, 返回匹配到的商品和得分
			6.评分对齐, 按三个评分等级进行处理
			7.写入最终历史?
			8.返回结果
        """

        # 1.参数校验
        session_id, original_query = self.init_params(state)

        # 2.在mongoDB获取该session的历史对话
        history_list = mongo_get_recent_message_by_session(session_id, 10)

        # 3.存储用户本轮对话
        mongo_upsert_data(session_id, "user", original_query)

        # 4.把历史对话+本轮对话传给LLM, 进行query改写和商品名识别
        item_names, rewritten_query = self.rewrite_query_and_item_confirm(history_list, original_query)
        # 如果item_names没有内容?

        # 5.商品名 混合向量化+查询milvus, 返回匹配到的商品和得分
        match_item_and_score = self.get_milvus_match_items(item_names)

        # 6.评分对齐, 按三个评分等级进行处理, 对置信度(评分)进行过滤
        confirm_items, optional_items = self.score_align(match_item_and_score)

        # 7.写入最终历史?
        result_state = self.backfill_and_get_answer(confirm_items, optional_items, session_id, rewritten_query)

        # 8.返回结果
        return result_state

    def init_params(self, state: QueryGraphState):
        session_id = state.get("session_id")
        if not session_id:
            raise ValueError(f"session_id 不能为空")

        original_query = state.get("original_query")
        if not original_query:
            raise ValueError(f"original_query 不能为空")
        return session_id, original_query

    def rewrite_query_and_item_confirm(self, history_list: List[Dict], original_query) -> Tuple[list[str], str]:

        llm = get_llm_model()

        # 角色: 内容
        history_text = [f"{history.get('role')}: {history.get('text')}" for history in history_list]

        history_content = "\n".join(history_text)

        messages = [
            SystemMessage(ITEM_NAME_EXTRACT_SYSTEM_PROMPT),
            HumanMessage(ITEM_NAME_EXTRACT_TEMPLATE.format(history_text=history_content, original_query=original_query))
        ]

        result = llm.invoke(input=messages).content

        json_result: dict = json.loads(result)
        item_names: list[str] = json_result.get("item_names", [])
        # 简单清洗, 特殊符号替换和去重
        if item_names:
            item_names = [item_name.replace(" ", "").replace("\n", "").replace("\r", "") for item_name in item_names]
            item_names = list(set(item_names))

        rewritten_query: str = json_result.get("rewritten_query", original_query)
        return item_names, rewritten_query

    def get_milvus_match_items(self, item_names: list[str]) -> List[Dict]:
        """
           把分析出的item_names逐个向量化，并执行混合搜索，获取匹配评分
           :return: 格式：
                [
                    {
                        "extracted_name": "hak180烫金机"
                        "matches": [                          # 该商品名的TopN匹配结果，无则空列表
                            {
                                "item_name": "BrotherHAK180烫金机D01WD7001-00",  # Milvus中存储的标准化商品名
                                "score": 0.80                  # 混合搜索的相似度评分（0-1，越高越相似）
                            },
                            {
                                "item_name": "HAK170烫金机",  # Milvus中存储的标准化商品名
                                "score": 0.80                  # 混合搜索的相似度评分（0-1，越高越相似）
                            },
                        ]
                    },
                    ...
                ]
        """
        if item_names:
            embedding_contents = get_embedding_for_milvus(item_names)
            dense = embedding_contents.get('dense')
            sparse = embedding_contents.get('sparse')

            final_result = []

            for idx, item_name in enumerate(item_names):
                # 商品名 混合向量化+查询milvus, 返回匹配到的商品和得分
                item_name_dense = dense[idx]
                item_name_sparse = sparse[idx]

                # 检索Milvus
                match_items = self.retrieval_milvus(item_name_dense, item_name_sparse)

                # 取出milvus里匹配出来的多个item_names ,组装数据返回
                matches = [
                    {"original_item_name": item_name,
                     "search_item_name": match_item.get("entity", {}).get("item_name", ""),
                     "score": match_item.get("distance")}
                    for match_item in match_items
                ]
                final_result.extend(matches)
            return final_result
        return None

    def retrieval_milvus(self, item_name_dense, item_name_sparse) -> list[dict]:
        # 混合查询milvus
        dense_request = AnnSearchRequest(
            data=[item_name_dense],
            anns_field="dense_vector",
            param={"metric_type": "COSINE"},
            limit=5
        )

        sparse_request = AnnSearchRequest(
            data=[item_name_sparse],
            anns_field="sparse_vector",
            param={"metric_type": "IP"},
            limit=5
        )

        ranker = WeightedRanker(0.8, 0.3, norm_score=True)

        milvus_client = get_milvus_client()
        search_result: List[List[dict]] = milvus_client.hybrid_search(
            KBImportConfig.ITEM_NAME_COLLECTION,
            [dense_request, sparse_request],
            ranker=ranker,
            limit=5,
            output_fields=["item_name"]
        )
        match_items = search_result[0]
        return match_items

    def score_align(self, match_item_and_score: List[Dict]) -> Tuple[List[str], List[str]]:
        """
            6 根据Milvus搜索评分，逐个对齐step4提取的item_names，生成「确认商品名」和「候选商品名」
            对齐规则（优先级a>b>c>d）：
                    a  如果只评分高于0.85 → 直接确认该商品名
                    b  如果无0.85分以上结果 → 取分数≥0.6的最高前3个作为候选
                    c  如果无0.6分及以上结果 → 不返回任何商品名（确认+候选均为空）
            :param match_item_and_score: 列表[字典] - step5的返回结果，每个商品名的搜索匹配数据（格式同step5返回值）
            :return: 字典 - 商品名对齐结果，包含确认列表和候选列表，格式：
                     {
                         "confirmed_item_names": ["确认商品名1", "确认商品名2"],  # 去重后的确认商品名，无则空列表
                         "options": ["候选商品名1", "候选商品名2", ...]          # 去重后的候选商品名，无则空列表
                     }
        """
        confirm_items=[]
        optional_items=[]
        if match_item_and_score:
            confirm_items = [search_item.get("search_item_name")
                             for search_item in match_item_and_score
                             if search_item.get("score") >= 0.90]

            optional_items = [search_item.get("search_item_name")
                              for search_item in match_item_and_score
                              if 0.6 <= search_item.get("score") < 0.90]

        return confirm_items, optional_items

    def backfill_and_get_answer(self, confirm_items: List[str], optional_items: List[str],
                                session_id: str, rewritten_query: str) -> Dict:
        """根据item, 生成答案和最终确认的实体"""
        answer = ""
        final_confirm_items = []

        if confirm_items:
            # 无答案
            final_confirm_items = confirm_items
        elif optional_items:
            answer = f"请问你是要咨询哪种商品? [{', '.join(optional_items)}]"
        else:
            answer = f"未识别到你想要咨询的商品, 请重新输入商品名"

        if answer:
            # 本次answer写入对话历史
            mongo_upsert_data(session_id, "assistant", answer)

        # 更新最近十条记录
        recent_record = mongo_get_recent_message_by_session(session_id)
        ids = [str(record.get("_id")) for record in recent_record]
        if ids:
            mongo_update_item_name_by_id(ids, final_confirm_items, rewritten_query)

        # 最终state
        result_state = {
            "answer": answer,
            "item_names": final_confirm_items,
            "rewritten_query": rewritten_query,
            "history": recent_record
        }

        return result_state


if __name__ == "__main__":
    # 初始化图状态
    # 模拟会话历史
    session_id = "test_001"
    # mongo_upsert_data(session_id, "user", "咨询下烫金机。")
    # mongo_upsert_data(session_id, "assistant", "您好。请问是哪个型号")
    # mongo_upsert_data(session_id, "user", "hak180")
    # mongo_upsert_data(session_id, "assistant", "具体有什么问题呢？")

    # 初始化图状态
    init_state = {
        "session_id": "test_001",
        "original_query": "咋用?"
    }

    # 创建节点对象
    node_item_name_confirm = NodeItemNameConfirm()
    # 执行节点的单元测试
    result = node_item_name_confirm(init_state)
    # 将返回的图状态进行json序列化
    logger.info(format_json(result))
