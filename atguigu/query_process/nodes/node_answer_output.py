'''
@Author  :61022
@Time    :2026/8/9
@Desc    :
'''
from email import message

from langchain_core.messages import HumanMessage, SystemMessage

from atguigu.query_process.base import NodeBase
from atguigu.query_process.state import QueryGraphState
from atguigu.tool.llm_util import get_llm_model
from atguigu.tool.logger import logger

class NodeAnswerOutput(NodeBase):
    """
    节点功能: 最终答案生成
    """

    # 覆盖基类的 name 属性，标识节点名称
    @property
    def name(self) -> str:
        return "node_answer_output"

    def process(self, state: QueryGraphState) -> QueryGraphState:
        """
        节点逻辑
        :param state: 工作流状态对象
        :return: 更新后的状态对象
        """
        answer=state.get("answer")
        if not answer:
            rrf_chunks:list[dict] = state.get("rrf_chunks")
            rewritten_query = state.get("rewritten_query")

            content="\n".join([f"{rrf_chunk.get('title')}, {rrf_chunk.get('content')}" for rrf_chunk in rrf_chunks])
            message=f"用户问题:{rewritten_query}\n 召回答案:{content}"

            llm = get_llm_model()

            input = [
                SystemMessage("根据提供的召回答案, 回复用户问题"),
                HumanMessage(message)
            ]

            answer=llm.invoke(input).content

        return {"answer":answer}
