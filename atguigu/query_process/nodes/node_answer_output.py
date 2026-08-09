'''
@Author  :61022
@Time    :2026/8/9
@Desc    :
'''

from atguigu.query_process.base import NodeBase
from atguigu.query_process.state import QueryGraphState
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

        # TODO
        logger.info(f"【{self.name}】节点逻辑")

        return state