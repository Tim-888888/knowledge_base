'''
@Author  :61022
@Time    :2026/8/9
@Desc    :
'''

"""
查询流程节点基类

定义统一的节点接口规范，提供通用功能
"""
from abc import abstractmethod, ABC

from atguigu.query_process.state import QueryGraphState
from atguigu.tool.logger import logger


class NodeBase(ABC):

    @property
    @abstractmethod
    def name(self) -> str:
        """节点名称，子类必须实现。"""
        raise NotImplementedError

    def __call__(self, state: QueryGraphState):
        """
        节点执行入口
        """
        try:
            logger.info(f"[{self.name}] 开始执行...")

            result = self.process(state)

            logger.info(f"[{self.name}] 结束执行...")

            return result
        except Exception as e:
            logger.error(f"[{self.name}] 执行失败: {e}")
            raise

    @abstractmethod
    def process(self, state: QueryGraphState):
        """
        节点的核心处理逻辑
        :return:
        """
        pass