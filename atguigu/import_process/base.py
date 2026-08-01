'''
@Author  :61022
@Time    :2026/7/30
@Desc    :
'''
from abc import ABC, abstractmethod

from atguigu.import_process.state import ImportGraphState
from atguigu.tool.logger import logger


# 定义LangGraph节点基类
class NodeBase(ABC):
    # name = "base_node"
    #
    # def __init__(self):
    #     # 校验节点名称
    #     if self.name == "base_node":
    #         raise Exception(f"{self.__class__.__name__}, 请定义节点名称")

    @property
    @abstractmethod
    def name(self) -> str:
        """节点名称，子类必须实现。"""
        raise NotImplementedError

    def __call__(self, state: ImportGraphState) -> ImportGraphState:
        """# 给LangGraph调用, Node子类对象调用的时候执行"""
        # 统一打印日志, 统一处理异常
        try:
            # 打印开始结束日志
            logger.info(f"[{self.name}] 开始执行节点")
            state = self.process(state)
            logger.info(f"[{self.name}] 结束执行节点")
            return state
        except Exception as e:
            logger.error(f"节点：{self.name} 执行异常：{e}")
            raise # 不raise e, 可以保留堆栈

    @abstractmethod
    def process(self, state: ImportGraphState) -> ImportGraphState:
        # 业务处理逻辑
        pass
