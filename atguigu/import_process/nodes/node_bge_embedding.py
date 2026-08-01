'''
@Author  :61022
@Time    :2026/7/31
@Desc    :
'''
from atguigu.import_process.base import NodeBase
from atguigu.import_process.state import ImportGraphState


class NodeBGEEmbedding(NodeBase):
    """
    混合向量化节点：使用 BGE-M3 模型将文本转换为向量
    """

    @property
    def name(self) -> str:
        return "node_bge_embedding"

    def process(self, state: ImportGraphState):


        return state