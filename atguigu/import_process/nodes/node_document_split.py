'''
@Author  :61022
@Time    :2026/7/31
@Desc    :
'''
from atguigu.import_process.base import NodeBase
from atguigu.import_process.state import ImportGraphState


class NodeDocumentSplit(NodeBase):
    """
    文档切分节点：智能文档切片
    """

    @property
    def name(self) -> str:
        return "node_document_split"

    def process(self, state: ImportGraphState):


        return state