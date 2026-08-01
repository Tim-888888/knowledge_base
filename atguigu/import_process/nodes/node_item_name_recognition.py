'''
@Author  :61022
@Time    :2026/7/31
@Desc    :
'''
from atguigu.import_process.base import NodeBase
from atguigu.import_process.state import ImportGraphState


class NodeItemNameRecognition(NodeBase):
    """
    主体识别节点：主体识别与标签提取
    """

    @property
    def name(self) -> str:
        return "node_item_name_recognition"

    def process(self, state: ImportGraphState):


        return state