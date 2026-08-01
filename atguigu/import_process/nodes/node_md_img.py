'''
@Author  :61022
@Time    :2026/7/31
@Desc    :
'''
from atguigu.import_process.base import NodeBase
from atguigu.import_process.state import ImportGraphState


class NodeMDImg(NodeBase):
    """
    MarkDown图片处理节点：多模态图片理解, 图转文
    """

    @property
    def name(self) -> str:
        return "node_md_img"

    def process(self, state: ImportGraphState):


        return state