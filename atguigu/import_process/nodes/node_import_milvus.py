'''
@Author  :61022
@Time    :2026/7/31
@Desc    :
'''
from atguigu.import_process.base import NodeBase
from atguigu.import_process.state import ImportGraphState


class NodeImportMilvus(NodeBase):
    """
    导入向量库节点：数据持久化
    """

    @property
    def name(self) -> str:
        return "node_import_milvus"

    def process(self, state: ImportGraphState):


        return state