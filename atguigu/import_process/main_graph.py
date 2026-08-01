'''
@Author  :61022
@Time    :2026/7/31
@Desc    :
'''

from dotenv import load_dotenv
from langgraph.constants import START, END
from langgraph.graph import StateGraph
from langgraph.graph.state import CompiledStateGraph

from atguigu.import_process.nodes.node_bge_embedding import NodeBGEEmbedding
from atguigu.import_process.nodes.node_document_split import NodeDocumentSplit
from atguigu.import_process.nodes.node_entry import NodeEntry
from atguigu.import_process.nodes.node_import_milvus import NodeImportMilvus
from atguigu.import_process.nodes.node_item_name_recognition import NodeItemNameRecognition
from atguigu.import_process.nodes.node_md_img import NodeMDImg
from atguigu.import_process.nodes.node_pdf_to_md import NodePDFToMD
from atguigu.import_process.state import ImportGraphState
from atguigu.tool.json_format_util import parse_json

load_dotenv()


# 面向对象方式, 创建主图
class KBImportWorkflow:
    def __init__(self):
        # 做节点初始化, 节点注册到边, 图对象的单例模式
        self._builder = StateGraph(state_schema=ImportGraphState)
        # 节点初始化, 添加节点
        self._add_nodes()
        # 节点注册到边
        self._register_edge()
        # 保存单例图 (懒加载)
        self._graph: CompiledStateGraph = None

    def _init_node(self):
        self._node_entry = NodeEntry()
        self._node_pdf_to_md = NodePDFToMD()
        self._node_md_img = NodeMDImg()
        self._node_document_split = NodeDocumentSplit()
        self._node_item_name_recognition = NodeItemNameRecognition()
        self._node_bge_embedding = NodeBGEEmbedding()
        self._node_import_milvus = NodeImportMilvus()

    def _add_nodes(self):
        self._init_node()
        self._builder.add_node(self._node_entry.name, self._node_entry)
        self._builder.add_node(self._node_pdf_to_md.name, self._node_pdf_to_md)
        self._builder.add_node(self._node_md_img.name, self._node_md_img)
        self._builder.add_node(self._node_document_split.name, self._node_document_split)
        self._builder.add_node(self._node_item_name_recognition.name, self._node_item_name_recognition)
        self._builder.add_node(self._node_bge_embedding.name, self._node_bge_embedding)
        self._builder.add_node(self._node_import_milvus.name, self._node_import_milvus)

    def _router_after_node_entry(self, state: ImportGraphState):
        if state.get('is_pdf_read_enabled', None):
            return self._node_pdf_to_md.name
        elif state.get('is_md_read_enabled', None):
            return self._node_md_img.name
        else:
            return END

    def _register_edge(self):
        # 入口节点
        self._builder.set_entry_point(self._node_entry.name)
        # 条件边
        self._builder.add_conditional_edges(self._node_entry.name, self._router_after_node_entry)
        # 静态边
        self._builder.add_edge(self._node_pdf_to_md.name, self._node_md_img.name)
        self._builder.add_edge(self._node_md_img.name, self._node_document_split.name)
        self._builder.add_edge(self._node_document_split.name, self._node_item_name_recognition.name)
        self._builder.add_edge(self._node_item_name_recognition.name, self._node_bge_embedding.name)
        self._builder.add_edge(self._node_bge_embedding.name, self._node_import_milvus.name)
        self._builder.add_edge(self._node_import_milvus.name, END)

    def _compile(self):
        if self._graph is None:
            self._graph = self._builder.compile()
        return self._graph

    def run(self, init_state: ImportGraphState, is_stream: bool = False) -> ImportGraphState:
        self._compile()
        if is_stream:
            return self._graph.stream(input=init_state)
        else:
            return self._graph.invoke(input=init_state)

    @classmethod
    def create_and_run(cls, init_state: ImportGraphState, is_stream: bool):
        # 类方法调用
        return cls().run(init_state, is_stream)


if __name__ == '__main__':
    graph = KBImportWorkflow()
    init_state = {
        "local_file_path": r"E:\大模型学习\其他班文件\0331班\尚硅谷大模型项目之掌柜智库\2.资料\04-设备手册汇总\doc\Aolynk CB304n Cable网桥 用户手册-5W100-整本手册.pdf"}
    result = graph.run(init_state, is_stream=False)

    # result = MainGraph.create_and_run(init_state, is_stream=False)
    print(parse_json(result))
