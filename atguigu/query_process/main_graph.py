'''
@Author  :61022
@Time    :2026/8/9
@Desc    :
'''

from typing import Optional

from langgraph.constants import END
from langgraph.graph import StateGraph

from atguigu.query_process.nodes.node_answer_output import NodeAnswerOutput
from atguigu.query_process.nodes.node_item_name_confirm import NodeItemNameConfirm
from atguigu.query_process.nodes.node_rerank import NodeRerank
from atguigu.query_process.nodes.node_rrf import NodeRrf
from atguigu.query_process.nodes.node_search_embedding import NodeSearchEmbedding
from atguigu.query_process.nodes.node_search_embedding_hyde import NodeSearchEmbeddingHyde
from atguigu.query_process.nodes.node_web_search_mcp import NodeWebSearchMcp
from atguigu.query_process.state import QueryGraphState
from atguigu.tool.json_format_util import parse_json
from atguigu.tool.logger import logger
from atguigu.tool.mongo_history_utils import format_json


class KBQueryWorkflow:
    """
    知识库查询工作流类
    封装LangGraph工作流的构建、编译、执行逻辑，支持自定义配置和多实例运行
    """
    def __init__(self):
        """初始化工作流：创建状态图、注册节点、定义路由规则"""
        # 1. 初始化LangGraph状态图
        self.workflow = StateGraph(QueryGraphState)
        # 2. 初始化所有业务节点（实例属性，支持多实例隔离）
        # self._init_nodes()
        # 3. 注册节点到工作流
        self._register_nodes()
        # 4. 设置入口和路由规则
        self._setup_routes()
        # 5. 编译工作流（懒加载，首次执行时编译）
        self._compiled_app: Optional[object] = None

    def _init_nodes(self):
        """初始化所有业务节点（私有方法，封装节点创建逻辑）"""
        self.node_item_name_confirm = NodeItemNameConfirm()
        self.node_search_embedding = NodeSearchEmbedding()
        self.node_search_embedding_hyde = NodeSearchEmbeddingHyde()
        self.node_web_search_mcp = NodeWebSearchMcp()
        self.node_rrf = NodeRrf()
        self.node_rerank = NodeRerank()
        self.node_answer_output = NodeAnswerOutput()

    def _register_nodes(self):
        """注册所有节点到工作流（私有方法，统一管理节点注册）"""
        self._init_nodes()
        # 节点标识与实例属性名保持一致，便于维护
        self.workflow.add_node(self.node_item_name_confirm.name, self.node_item_name_confirm) # 确认主体
        self.workflow.add_node(self.node_search_embedding.name, self.node_search_embedding) # 向量搜索
        self.workflow.add_node(self.node_search_embedding_hyde.name, self.node_search_embedding_hyde) #假设性答案向量搜索
        self.workflow.add_node(self.node_web_search_mcp.name, self.node_web_search_mcp) # 联网搜索
        self.workflow.add_node(self.node_rrf.name, self.node_rrf) # 排序
        self.workflow.add_node(self.node_rerank.name, self.node_rerank) # 重排
        self.workflow.add_node(self.node_answer_output.name, self.node_answer_output) # 生成

    def _route_after_item_name_confirm(self, state: QueryGraphState) -> str:
        """主体名称确认后的条件路由函数"""
        if state.get("answer"):
            """
            这主要发生在 node_item_name_confirm 节点无法直接确定唯一的商品型号，从而需要“反问用户”或“拒绝回答”的场景。
            具体来说，有以下两种情况会导致 state 中直接出现 answer ，从而跳过后续的检索流程，直接输出：
            1. 多选一（反问用户） ：
            - 场景 ：用户问得太模糊（比如“华为P60”），系统发现数据库里有“华为P60 128G”和“华为P60 Art”两个型号，不足以直接确认。
            - 处理 ：节点会生成一条反问句作为 answer ，例如：“您是想问以下哪个产品：华为P60 128G、华为P60 Art？请明确一下型号。”
            - 结果 ：此时不需要去检索文档了，直接把这句话发给用户让他选。

            2. 查无此人（拒绝回答） ：
            - 场景 ：用户问了一个系统里压根没有的商品（比如“小米15”，库里只有华为的数据）。
            - 处理 ：节点会生成一条拒绝句作为 answer ，例如：“抱歉，未找到相关产品，请提供准确型号以便我为您查询。”
            - 结果 ：同样不需要后续检索，直接结束流程。
            """
            # return "node_answer_output"
            return self.node_answer_output.name

        # 否则继续搜索流程
        # return ["node_search_embedding", "node_search_embedding_hyde", "node_web_search_mcp"]
        return [self.node_search_embedding.name, self.node_search_embedding_hyde.name, self.node_web_search_mcp.name]


    def _setup_routes(self):
        """设置工作流路由规则"""
        # 1、设置入口节点
        self.workflow.set_entry_point(self.node_item_name_confirm.name)

        # 2、注册条件路由边
        self.workflow.add_conditional_edges(
            self.node_item_name_confirm.name,
            self._route_after_item_name_confirm,
            {
                self.node_answer_output.name: self.node_answer_output.name,
                self.node_search_embedding.name: self.node_search_embedding.name,
                self.node_search_embedding_hyde.name: self.node_search_embedding_hyde.name,
                self.node_web_search_mcp.name: self.node_web_search_mcp.name
            }
        )

        # 3. 多路搜索结果合并
        self.workflow.add_edge(self.node_search_embedding.name, self.node_rrf.name)
        self.workflow.add_edge(self.node_search_embedding_hyde.name, self.node_rrf.name)
        self.workflow.add_edge(self.node_web_search_mcp.name, self.node_rrf.name)

        # 4. 排序 -> 重排 -> 生成 -> 结束
        self.workflow.add_edge(self.node_rrf.name, self.node_rerank.name)
        self.workflow.add_edge(self.node_rerank.name, self.node_answer_output.name)
        self.workflow.add_edge(self.node_answer_output.name, END)


    def compile(self):
        """编译工作流（公开方法，支持手动触发编译）"""
        if not self._compiled_app:
            self._compiled_app = self.workflow.compile()
        return self._compiled_app

    def run(self, initial_state: QueryGraphState, stream: bool = False) -> QueryGraphState:
        """
        统一执行入口，支持切换invoke/stream
        :param initial_state:  初始状态对象
        :param stream: 是否是流式输出
        :return: 执行完成后的状态对象
        """
        """"""
        if self._compiled_app is None:
            self._compiled_app = self.workflow.compile()

        self._compiled_app.get_graph().print_ascii()

        if stream:
            return self._compiled_app.stream(initial_state)
        else:
            return self._compiled_app.invoke(initial_state)

    @classmethod
    def create_and_run(cls, init_state: QueryGraphState, stream: bool = False) -> QueryGraphState:
        return cls().run(init_state, stream)

# ===================== 用法示例 =====================
if __name__ == "__main__":

    # 定义初始状态
    init_state = {
        "session_id": "test_001",
        "original_query": "请问hak180烫金机怎么使用？"
    }

    workflow = KBQueryWorkflow()
    final_state = workflow.run(init_state)
    logger.warning(format_json(final_state['answer']))
    # 流式输出
    # for chunk in workflow.run(init_state, stream=True):
    #     logger.warning(chunk)

    # 打印编译后的图结构
    # logger.info(workflow.compile().get_graph().draw_ascii())