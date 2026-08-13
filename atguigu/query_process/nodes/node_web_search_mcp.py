'''
@Author  :61022
@Time    :2026/8/9
@Desc    :
'''
import asyncio
import json

from agents.mcp import MCPServerStreamableHttp

from atguigu.config.config import KBImportConfig
from atguigu.query_process.base import NodeBase
from atguigu.query_process.state import QueryGraphState
from atguigu.tool.json_format_util import parse_json
from atguigu.tool.logger import logger


# from atguigu.tool.mongo_history_utils import format_json


class NodeWebSearchMcp(NodeBase):
    """
    节点功能，调用外部MCP搜索引擎补充信息
    """

    # 覆盖基类的 name 属性，标识节点名称
    @property
    def name(self) -> str:
        return "node_web_search_mcp"

    def process(self, state: QueryGraphState) -> QueryGraphState:
        """
        节点逻辑
        :param state: 工作流状态对象
        :return: 更新后的状态对象
        """

        query = state.get("rewritten_query", "")
        docs = []
        # 如果没有查询内容，直接返回
        if query:
            result = asyncio.run(self._mcp_call(query))
            if result:
                pages = json.loads(result.content[0].text).get("pages") or []
                # 统一输出结构化结果，供后续 rerank/引用使用
                # 每条：{title, url, snippet}

                for item in pages:
                    snippet = (item.get("snippet") or "").strip()
                    url = (item.get("url") or "").strip()
                    title = (item.get("title") or "").strip()
                    if not snippet:
                        continue
                    docs.append({"title": title, "url": url, "snippet": snippet, "source": "web"})

                logger.info("MCP 搜索结果:", docs)

        if docs:
            return {"web_search_docs": docs}
        return {}

    async def _mcp_call(self, query):
        search_mcp = MCPServerStreamableHttp(
            name="search_mcp",
            params={
                "url": KBImportConfig.MCP_DASHSCOPE_BASE_URL,
                "headers": {"Authorization": f"Bearer {KBImportConfig.OPENAI_API_KEY}"},
                "timeout": 30,
            },
            cache_tools_list=True,
            max_retry_attempts=3,
        )

        try:
            await search_mcp.connect()
            result = await search_mcp.call_tool(
                tool_name="bailian_web_search",
                arguments={"query": query, "count": 5},
            )
            return result
        finally:
            await search_mcp.cleanup()
        # return state


if __name__ == "__main__":
    init_state = {
        "rewritten_query": "关于brother HAK180烫金机，如何调节转印温度？"
    }

    # 执行节点的业务调用
    node_web_search_mcp = NodeWebSearchMcp()
    result = node_web_search_mcp(init_state)
    logger.info(parse_json(result))
    # logger.info(format_json(result, indent=4))
