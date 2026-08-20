'''
@Author  :61022
@Time    :2026/8/9
@Desc    :
'''
import re

from langchain_core.messages import HumanMessage

from atguigu.query_process.base import NodeBase
from atguigu.query_process.prompt import ANSWER_PROMPT
from atguigu.query_process.state import QueryGraphState
from atguigu.tool.llm_util import get_llm_model
from atguigu.tool.mongo_history_utils import mongo_upsert_data
from atguigu.tool.task_utils import put_queue_data


class NodeAnswerOutput(NodeBase):
    """
    节点功能: 最终答案生成
    """

    # 覆盖基类的 name 属性，标识节点名称
    @property
    def name(self) -> str:
        return "node_answer_output"

    def process(self, state: QueryGraphState) -> QueryGraphState:
        """
        节点逻辑
        :param state: 工作流状态对象
        :return: 更新后的状态对象
        """
        answer = state.get("answer")
        task_id = state.get("task_id")
        if answer:
            # 有答案, 走澄清或者闲聊意图, 直接向前端输出结果
            put_queue_data(task_id, "final", {"answer": answer})
        else:
            # 组装prompt
            content, history, item_names, reranked_docs, rewritten_query = self.build_prompt(state)

            # prompt发送给LLM, 获取最终答案
            answer = self.get_llm_answer(answer, content, history, item_names, rewritten_query, state, task_id)

            # 获取图片, 组装事件写入队列
            image_urls = self.send_image_urls(reranked_docs, task_id)

            # ai回复写入历史对话
            session_id = state.get("session_id")
            mongo_upsert_data(session_id, "assistant", answer, rewritten_query, item_names, image_urls=image_urls)

        return {"answer": answer}

    def send_image_urls(self, reranked_docs: str, task_id: str):
        seen = set()  # 用于去重，避免同一张图片重复出现
        md_img_pattern = re.compile(r'!\[.*?\]\((.*?)\)')
        for i, doc in enumerate(reranked_docs):
            # 检查 text 字段中的 Markdown 图片 (主要针对 Local Chunk)
            text = doc.get("content")
            matches = md_img_pattern.findall(text)
            for img_url in matches:
                img_url = img_url.strip()
                if img_url and img_url not in seen:
                    seen.add(img_url)
        seen = list(seen)
        image_urls = "\n".join(seen)
        put_queue_data(task_id, "final", {"image_urls": image_urls})
        return seen

    def get_llm_answer(self, answer: str, content: str, history: str, item_names: str, rewritten_query: list[dict],
                       state: QueryGraphState, task_id: str) -> str:
        llm = get_llm_model()

        message = [
            HumanMessage(ANSWER_PROMPT.format(context=content, history=history, item_names=item_names,
                                              question=rewritten_query))
        ]

        answer = ""
        answer_generator = llm.stream(message)
        for chunk in answer_generator:
            result_content = chunk.content
            put_queue_data(task_id, "delta", {"delta": result_content})
            answer += result_content

        return answer

    def build_prompt(self, state: QueryGraphState) -> tuple[str, str, str, str, list[dict]]:
        reranked_docs: list[dict] = state.get("reranked_docs")
        content = "\n\n".join(
            [f"[{doc.get('source')}] [{doc.get('title')}] [{idx}] [{doc.get('url')}] [{doc.get('content')}]" for
             idx, doc in enumerate(reranked_docs)])
        rewritten_query = state.get("rewritten_query")
        history_list = state.get("history")
        history = "\n".join([f'{history.get("role", "")}: {history.get("text", "")}' for history in history_list])
        item_names = ",".join(state.get("item_names"))
        return content, history, item_names, reranked_docs, rewritten_query
