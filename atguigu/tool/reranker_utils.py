'''
@Author  :61022
@Time    :2026/8/13
@Desc    :
'''
from idlelib import query
from typing import List

import dashscope
from http import HTTPStatus

from atguigu.config.config import KBImportConfig


def get_reranker_result(query: str, documents: List[str]) -> List[float]:
    dashscope.base_http_api_url = KBImportConfig.TEXT_RERANK_BASE_URL
    dashscope.api_key = KBImportConfig.TEXT_RERANK_API_KEY

    resp = dashscope.TextReRank.call(
        model=KBImportConfig.TEXT_RERANK_MODEL,
        query=query,
        documents=documents,
        top_n=len(documents),
        return_documents=False,
        instruct=KBImportConfig.TEXT_RERANK_INSTRUCT
    )

    if resp.status_code != HTTPStatus.OK:
        raise RuntimeError(f"获取请求失败, status_code={resp.status_code}, message={resp.message}")

    results = resp.output.results
    # 维持documents的顺序, 填充分数
    scores = [0.0] * len(documents)
    for result in results:
        index = result.index
        relevance_score = result.relevance_score
        scores[index] = relevance_score

    return scores


if __name__ == '__main__':
    query = "什么是重排序模型"
    documents = [
        "重排序模型广泛应用于搜索引擎和推荐系统，用于按相关性对候选文本排序",
        "量子计算是计算科学的前沿领域",
        "预训练语言模型的发展为重排序模型带来了新的突破"
    ]
    print(get_reranker_result(query, documents))
