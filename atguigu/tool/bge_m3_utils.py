'''
@Author  :61022
@Time    :2026/8/6
@Desc    :
'''
import json
from typing import List

from pymilvus.model.hybrid import BGEM3EmbeddingFunction

from atguigu.config.config import KBImportConfig

model = None


# 单例 懒加载
def bge_m3_embedding_files(docs: List[str]):
    global model
    if not model:
        model = BGEM3EmbeddingFunction(
            model_name=KBImportConfig.BGE_M3_PATH,
            device=KBImportConfig.BGE_DEVICE,
            use_fp16=KBImportConfig.BGE_FP16
        )

    return model.encode_documents(docs)


def get_embedding_for_milvus(docs: List[str]):
    embedding_content = bge_m3_embedding_files(docs)

    return {
        "dense": [chunk.tolist() for chunk in embedding_content["dense"]],
        "sparse": [dict(zip(chunk.indices.tolist(), chunk.data.tolist())) for chunk in embedding_content["sparse"]],
    }


if __name__ == '__main__':
    docs = [
        "Artificial intelligence was founded as an academic discipline in 1956.",
        "Alan Turing was the first person to conduct substantial research in AI.",
        "Born in Maida Vale, London, Turing was raised in southern England.",
    ]

    embedding_content = get_embedding_for_milvus(docs)
    result = json.dumps(embedding_content, indent=4, ensure_ascii=False)
    print(result)
    # dense = embedding_content["dense"]
    # sparse = embedding_content["sparse"]
    #
    # print(len(dense), len(sparse))
