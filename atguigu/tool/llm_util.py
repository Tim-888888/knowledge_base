'''
@Author  :61022
@Time    :2026/8/7
@Desc    :
'''
from langchain_openai import ChatOpenAI

from atguigu.config.config import KBImportConfig

llm_model: dict = {}


def get_llm_model(model: str | None = None, response_json: bool = False) -> ChatOpenAI:
    global llm_model
    if not model:
        model = KBImportConfig.LLM_DEFAULT_MODEL

    llm_key = (model, response_json)

    # 实现单例
    if llm_key in llm_model:
        return llm_model[llm_key]

    model_kwargs: dict = {}
    if response_json:
        model_kwargs["response_format"] = {"type": "json_object"}

    model_client = ChatOpenAI(
        api_key=KBImportConfig.CLOSEAI_API_KEY,
        base_url=KBImportConfig.CLOSEAI_API_BASE,
        model=model,
        reasoning_effort="none",  # 不思考
        verbosity="low",  # 简洁输出
        use_responses_api=False,
        model_kwargs=model_kwargs
    )

    llm_model[llm_key] = model_client

    return model_client


if __name__ == '__main__':
    llm = get_llm_model()
    print(llm.invoke(input="你是什么模型").content)
