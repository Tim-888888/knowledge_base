'''
@Author  :61022
@Time    :2026/8/1
@Desc    :
'''
import os

from dotenv import load_dotenv

load_dotenv(override=True)


class KBImportConfig:
    MINERU_API_TOKEN = os.getenv("MINERU_API_TOKEN")
    MINERU_BASE_URL = os.getenv("MINERU_BASE_URL")

    MINERU_MODEL_SOURCE = os.getenv("MINERU_MODEL_SOURCE")
    MODELSCOPE_OFFLINE = os.getenv("MODELSCOPE_OFFLINE")
    MODELSCOPE_CACHE = os.getenv("MODELSCOPE_CACHE")
    HF_HOME = os.getenv("HF_HOME")
    MD_ROOT_DIR = os.getenv("MD_ROOT_DIR")

    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    OPENAI_API_BASE = os.getenv("OPENAI_API_BASE")
    LLM_DEFAULT_MODEL = os.getenv("LLM_DEFAULT_MODEL")
    LLM_DEFAULT_TEMPERATURE = float(os.getenv("LLM_DEFAULT_TEMPERATURE"))
    VL_MODEL = os.getenv("VL_MODEL")
    ITEM_MODEL = os.getenv("ITEM_MODEL")

    BGE_M3_PATH = os.getenv("BGE_M3_PATH")
    BGE_M3 = os.getenv("BGE_M3")
    BGE_DEVICE = os.getenv("BGE_DEVICE")
    BGE_FP16 = os.getenv("BGE_FP16") in ("1", "True", "true", 1)
    BGE_RERANKER_LARGE = os.getenv("BGE_RERANKER_LARGE")
    BGE_RERANKER_DEVICE = os.getenv("BGE_RERANKER_DEVICE")
    BGE_RERANKER_FP16 = os.getenv("BGE_RERANKER_FP16")

    EMBEDDING_DIM = os.getenv("EMBEDDING_DIM")
    EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL")

    MILVUS_URL = os.getenv("MILVUS_URL")
    CHUNKS_COLLECTION = os.getenv("CHUNKS_COLLECTION")
    ITEM_NAME_COLLECTION = os.getenv("ITEM_NAME_COLLECTION")
    MILVUS_METRIC_TYPE = os.getenv("MILVUS_METRIC_TYPE")
    MILVUS_MIN_COSINE_SCORE = os.getenv("MILVUS_MIN_COSINE_SCORE")

    MONGO_URL = os.getenv("MONGO_URL")
    MONGO_DB_NAME = os.getenv("MONGO_DB_NAME")

    MINIO_ENDPOINT = os.getenv("MINIO_ENDPOINT")
    MINIO_ACCESS_KEY = os.getenv("MINIO_ACCESS_KEY")
    MINIO_SECRET_KEY = os.getenv("MINIO_SECRET_KEY")
    MINIO_BUCKET_NAME = os.getenv("MINIO_BUCKET_NAME")
    MINIO_IMG_DIR = os.getenv("MINIO_IMG_DIR")

    MCP_DASHSCOPE_BASE_URL = os.getenv("MCP_DASHSCOPE_BASE_URL")

    LOCAL_DIR = os.getenv("LOCAL_DIR")
