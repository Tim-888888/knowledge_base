'''
@Author  :61022
@Time    :2026/8/9
@Desc    :
'''
import json
from datetime import datetime
from typing import List, Any
from unittest import result

from bson import ObjectId
from pymongo import MongoClient, cursor
from pymongo.synchronous.collection import Collection

from atguigu.config.config import KBImportConfig
from atguigu.tool.logger import logger


# 用于管理历史对话
# 一个文档数据库, 存储格式是bson, 一种二进制json格式
# 里面存储结构是 db(库) -> collection(表) -> document(数据行)
class HistoryMongoTool:
    def __init__(self):
        self.mongo_url = KBImportConfig.MONGO_URL
        self.mongo_db_name = KBImportConfig.MONGO_DB_NAME
        # 客户端
        self.mongo_client = MongoClient(host=self.mongo_url)
        # 库
        self.mongo_db = self.mongo_client[self.mongo_db_name]
        # 集合
        self.chat_message_collection: Collection = self.mongo_db['chat_message']

        # 索引
        self.chat_message_collection.create_index([("session_id", 1), ("ts", -1)])


# 预加载
_get_history_mongo_tool = HistoryMongoTool()


def get_history_mongo_tool():
    """
    获取mongoDB客户端工具
    数据结构:
    {
        "session_id": "",
        "role": "",
        "text": "",
        "rewritten_query": "",
        "item_names": [],
        "ts": 123
    }
    :return:
    """
    global _get_history_mongo_tool
    if not _get_history_mongo_tool:
        _get_history_mongo_tool = HistoryMongoTool()

    return _get_history_mongo_tool


# 增
def mongo_upsert_data(session_id: str,
                      role: str,
                      text: str,
                      rewritten_query: str = "",
                      item_names: list[str] = None,
                      message_id: str = None):
    mongo_tool = get_history_mongo_tool()

    try:
        message = {
            "session_id": session_id,
            "role": role,
            "text": text,
            "rewritten_query": rewritten_query,
            "item_names": item_names,
            "ts": datetime.now().timestamp()
        }

        # 根据主键id是否存在, 进行更新/新增
        if message_id:
            # 修改
            mongo_tool.chat_message_collection.update_one({"_id": ObjectId(message_id)}, {"$set": message})
            return message_id
        else:
            # 新增
            result = mongo_tool.chat_message_collection.insert_one(message)
            return str(result.inserted_id)
    except Exception as e:
        logger.error(f"mongo更新/插入失败 {session_id}, {e}")
        raise


# 删
def mongo_delete_session(session_id: str):
    mongo_tool = get_history_mongo_tool()
    try:
        result = mongo_tool.chat_message_collection.delete_many({"session_id": session_id})
        return result.deleted_count
    except Exception as e:
        logger.error(f"mongo删除失败 {session_id}")
        return 0


# 改
def mongo_update_item_name_by_id(ids: List[str], item_names: List[str], rewritten_query:str="") -> int:
    """更新当前聊天记录里面的商品名字段 (通过意图识别获取)"""
    mongo_tool = get_history_mongo_tool()
    try:
        obj_ids = [ObjectId(id) for id in ids]
        result = mongo_tool.chat_message_collection.update_many({"_id": {"$in": obj_ids}},
                                                                {"$set": {"item_names": item_names, "rewritten_query":rewritten_query}})
        return result.modified_count
    except Exception as e:
        logger.error(f"mongo 更新item_name失败")
        return 0


# 查最近的N条历史对话
def mongo_get_recent_message_by_session(session_id: str, limit: int = 10):
    """获取最近的N条历史对"""
    mongo_tool = get_history_mongo_tool()
    cursor = mongo_tool.chat_message_collection.find({"session_id": session_id}).sort("ts", -1).limit(limit)
    return list(cursor)


# 定义自定义 JSON Encoder，解决原生json工具无法序列化ObjectId的问题
class MongoJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, ObjectId):
            return str(obj)
        if isinstance(obj, datetime):
            return obj.isoformat()
        return super().default(obj)


def format_json(data: Any, indent: int = 4, ensure_ascii: bool = False) -> str:
    return json.dumps(data, indent=indent, ensure_ascii=ensure_ascii, cls=MongoJSONEncoder)

if __name__ == '__main__':
    # result=mongo_insert_data("test_001", "user","你好有烫金机吗?")
    # print(result)

    # result=mongo_get_recent_message_by_session("test_001")
    # print(result)

    # result=mongo_insert_data("test_001", "user","你好你好有烫金机吗?", message_id='6a79b03b3425e7d13f24c8ed')
    # print(result)

    # result=mongo_update_item_name_by_id(['6a79b03b3425e7d13f24c8ed'], ["烫金机一号", "烫金机二号"])
    # print(result)

    result = mongo_delete_session("test_001")
    print(result)
