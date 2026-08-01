'''
@Author  :61022
@Time    :2026/8/1
@Desc    :
'''
import json


def parse_json(state):
    """字典转格式化的json"""
    return json.dumps(state, indent=4, ensure_ascii=False)