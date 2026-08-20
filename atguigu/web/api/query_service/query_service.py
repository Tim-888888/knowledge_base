'''
@Author  :61022
@Time    :2026/8/17
@Desc    :
'''
import json
import uuid
from pathlib import Path

import fastapi
import time
import uvicorn
from fastapi import FastAPI
from fastapi.params import Body
from pydantic import BaseModel, Field
from starlette.background import BackgroundTasks
from starlette.middleware.cors import CORSMiddleware
from starlette.responses import FileResponse, StreamingResponse

from atguigu.query_process.main_graph import KBQueryWorkflow
from atguigu.tool.mongo_history_utils import mongo_get_recent_message_by_session, mongo_delete_session
from atguigu.tool.task_utils import update_task_status, TASK_STATUS_PROCESSING, put_queue_data, get_task_info, \
    TASK_STATUS_COMPLETED, TASK_STATUS_FAILED, create_queue, get_queue_data

# 1. 创建应用
app = FastAPI(
    title="掌柜智库-查询API",
    description="此文档是掌柜智库查询流程的API接口说明"
)

# 2. 跨域
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 允许的源
    allow_credentials=True,  # 允许携带cookie
    allow_methods=["*"],  # 允许的请求方法
    allow_headers=["*"],  # 允许的请求头
)


# 3. 静态页面路由
# @app.get("/chat.html")  # 对外访问地址
# async def chat():
#     # 拼接HTML文件绝对路径
#     html_path = Path(__file__).absolute().parent.parent / "page" / "chat.html"
#
#     return FileResponse(html_path)

@app.get("/health")
async def health():
    # 健康检测
    return {"health": "ok"}


@app.get("/history/{session_id}")
async def history(session_id: str = fastapi.Path(..., description="session_id")):
    print(session_id)
    history_list = reversed(mongo_get_recent_message_by_session(session_id, 100))

    history_list = [{
        "_id": str(history.get("_id", "")),
        "session_id": history.get("session_id", ""),
        "role": history.get("role", ""),
        "text": history.get("text", ""),
        "rewritten_query": history.get("rewritten_query", ""),
        "item_names": history.get("item_names", ""),
    } for history in history_list]

    return {"items": history_list}


@app.delete("/history/{session_id}")
async def delete_history(session_id: str = fastapi.Path(..., description="session_id")):
    delete_num = mongo_delete_session(session_id)
    return {"delete_num": delete_num}


class QueryParam(BaseModel):
    query: str = Field(..., description="用户问题")
    session_id: str = Field(..., description="会话ID")


def run_query_graph(original_query: str, session_id: str, task_id: str):
    try:
        update_task_status(task_id, TASK_STATUS_PROCESSING)
        put_queue_data(task_id, "progress", get_task_info(task_id))

        init_state = {
            "session_id": session_id,
            "task_id": task_id,
            "original_query": original_query
        }

        KBQueryWorkflow.create_and_run(init_state)

        update_task_status(task_id, TASK_STATUS_COMPLETED)
        put_queue_data(task_id, "progress", get_task_info(task_id))
    except Exception as e:
        update_task_status(task_id, TASK_STATUS_FAILED)
        put_queue_data(task_id, "error", get_task_info(task_id))
        raise


@app.post("/query")
async def query(background_tasks: BackgroundTasks, query_param: QueryParam = Body(..., description="查询参数")):
    """运行检索逻辑"""
    task_id = str(uuid.uuid4())
    create_queue(task_id)
    # 启动后台任务, 调用查询主图
    background_tasks.add_task(run_query_graph, query_param.query, query_param.session_id, task_id)

    return {"task_id": task_id}  # 用于启动SSE, 获取对话结果


def get_stream_data(task_id: str):
    queue = get_queue_data(task_id)
    while True:
        event_data = queue.get()
        yield f"event: {event_data.get('event')}\n"
        yield f"data: {json.dumps(event_data.get('data'))}\n\n"
        # time.sleep(0.1)


@app.get("/stream/{task_id}", description="获取查询任务状态")
async def get_stream_result(task_id: str = fastapi.Path(..., description="task_id")):
    return StreamingResponse(get_stream_data(task_id), media_type="text/event-stream")


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8001)
