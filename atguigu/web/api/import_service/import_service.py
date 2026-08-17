'''
@Author  :61022
@Time    :2026/8/17
@Desc    :
'''
import shutil
import time
import uuid
from datetime import datetime

from pathlib import Path

import fastapi
import uvicorn
from fastapi import FastAPI, UploadFile, BackgroundTasks
from fastapi.params import File
from starlette.middleware.cors import CORSMiddleware

from atguigu.config.config import KBImportConfig
from atguigu.import_process.main_graph import KBImportWorkflow
from atguigu.tool.minio_utils import get_minio_client
from atguigu.tool.task_utils import *

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


def run_main_graph(task_id: str, file_path: str):
    init_state = {
        "local_dir": KBImportConfig.LOCAL_DIR,
        "local_file_path": file_path,
        "task_id": task_id
    }
    try:
        update_task_status(task_id, TASK_STATUS_PROCESSING)
        KBImportWorkflow.create_and_run(init_state, False)
        update_task_status(task_id, TASK_STATUS_COMPLETED)
    except Exception as e:
        update_task_status(task_id, TASK_STATUS_FAILED)
        raise


@app.post("/upload")
async def upload(background_task: BackgroundTasks, file: UploadFile = File(..., description="上传文件")):
    # task_id uuid4
    task_id = str(uuid.uuid4())

    add_running_task(task_id, "upload_file")
    start_time = time.time()
    # 保存文件在本地
    file_dir = Path(rf"E:\upload\{datetime.now().strftime('%Y%m%d')}") / task_id
    if not file_dir.exists():
        file_dir.mkdir(parents=True, exist_ok=True)
    file_path = str(file_dir / file.filename)
    with open(file_path, mode="wb") as f:
        shutil.copyfileobj(file.file, f, 1024 * 1024)

    # 转储到minio upload/时间/task_id/filename
    minio_client = get_minio_client()
    minio_client.fput_object(
        bucket_name=KBImportConfig.MINIO_BUCKET_NAME,
        object_name=f"upload-pdf/{datetime.now().strftime('%Y%m%d')}/{task_id}/{file.filename}",
        file_path=file_path
    )

    add_done_task(task_id, "upload_file")
    add_node_duration(task_id, "upload_file", time.time() - start_time)

    # background_task调用后台任务执行入库主图
    background_task.add_task(run_main_graph, task_id, file_path)

    return {"task_id": task_id}


@app.get("/status/{task_id}")
async def upload(task_id: str = fastapi.Path(..., description="任务id")):
    return get_task_info(task_id)


if __name__ == '__main__':
    uvicorn.run(app, host="0.0.0.0", port=8000)
