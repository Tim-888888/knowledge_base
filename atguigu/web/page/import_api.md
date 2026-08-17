# 知识库文件导入接口文档

> 面向：Python / FastAPI 后端开发  
> 前端来源：`atguigu/web/page/import.html`  
> 接口状态：前端契约已明确，后端路由尚未实现

## 1. 接口范围

当前导入页面只依赖两个接口：

| 接口 | 方法 | 用途 |
|---|---|---|
| `/upload` | `POST` | 上传单个 PDF 或 Markdown 文件，并创建异步导入任务 |
| `/status/{task_id}` | `GET` | 轮询导入任务状态、节点进度和节点耗时 |

前端当前使用的服务地址：

```text
http://127.0.0.1:8000
```

页面的完整调用顺序：

```text
选择或拖入文件
    -> POST /upload
    -> 获得 task_id
    -> 每 1.5 秒调用 GET /status/{task_id}
    -> status=completed 时停止轮询并显示 100%
    -> status=failed 时停止轮询并显示失败
```

页面允许一次选择多个文件，但会为每个文件分别调用一次 `/upload`，因此后端接口仍按“单文件上传”设计。

---

## 2. 上传文件

### 2.1 基本信息

```http
POST /upload
Content-Type: multipart/form-data
```

该接口应在完成文件接收、基础校验和任务创建后尽快返回，不等待整个知识库导入工作流执行完成。

建议成功状态码使用 `202 Accepted`；现有页面通过 `response.ok` 判断结果，因此任何 `2xx` 状态码都可以被页面接受。

### 2.2 请求参数

请求体类型：`multipart/form-data`

| 字段 | 类型 | 必填 | 页面要求 | 说明 |
|---|---|---:|---|---|
| `file` | binary | 是 | 必须 | 单个待导入文件，字段名必须是 `file` |

当前页面允许的文件扩展名：

```text
.pdf
.md
```

后端不能只依赖前端的 `accept` 属性，仍需自行校验文件后缀、文件内容和文件大小。最大文件大小当前页面没有定义，需要后端配置后再补充到接口契约中。

### 2.3 请求示例

```bash
curl -X POST "http://127.0.0.1:8000/upload" \
  -H "Accept: application/json" \
  -F "file=@./manual.pdf"
```

### 2.4 成功响应

```http
HTTP/1.1 202 Accepted
Content-Type: application/json
```

```json
{
  "task_id": "5d72a4cb-8384-4df0-90b8-b3fd4ac76f0e"
}
```

| 字段 | 类型 | 必填 | 页面使用方式 |
|---|---|---:|---|
| `task_id` | string | 是 | 拼接到 `/status/{task_id}` 并开始轮询 |

`task_id` 建议使用 UUID 字符串，并保证在任务状态有效期内全局唯一。

### 2.5 建议错误响应

错误响应统一使用 FastAPI 常见结构：

```json
{
  "detail": "错误原因"
}
```

| HTTP 状态码 | 场景 |
|---:|---|
| `400` | 文件名或请求内容不合法 |
| `413` | 文件超过后端允许的最大大小 |
| `415` | 文件类型不是 `.pdf` 或 `.md` |
| `422` | 缺少 `file` 字段或表单结构错误 |
| `500` | 文件保存或任务创建失败 |

上传接口返回非 `2xx` 时，页面会显示“上传失败”，并且不会开始状态轮询。

---

## 3. 查询导入任务状态

### 3.1 基本信息

```http
GET /status/{task_id}
Accept: application/json
```

页面每 `1500ms` 调用一次该接口。

### 3.2 路径参数

| 参数 | 类型 | 必填 | 说明 |
|---|---|---:|---|
| `task_id` | string | 是 | `/upload` 返回的任务唯一标识 |

请求示例：

```bash
curl "http://127.0.0.1:8000/status/5d72a4cb-8384-4df0-90b8-b3fd4ac76f0e"
```

### 3.3 处理中响应

```json
{
  "task_id": "5d72a4cb-8384-4df0-90b8-b3fd4ac76f0e",
  "status": "processing",
  "done_list": [
    "upload_file",
    "检查文件"
  ],
  "running_list": [
    "PDF转Markdown"
  ],
  "durations": {
    "检查文件": 0.12
  },
  "error": null
}
```

### 3.4 完成响应

```json
{
  "task_id": "5d72a4cb-8384-4df0-90b8-b3fd4ac76f0e",
  "status": "completed",
  "done_list": [
    "upload_file",
    "检查文件",
    "PDF转Markdown",
    "Markdown图片处理",
    "文档切分",
    "主体名称识别",
    "向量生成",
    "导入向量库"
  ],
  "running_list": [],
  "durations": {
    "检查文件": 0.12,
    "PDF转Markdown": 8.46,
    "Markdown图片处理": 3.21,
    "文档切分": 0.18,
    "主体名称识别": 1.35,
    "向量生成": 2.76,
    "导入向量库": 0.91
  },
  "error": null
}
```

### 3.5 失败响应

任务已经创建但工作流执行失败时，接口仍建议返回 `200 OK`，通过业务状态 `failed` 通知页面停止轮询：

```json
{
  "task_id": "5d72a4cb-8384-4df0-90b8-b3fd4ac76f0e",
  "status": "failed",
  "done_list": [
    "upload_file",
    "检查文件"
  ],
  "running_list": [],
  "durations": {
    "检查文件": 0.12
  },
  "error": "PDF解析服务调用失败"
}
```

当前页面只根据 `status=failed` 显示通用失败提示，尚未展示 `error` 字段；保留该字段便于后续增强页面和排查问题。

### 3.6 响应字段

| 字段 | 类型 | 必填 | 页面要求与语义 |
|---|---|---:|---|
| `task_id` | string | 建议 | 原样返回任务ID，便于调试；当前页面不读取 |
| `status` | string | 是 | 只能使用 `processing`、`completed`、`failed` |
| `done_list` | string[] | 是 | 已完成步骤，参与进度计算并逐项显示日志 |
| `running_list` | string[] | 是 | 当前运行步骤；非空时页面额外增加半步进度 |
| `durations` | object | 是 | 键为步骤名称，值为该步骤耗时秒数 |
| `error` | string/null | 建议 | 任务失败原因；当前页面暂不展示 |

`done_list`、`running_list`、`durations` 必须始终返回正确类型。没有数据时返回空数组或空对象，不要返回 `null`。

### 3.7 未找到任务

建议返回：

```http
HTTP/1.1 404 Not Found
Content-Type: application/json
```

```json
{
  "detail": "任务不存在或状态已过期"
}
```

注意：当前页面遇到状态接口非 `2xx` 时只会在控制台输出错误，定时器仍会继续轮询。后续前端应补充连续失败次数或 `404` 后停止轮询的逻辑。

---

## 4. 工作流节点与进度约定

页面根据文件后缀写死了总步骤数：

| 文件类型 | 总步骤数 | 原因 |
|---|---:|---|
| PDF | 8 | 上传步骤 + 7 个 LangGraph 节点 |
| Markdown | 7 | 上传步骤 + 6 个 LangGraph 节点，跳过 PDF 转 Markdown |

PDF 的步骤顺序：

```text
upload_file
node_entry
node_pdf_to_md
node_md_img
node_document_split
node_item_name_recognition
node_bge_embedding
node_import_milvus
```

Markdown 的步骤顺序：

```text
upload_file
node_entry
node_md_img
node_document_split
node_item_name_recognition
node_bge_embedding
node_import_milvus
```

进度计算方式来自当前页面：

```text
基础进度 = done_list.length / totalNodes * 100
存在运行中节点时，再增加 0.5 / totalNodes * 100
任务未完成时，最高显示 95%
status=completed 时直接显示 100%
```

### 4.1 节点名称契约注意事项

当前页面会对 `done_list` 中值为 `upload_file` 的项跳过日志渲染，但仍利用它参与进度计算。因此，后端若要完全兼容当前页面：

1. 上传完成后，应把原始字符串 `upload_file` 放进 `done_list`。
2. `durations` 的键应与页面最终收到的步骤名称完全一致。
3. 同一个步骤不能同时出现在 `done_list` 和 `running_list` 中。
4. 步骤完成后，应先从 `running_list` 移除，再加入 `done_list`。

当前 `task_utils.py` 会把 `upload_file` 转换成中文“开始上传文件”，与页面的 `item === 'upload_file'` 判断不一致。后端实现前必须统一这一契约，否则上传日志会重复显示，并可能影响维护者对步骤数量的判断。

---

## 5. FastAPI 数据模型建议

下面仅定义接口数据契约，不包含文件存储和工作流业务实现：

```python
from enum import Enum
from typing import Annotated
from uuid import UUID

from fastapi import File, UploadFile
from pydantic import BaseModel, Field


class TaskStatus(str, Enum):
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class UploadResponse(BaseModel):
    task_id: UUID


class TaskStatusResponse(BaseModel):
    task_id: UUID
    status: TaskStatus
    done_list: list[str] = Field(default_factory=list)
    running_list: list[str] = Field(default_factory=list)
    durations: dict[str, float] = Field(default_factory=dict)
    error: str | None = None


UploadFileParam = Annotated[
    UploadFile,
    File(description="待导入的 PDF 或 Markdown 文件"),
]
```

建议的路由签名：

```python
@app.post("/upload", response_model=UploadResponse, status_code=202)
async def upload_file(file: UploadFileParam) -> UploadResponse:
    ...


@app.get("/status/{task_id}", response_model=TaskStatusResponse)
async def get_task_status(task_id: UUID) -> TaskStatusResponse:
    ...
```

---

## 6. CORS 配置

页面与 FastAPI 服务通常运行在不同端口，后端需要配置 CORS。开发环境可以临时允许全部来源：

```python
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

生产环境应把 `allow_origins` 改成真实前端地址，不应长期使用通配符。

---

## 7. 后端实现要求

1. 接收到文件后先完成基础校验，再生成 `task_id`。
2. 文件名必须进行安全处理，不能直接信任客户端文件名或客户端路径。
3. 在返回 `/upload` 响应前，至少应保证任务状态已经初始化为 `processing`，避免页面立刻轮询时得到 `404`。
4. 后台工作流发生异常时，必须把状态更新为 `failed`，清空 `running_list`，并记录可诊断的错误信息。
5. 工作流正常结束后，必须把状态更新为 `completed`，并清空 `running_list`。
6. `/status/{task_id}` 不应执行耗时业务，只负责读取任务状态并快速响应。
7. 文件上传、任务状态更新和状态查询之间必须使用同一个 `task_id`。

---

## 8. 当前代码实现差距

截至本文档生成时，项目当前状态如下：

1. `atguigu/web/api/import_service/import_service.py` 只创建了 FastAPI 应用并配置 CORS，尚未实现 `/upload` 和 `/status/{task_id}`。
2. `atguigu/tool/task_utils.py` 已经提供 `processing/completed/failed`、运行节点、完成节点和耗时的数据结构，但它是单进程内存状态。
3. 当前 LangGraph 节点基类尚未调用 `add_running_task()`、`add_done_task()` 和 `add_node_duration()`，状态追踪还没有真正接入工作流。
4. 单进程内存状态在进程重启后会丢失，也不能直接支持多个 Uvicorn Worker；当前接口文档不指定新的持久化方案，但后端实现时必须明确运行边界。
5. `upload_file` 的英文节点ID与中文展示名称存在前后端不一致，需要在实现接口前确定统一规则。

---

## 9. 联调验收清单

- [ ] 上传 `.pdf` 能返回非空 `task_id`
- [ ] 上传 `.md` 能返回非空 `task_id`
- [ ] 不支持的文件类型返回 `415`
- [ ] 缺少 `file` 字段返回 `422`
- [ ] 上传成功后立即查询状态不会返回 `404`
- [ ] `processing` 响应包含四个页面必需字段
- [ ] PDF 的进度按 8 步计算
- [ ] Markdown 的进度按 7 步计算
- [ ] 节点完成后从 `running_list` 移入 `done_list`
- [ ] `durations` 的键能对应页面展示的节点名称
- [ ] `completed` 会让页面停止轮询并显示 100%
- [ ] `failed` 会让页面停止轮询并显示失败
- [ ] 多文件上传时，每个文件拥有独立的 `task_id` 和任务状态
- [ ] 浏览器跨域请求不会被 CORS 拦截
