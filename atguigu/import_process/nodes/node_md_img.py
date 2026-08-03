'''
@Author  :61022
@Time    :2026/7/31
@Desc    :
'''
import base64
import re
import time
from collections import deque
from pathlib import Path
from typing import List, Tuple

from langchain.chat_models import init_chat_model
from minio import Minio
from minio.deleteobjects import DeleteObject

from atguigu.config.config import KBImportConfig
from atguigu.import_process.base import NodeBase
from atguigu.import_process.state import ImportGraphState
from atguigu.tool.json_format_util import parse_json
from atguigu.tool.logger import logger
from atguigu.tool.minio_utils import get_minio_client


class NodeMDImg(NodeBase):
    """
    MarkDown图片处理节点：多模态图片理解, 利用图片+前后文 生成图片摘要
    md里面图片的格式 ![](images/f3349cded08d6686a93d0a81b9a64ec1e50d9a82cbb88541b37027f085813a15.jpg)
    """

    @property
    def name(self) -> str:
        return "node_md_img"

    def process(self, state: ImportGraphState):
        # 参数校验
        md_path, md_content, file_title = self.parameter_validation(state)

        # 获取有效的md图片, 过滤不存在md里的图片, 获取图片的前后文
        list_image = self.get_md_images(md_path, md_content, file_title)

        if not list_image:
            logger.info("未检测到md文件使用了图片")
            return {"md_content": md_content}

        # 设置限流窗口机制, 请求大模型, 获取图片摘要
        images_summary = self.get_images_summary(list_image)

        # 图片写入MinIO, 获取url地址
        self.wirte_images_to_minio(file_title, images_summary)

        # 图片摘要和url地址回写到md文档中图片对应位置
        md_content = self.write_md_content(md_content, images_summary)

        return {"md_content": md_content}

    def parameter_validation(self, state: ImportGraphState) -> Path:
        logger.info("参数校验开始")
        md_path = state.get("md_path")
        if not md_path:
            raise RuntimeError(f"md_path 必须提供")

        file_title = state.get("file_title")
        if not file_title:
            raise RuntimeError(f"file_title 必须提供")

        # 获取节点输入 + 校验
        md_path_obj = Path(md_path)
        if not md_path_obj.exists():
            raise RuntimeError(f"md文件不存在, 请检查路径: {md_path}")

        with open(md_path, "r", encoding="utf-8") as f:
            md_content = f.read()
        logger.info("参数校验完成")
        return md_path, md_content, file_title

    def get_md_images(
            self,
            md_path: str,
            md_content: str,
            file_title: str,
            context_len=100
    ) -> List[Tuple[str, str, Tuple[str, str]]]:
        """提取Markdown图片，并获取忽略其他图片后的前后文。"""
        logger.info("获取md图片+上下文开始")
        md_path_obj = Path(md_path)

        image_extensions = {
            ".jpg", ".jpeg", ".png",
            ".gif", ".bmp", ".webp"
        }

        # 匹配Markdown图片，并分别捕获alt和图片地址
        image_pattern = re.compile(
            r'!\[(?P<alt>[^\]]*)\]\((?P<path>[^)]+)\)',
            re.IGNORECASE
        )

        # 按照图片在Markdown中的顺序获取
        matches = list(image_pattern.finditer(md_content))

        # 将所有图片语法替换成等长空格
        # 等长替换可以保证原来的下标不发生变化
        masked_chars = list(md_content)

        for match in matches:
            start, end = match.span()
            masked_chars[start:end] = " " * (end - start)

        masked_content = "".join(masked_chars)

        valid_images = []

        for match in matches:
            image_relative_path = match.group("path").strip()
            image_path_obj = md_path_obj.parent / image_relative_path
            logger.warning(str(image_path_obj))
            # 过滤不支持的格式和不存在的图片
            if (
                    image_path_obj.suffix.lower() not in image_extensions
                    or not image_path_obj.is_file()
            ):
                logger.warning(f"图片[{str(image_path_obj)}], 在md文件中不存在或格式不支持, 不进行处理")
                continue

            # 图片在原始Markdown中的完整位置
            start_index, end_index = match.span()

            # 因为图片标签已经被替换为空格，所以能够越过连续图片
            pre_content = masked_content[:start_index]
            post_content = masked_content[end_index:]

            # 合并换行和多余空格
            pre_content = re.sub(r"\s+", " ", pre_content).strip()
            post_content = re.sub(r"\s+", " ", post_content).strip()

            # 取最近的前后文
            pre_content = pre_content[-context_len:]
            post_content = post_content[:context_len]

            valid_images.append((
                file_title,
                str(image_path_obj),
                (pre_content, post_content)
            ))
        logger.info("获取md图片+上下文完成")
        return valid_images

    def get_images_summary(self, list_image: List[Tuple[str, str, Tuple[str, str]]], window_time=60, window_size=100) -> \
            List[dict]:
        """设置限流窗口机制, 请求大模型, 获取图片摘要"""
        logger.info("获取md图片摘要开始")
        images_summary = []
        window = deque()

        # 创建聊天完成请求
        llm = init_chat_model(
            model_provider="openai",
            model=KBImportConfig.VL_MODEL,
            api_key=KBImportConfig.OPENAI_API_KEY,
            base_url=KBImportConfig.OPENAI_API_BASE,
            temperature=KBImportConfig.LLM_DEFAULT_TEMPERATURE,
            extra_body={"enable_thinking": False}
        )

        for file_title, image_path, context in list_image:
            try:
                # 定义限流窗口：60秒内最多请求100次
                while True:
                    now = time.monotonic()  # 返回一个只会单调递增的计时值, 一般用来算时间差值

                    # 删除已经离开时间窗口的请求记录 第一次循环是空窗口, 要跳出循环
                    while window and now - window[0] >= window_time:
                        window.popleft()

                    # 当前窗口不满100次, 有空位，可以发送请求
                    if len(window) < window_size:
                        break

                    # [拦截等待] 当队列满100个请求, 开始等待最早的一次请求离开时间窗口, (now - window[0])表示第一个元素已经过去了多长时间
                    wait_time = window_time - (now - window[0])
                    time.sleep(max(wait_time, 0))

                # 有空位, 记录本次请求开始时间, 开始发新的一个请求
                window.append(time.monotonic())

                # 把图片内容转base64编码
                with open(image_path, "rb") as f:
                    f_bytes = f.read()
                    image_base64_str = base64.b64encode(f_bytes).decode("utf-8")
                # 请求大模型
                messages = [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": f"""这是"{file_title}"文件中的一张图片，图片上文部分为"{context[0]}"，下文部分为"{context[1]}"，请用中文简要总结这张图片的内容，用于 Markdown 图片标题。不要让我选择, 直接给答案"""
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{image_base64_str}"
                                }
                            }
                        ]
                    }
                ]
                content = llm.invoke(messages).content

                logger.info(f"图片[{image_path}]已处理, 图片摘要[{content}]")
                # 组装结果
                images_summary.append({
                    "image_path": image_path,
                    "image_name": Path(image_path).name,
                    "image_summary": content,
                })
            except Exception as e:
                logger.error(f"发生未知异常, 图片摘要赋予默认值. {e}")
                images_summary.append({
                    "image_path": image_path,
                    "image_name": Path(image_path).name,
                    "image_summary": "图片摘要",
                })

        logger.info("获取所有md图片摘要完成")
        return images_summary

    def wirte_images_to_minio(self, file_title: str, images_summary: List[dict]):
        # 构造MinIO存放图片的路径 bucket/file_title/图片
        upload_path = f"{Path(KBImportConfig.MINIO_IMG_DIR)}/{file_title}".replace(" ", "")

        client = get_minio_client()
        # 先删除upload_path整个路径(幂等操作)
        self.delete_minio_objects(client, upload_path)

        # 写入md对应的所有图片 获取url
        self.upload_images(client, images_summary, upload_path)

    def delete_minio_objects(self, client: Minio | None, upload_path: str):
        try:
            list_objects = client.list_objects(
                bucket_name=KBImportConfig.MINIO_BUCKET_NAME,
                prefix=upload_path,
                recursive=True
            )
            delete_obj_list = [DeleteObject(obj.object_name) for obj in list_objects]
            if delete_obj_list:
                errors = client.remove_objects(
                    bucket_name=KBImportConfig.MINIO_BUCKET_NAME,
                    delete_object_list=delete_obj_list
                )
                for error in errors:
                    logger.error(f"MinIO 删除失败, {error}")
        except Exception as e:
            logger.error(f"MinIO 删除失败, {e}")

    def upload_images(self, client: Minio, images_summary: List[dict], upload_path: str):
        for summary_dict in images_summary:
            image_path = summary_dict['image_path']
            image_name = summary_dict['image_name']
            object_name = f"{upload_path}/{image_name}"
            result = client.fput_object(
                KBImportConfig.MINIO_BUCKET_NAME, object_name, image_path
            )
            url = f"http://{KBImportConfig.MINIO_ENDPOINT}/{KBImportConfig.MINIO_BUCKET_NAME}/{result.object_name}"
            summary_dict['image_url'] = url

    def write_md_content(self, md_content: str, images_summary: List[dict]) -> str:
        """正则替换md图片摘要和MinIO的url"""
        for summary_dict in images_summary:
            image_name = summary_dict['image_name']
            image_summary = summary_dict['image_summary']
            image_url = summary_dict['image_url']

            pattern = re.compile(
                r"!\[.*?\]\(.*?" + re.escape(image_name) + r"\)",
                re.IGNORECASE
            )
            md_content = pattern.sub(lambda m: f"![{image_summary}]({image_url})", md_content)
        return md_content


if __name__ == '__main__':
    node = NodeMDImg()
    init_state = {
        "pdf_path": "E:\\output\\hak180产品安全手册.pdf",
        "local_dir": "E:\\output",
        "md_path": "E:\\output\\hak180产品安全手册\\hak180产品安全手册.md",
        "file_title": "hak180产品安全手册"
    }

    print(parse_json(node(init_state)))
