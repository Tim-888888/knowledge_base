'''
@Author  :61022
@Time    :2026/7/31
@Desc    :
'''
import os
import shutil
import time
from pathlib import Path
from zipfile import ZipFile

from dotenv import load_dotenv
from dotenv.main import with_warn_for_invalid_lines
from requests import HTTPError
from urllib3.util import url

from atguigu.config.config import KBImportConfig
from atguigu.import_process.base import NodeBase
from atguigu.import_process.state import ImportGraphState
import requests

from atguigu.tool.logger import logger

load_dotenv(override=True)


class NodePDFToMD(NodeBase):
    """
    PDF 转 Markdown 节点：PDF结构化解析, 使用MinerU

    功能:
        把文件上传到MinerU, 轮询获取下载url, 下载, 解压, 定义输出路径, 写md_content, 写md_path
    输入:
        file_title
        pdf_path
    输出:
        md_content
        md_path
    思路:
        1.把文件上传到MinerU + 轮询获取下载url(API参照官网的去改, 加入各种校验, 合理报错)
        2.按照url下载 + 解压
        3.写md_content + md_path
    """

    @property
    def name(self) -> str:
        return "node_pdf_to_md"

    def process(self, state: ImportGraphState):
        # 参数校验
        pdf_path_obj, local_dir_obj = self.parameter_validation(state)

        # 上传pdf到MinerU服务器, 获取下载url
        url = self.upload_and_get_url(pdf_path_obj)
        print(url)

        # 下载pdf + 解压pdf + 文件改名
        new_path, md_content = self.download_and_unzip(url, pdf_path_obj, local_dir_obj)

        # 组装state返回结果
        state["md_path"] = new_path
        state["md_content"] = md_content

        return state

    def parameter_validation(self, state: ImportGraphState) -> Path:
        pdf_path = state.get("pdf_path")
        if not pdf_path:
            raise RuntimeError(f"pdf_path 必须提供")

        # 获取节点输入 + 校验
        pdf_path_obj = Path(pdf_path)
        if not pdf_path_obj.exists():
            raise RuntimeError(f"pdf文件不存在, 请检查路径: {pdf_path}")

        local_dir = state.get("local_dir", KBImportConfig.LOCAL_DIR)
        local_dir_obj = Path(local_dir)
        if not local_dir_obj.exists():
            local_dir_obj.mkdir(parents=True, exist_ok=True)
        return pdf_path_obj, local_dir_obj

    def upload_pdf(self, pdf_path_obj: Path):
        """上传pdf到MinerU"""
        file_name = pdf_path_obj.name

        token = KBImportConfig.MINERU_API_TOKEN
        url = f"{KBImportConfig.MINERU_BASE_URL}/file-urls/batch"
        header = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {token}"
        }
        data = {
            "files": [
                {"name": file_name}
            ],
            "model_version": "vlm"
        }
        file_path = str(pdf_path_obj)

        response = requests.post(url, headers=header, json=data)
        if response.status_code != 200:
            raise RuntimeError('response not success. status:{} ,result:{}'.format(response.status_code, response))
        result = response.json()
        """
        响应体示例
        {
          "code": 0,
          "data": {
            "batch_id": "2bb2f0ec-a336-4a0a-b61a-241afaf9cc87",  #批量提取任务 id，可用于批量查询解析结果
            "file_urls": ["https://***"]                         #文件上传链接
          },
          "msg": "ok",
          "trace_id": "c876cd60b202f2396de1f9e39a1b0172"
        }
        """
        logger.info('响应成功. result:{}'.format(result))
        if result["code"] != 0:
            raise RuntimeError('请求数据错误,reason:{}'.format(result["msg"]))

        batch_id = result["data"]["batch_id"]
        url = result["data"]["file_urls"][0]
        logger.info('batch_id:{},url:{}'.format(batch_id, url))

        # 再把url上传, 查看上传状态
        with open(file_path, 'rb') as f:
            res_upload = requests.put(url, data=f)
            if res_upload.status_code != 200:
                logger.error(f"{url} 上传失败")
                raise HTTPError(f"{url} 上传失败")
            logger.info(f"{url} 上传成功")

        return batch_id

    def upload_and_get_url(self, pdf_path_obj: Path):
        # 上传pdf到MinerU
        batch_id = self.upload_pdf(pdf_path_obj)

        # 轮询获取下载url
        url = self.polling_get_url(batch_id)

        return url

    def polling_get_url(self, batch_id):
        token = KBImportConfig.MINERU_API_TOKEN

        url = f"{KBImportConfig.MINERU_BASE_URL}/extract-results/batch/{batch_id}"
        header = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {token}"
        }

        # 设定超时时间, 结束循环
        start_time = time.time()
        timeout_time = 300

        while True:
            # 设定超时时间, 结束循环
            current_time = time.time()
            if current_time - start_time > timeout_time:
                logger.error(f"获取下载url超时, 超时时间:{timeout_time}")
                raise TimeoutError(f"获取下载url超时, 超时时间:{timeout_time}")

            try:
                res = requests.get(url, headers=header)

                """
                响应体实例
                    {
                      "code": 0,
                      "data": {
                        "batch_id": "2bb2f0ec-a336-4a0a-b61a-241afaf9cc87",
                        "extract_result": [
                          {
                            "file_name": "example.pdf",
                            "state": "done",
                            "err_msg": "",
                            "full_zip_url": "https://cdn-mineru.openxlab.org.cn/pdf/018e53ad-d4f1-475d-b380-36bf24db9914.zip"
                          },
                          {
                            "file_name": "demo.pdf",
                            "state": "running",
                            "err_msg": "",
                            "extract_progress": {
                              "extracted_pages": 1,
                              "total_pages": 2,
                              "start_time": "2025-01-20 11:43:20"
                            }
                          }
                        ]
                      },
                      "msg": "ok",
                      "trace_id": "c876cd60b202f2396de1f9e39a1b0172"
                    }
                """

                # 失败重试逻辑
                if res.status_code != 200:
                    raise RuntimeError(f"获取下载url请求失败, status_code:{res.status_code}")

                res_json = res.json()
                if res_json.get("code") != 0:
                    raise RuntimeError(f"获取下载url失败, 错误信息: {res_json['msg']}")

                extract_result = res_json.get("data").get("extract_result")[0]
                if extract_result.get("state") != "done":
                    raise RuntimeError(f"pdf文件解析失败, 错误信息: {extract_result.get('err_msg')} res:{res}")

                full_zip_url = extract_result.get("full_zip_url")
                logger.info(f"获取下载url成功, url:{full_zip_url}")
                return full_zip_url
            except Exception as e:
                logger.error(f"{e}, 正在重试...")
                time.sleep(5)
                continue

    def download_and_unzip(self, url, pdf_path_obj: Path, local_dir_obj: Path):
        # 下载zip文件
        zip_path_res = requests.get(url)

        if zip_path_res.status_code != 200:
            raise RuntimeError(f"下载zip文件失败, status_code:{zip_path_res.status_code}")

        zip_file_content = zip_path_res.content
        zip_name = pdf_path_obj.stem + ".zip"
        zip_file_path = local_dir_obj / zip_name
        with open(zip_file_path, "wb") as f:
            f.write(zip_file_content)

        # 解压zip文件
        zip_file = ZipFile(zip_file_path)
        unzip_dir = local_dir_obj / pdf_path_obj.stem
        # 先删除目录
        if unzip_dir.exists():
            shutil.rmtree(unzip_dir)
        zip_file.extractall(unzip_dir)

        # 把full.md文件改名为pdf_path_obj.stem
        full_md_path = unzip_dir / "full.md"
        new_path = full_md_path.with_name(pdf_path_obj.stem + ".md")
        full_md_path.rename(new_path)

        # 读取md文件内容
        with open(new_path, "r", encoding="utf-8") as f:
            md_content = f.read()

        return str(new_path), md_content


if __name__ == '__main__':
    node = NodePDFToMD()
    init_state = {
        "pdf_path": r"E:\output\Aolynk CB304n Cable网桥 用户手册-5W100-整本手册.pdf"
    }
    node(init_state)
