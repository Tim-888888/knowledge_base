'''
@Author  :61022
@Time    :2026/7/31
@Desc    :
'''
import os
from pathlib import Path

from dotenv import load_dotenv

from atguigu.import_process.base import NodeBase
from atguigu.import_process.state import ImportGraphState
import requests

load_dotenv(override=True)

def upload_pdf(pdf_path_obj: Path):
    """上传pdf到MinerU"""
    token = os.getenv("MINERU_API_TOKEN")
    url = f"{os.getenv('MINERU_BASE_URL')}/file-urls/batch"
    header = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {token}"
    }
    data = {
        "files": [
            {"name": "demo.pdf", "data_id": "abcd"}
        ],
        "model_version": "vlm"
    }
    file_path = ["demo.pdf"]
    try:
        response = requests.post(url, headers=header, json=data)
        if response.status_code == 200:
            result = response.json()
            print('response success. result:{}'.format(result))
            if result["code"] == 0:
                batch_id = result["data"]["batch_id"]
                urls = result["data"]["file_urls"]
                print('batch_id:{},urls:{}'.format(batch_id, urls))
                for i in range(0, len(urls)):
                    with open(file_path[i], 'rb') as f:
                        res_upload = requests.put(urls[i], data=f)
                        if res_upload.status_code == 200:
                            print(f"{urls[i]} upload success")
                        else:
                            print(f"{urls[i]} upload failed")
            else:
                print('apply upload url failed,reason:{}'.format(result["msg"]))
        else:
            print('response not success. status:{} ,result:{}'.format(response.status_code, response))
    except Exception as err:
        print(err)


def upload_and_get_url(pdf_path_obj: Path):
    url = upload_pdf(pdf_path_obj)

    return ""


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
        # 获取节点输入 + 校验
        pdf_path=state.get("pdf_path")
        pdf_path_obj = Path(pdf_path)
        if not pdf_path_obj.exists():
            raise RuntimeError(f"pdf文件不存在, 请检查路径: {pdf_path}")

        url = upload_and_get_url(pdf_path_obj)

        return state