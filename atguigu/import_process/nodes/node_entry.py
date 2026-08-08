'''
@Author  :61022
@Time    :2026/7/31
@Desc    :
'''
import hashlib
from pathlib import Path

from atguigu.import_process.base import NodeBase
from atguigu.import_process.state import ImportGraphState
from atguigu.tool.json_format_util import parse_json
from atguigu.tool.logger import logger


class NodeEntry(NodeBase):
    """
    入口节点：任务分发
        根据文件后缀决定处理方式
        不兼容的文件要拦截
        输入: local_file_path
        输出: is_pdf_read_enabled, file_title, pdf_path/md_path
    """
    @property
    def name(self) -> str:
        return "node_entry"

    def process(self, state: ImportGraphState):
        # 获取单个输入文件路径
        local_file_path = state.get("local_file_path")

        # 拦截逻辑
        if not local_file_path:
            return RuntimeError(f"state local_file_path 必须提供")
        file_path = Path(local_file_path)

        # 拦截逻辑
        if not file_path.exists():
            raise Exception(f"文件不存在：{file_path}")

        file_suffix = file_path.suffix.lower()
        file_title = file_path.stem

        # 原始文档内容唯一标识
        sha256=hashlib.sha256()
        with open(file_path, 'rb') as f:
            for block in iter(lambda : f.read(1024*1024), b""):
                sha256.update(block)
        file_content_sha256 = sha256.hexdigest()

        # 根据文件后缀决定处理方式
        if file_suffix == ".pdf":
            state['is_pdf_read_enabled'] = True
            state['file_title'] = file_title
            state['pdf_path'] = str(file_path)
            state['file_content_sha256'] = str(file_content_sha256)
            return state
        elif file_suffix == ".md":
            state['is_md_read_enabled'] = True
            state['file_title'] = file_title
            state['md_path'] = str(file_path)
            state['file_content_sha256'] = str(file_content_sha256)
            return state
        else:
            raise Exception(f"不支持的文件类型: [{file_suffix}]")



if __name__ == '__main__':
    # 测试
    node = NodeEntry()
    state=node({"local_file_path": r"E:\output\hak180产品安全手册.pdf"})
    # state=node({"local_file_path": r"E:\大模型学习\其他班文件\0331班\尚硅谷大模型项目之掌柜智库\2.资料\04-设备手册汇总\doc\Aolynk CB304n Cable网桥 用户手册-5W100-整本手册.pdf"})
    logger.info(parse_json(state))
