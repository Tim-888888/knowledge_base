'''
@Author  :61022
@Time    :2026/7/31
@Desc    :
'''
import json
import re
from pathlib import Path
from typing import List, Any

from langchain_text_splitters import RecursiveCharacterTextSplitter

from atguigu.import_process.base import NodeBase
from atguigu.import_process.state import ImportGraphState
from atguigu.tool.json_format_util import parse_json
from atguigu.tool.logger import logger
from atguigu.tool.markdown_chunker import MarkdownChunker


class NodeDocumentSplit(NodeBase):
    """
    文档切分节点：智能文档切片

		*切chunk的作用其实要结合embedding看
		功能:对markdown文档进行chunk切分, 给后续embedding用, 两个注意点, 一是chunk大小, 二是chunk内语义的完整度
		输入:
			md_content
		输出:
			chunks
		思路:
			切分的几个要点:
				切太粗:Embedding向量表达能力不足, 语义被稀释, 而且超长会被BGE-M3截断, 语义丢失, 检索精度下降
				切太细:丢失完整语义, 检索精度下降
				切合适:保留完整语义, Embedding向量也能够充分表达这个chunk
			粗切关键点: 判断代码块的开始 和 判断标题的开始, 组装section字典列表
			细切关键点: 小于chunk长度不切, 把标题从正文摘除, 切分,再拼标题进去正文, 没有标题单独处理?
		亮点:
			一. markdown采用粗切+细切的方式, 让chunk长度尽量且贴近指定chunk长度, 尽量维持语义完整性
				粗切分(按标题切)(按md结构切分)
					1.按一级二级标题切
					2.代码围栏里面的代码不进行切分, 放一个chunk里面
				精细切分(长切短合)
					长切/短合: 用递归切分 (段落/句子/问号/分号/逗号/空格), 递归切分器内已经做了短合了, 会定义一个最大最小长度的chunk
				上下文语义保留: chunk保留文件名-一二级标题
			二. 采用评分机制
			三. 采用大模型进行切分
			四. chunk大小经验值:
				每个chunk：400～800 tokens
				最小chunk：100～150 tokens
				重叠部分：50～100 tokens
    """

    @property
    def name(self) -> str:
        return "node_document_split"

    def process(self, state: ImportGraphState):
        # 参数初始化
        md_content, md_path, file_title = self.init_param(state)

        # 粗切分: 按md结构切分 -> 按标题和代码围栏做切分
        # md_section_list = self.md_section_split(file_title, md_content)


        # 精细切分: 长切短合, 用递归切分器切分
        # chunks = self.md_fine_split(md_section_list)

        chunks = self.smart_split_chunk(file_title, md_content)

        # 落盘json测试用
        with open(Path(md_path).parent / (Path(md_path).stem + ".json"), mode="w", encoding='utf-8') as f:
            f.write(parse_json(chunks))

        return {"chunks": chunks}

    def smart_split_chunk(self, file_title: str, md_content: str, max_chunk_size = 500, min_chunk_size=200) -> list[dict[str, str | None | Any]]:
        chunker = MarkdownChunker(
            max_chunk_size=max_chunk_size,
            min_chunk_size=min_chunk_size
        )
        chunks = chunker.split_text(
            md_content,
            extra_metadata={
                "file_title": file_title,
            }
        )
        print(len(chunks))

        chunks = [{
            "file_title": chunk.metadata.get("file_title"),
            "nearest_heading_title": chunk.metadata.get("nearest_heading"),
            "nearest_heading_position": chunk.metadata.get("section_chunk_index"),
            "chunk": chunk.page_content,
            # 保留切分器生成的标题归属、section 内序号和原文行号等信息。
            # "metadata": dict(chunk.metadata),
        } for chunk in chunks]
        return chunks

    def init_param(self, state: ImportGraphState):
        md_content = state.get("md_content")
        if not md_content:
            logger.error("md_content 不能为空")
            raise ValueError("md_content 不能为空")

        md_path = state.get("md_path")
        md_path_obj = Path(md_path)
        if not md_path_obj.exists():
            logger.error(f"{md_path} 不存在")
            raise FileNotFoundError(f"{md_path} 不存在")

        with open(md_path_obj, mode="r", encoding="utf-8") as f:
            md_content = f.read()

        file_title = state.get("file_title")
        if not file_title:
            logger.warning("file_title 不能为空")
            file_title = md_path_obj.parent.stem

        return md_content, md_path, file_title

    def md_section_split(self, file_title: str, md_content: str) -> List[dict]:
        # 不同系统的换行符转换
        md_content.replace("\r\n", "\n").replace("\r", "\n")
        # 把content文本按行切分
        content_lines = md_content.split("\n")
        # 初始化参数
        md_section_list: List[dict] = []
        CODE_FENCE_PATTERN = r"^(```{3})|(~~~{3})"
        is_code_fence = False  # True
        code_fence_mark = ""  # ```
        TITLE_PATTERN = r'^\s*#{1,6}\s+.+'
        title_idx = 0

        # 循环content内容
        for idx, line in enumerate(content_lines):
            line = line.strip()
            # 判断line是否为代码围栏, 代码围栏内的#不应该被识别为标题
            code_match = re.match(CODE_FENCE_PATTERN, line)
            if code_match:
                if not is_code_fence:
                    # 代码围栏开启
                    is_code_fence = True
                    code_fence_mark = code_match.group(0)
                    continue
                else:
                    if code_fence_mark == code_match.group(0):
                        # 代码围栏结束
                        is_code_fence = False
                        code_fence_mark = ""
                        continue

            # 判断line是否为标题, 如果是标题, 应当把标题之前的正文和上一个标题, 归为一个chunk字典, 正文需要带上标题
            title_match = re.match(TITLE_PATTERN, line)
            if title_match and not is_code_fence:
                # 这里的逻辑是处理当前line是标题的逻辑
                # 切片获取这个标题之前的所有line
                current_lines = content_lines[title_idx:idx]
                section_content = "\n".join(current_lines).strip()
                # 更新上一个标题的下标
                title_idx = idx
                # 组装 section
                md_section_list.append({
                    "file_title": file_title,
                    "section_title": current_lines[0] if section_content.startswith("#") else "无标题",
                    "section_content": section_content
                })

        # 处理最后一个section
        current_lines = content_lines[title_idx:]
        section_content = "\n".join(current_lines).strip()
        md_section_list.append({
            "file_title": file_title,
            "section_title": current_lines[0] if section_content.startswith("#") else "无标题",
            "section_content": section_content
        })

        return md_section_list

    def md_fine_split(self, md_section_list: List[dict]) -> List[dict]:
        """精细切分: 长切短合, 用递归切分器切分, 切分后记录chunk的位置"""

        # 初始化参数
        chunks: List[dict] = []
        max_chunk_size = 500  # 含标题 + 正文
        overlay = 50
        splitter = RecursiveCharacterTextSplitter(
            separators=["\n\n", "\n", "。", "！", "？", "；", ".", "!", "?", ";", " "],
            chunk_size=max_chunk_size,
            chunk_overlap=overlay,
            length_function=len
        )

        for idx, md_section in enumerate(md_section_list):
            file_title: str = md_section.get("file_title")
            section_title: str = md_section.get("section_title")
            section_content: str = md_section.get("section_content")

            # 正文 (去除标题)
            section_content_without_title = section_content.replace(section_title, "", 1)
            # section_content_without_title_len = len(section_content) - len(section_title)

            # 拦截不切分的情况
            # 如果section_content长度小于max_chunk_size, 不切分
            if len(section_content) <= max_chunk_size:
                chunks.append({
                    "file_title": file_title,
                    "section_title": section_title,
                    "chunk": section_content,
                    "position": 0
                })
                continue

            # 如果section_content包含html表格数据, 不切分
            if "<table" in section_content_without_title:
                chunks.append({
                    "file_title": file_title,
                    "section_title": section_title,
                    "chunk": section_content,
                    "position": 0
                })
                continue

            # 超长标题, 单独成块
            # if len(section_title) >= max_chunk_size:
            #     chunks.append({
            #         "file_title": file_title,
            #         "section_title": section_title,
            #         "chunk": section_title,
            #         "position": 0
            #     })

            # 切分逻辑
            chunk_list = splitter.split_text(section_content_without_title)
            chunks.extend(
                [{
                    "file_title": file_title,
                    "section_title": section_title,
                    "chunk": section_title + "\n\n" + chunk,
                    "position": idx
                }
                    for idx, chunk in enumerate(chunk_list, start=1)
                ]
            )

        return chunks


if __name__ == '__main__':
    node = NodeDocumentSplit()
    init_state = {
        "md_path": r"E:\output\hak180产品安全手册\hak180产品安全手册_new.md",
        # "md_path": r"E:\output\Aolynk CB304n Cable网桥 用户手册-5W100-整本手册\Aolynk CB304n Cable网桥 用户手册-5W100-整本手册.md",
        "file_title": "hak180产品安全手册",
        # "file_title": "Aolynk CB304n Cable网桥 用户手册-5W100-整本手册",
        "md_content": "md_content"
    }

    print(parse_json(node(init_state)))
