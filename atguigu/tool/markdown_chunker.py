"""结构感知的 Markdown 文档切分工具。"""

from __future__ import annotations

import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from langchain_core.documents import Document
from markdown_it import MarkdownIt
from markdown_it.token import Token


# 长度计算函数的类型：接收一段文本，返回它的字符数或 Token 数。
LengthFunction = Callable[[str], int]


@dataclass(slots=True)
class _MarkdownBlock:
    """Markdown 解析器识别出的顶层结构块及其原文位置。"""

    block_type: str  # 结构类型，例如 paragraph、code、table、list。
    content: str  # 该结构块对应的原始 Markdown 文本。
    start_line: int  # 在源文件中的起始行号，从 1 开始。
    end_line: int  # 在源文件中的结束行号，包含该行。
    heading_level: int | None = None  # 标题级别；非标题结构为 None。
    heading_text: str | None = None  # 去掉 Markdown 标记后的标题正文。


@dataclass(slots=True)
class _Section:
    """由同一组一级、二级标题管辖的 Markdown 结构块集合。"""

    h1: str | None  # 当前一级标题；文档开头可能没有一级标题。
    h2: str | None  # 当前二级标题；一级标题直属内容没有二级标题。
    blocks: list[_MarkdownBlock] = field(default_factory=list)  # section 内的结构块。
    section_index: int = -1  # section 在原文中的顺序，用于区分同名标题。


@dataclass(slots=True)
class _ChunkDraft:
    """分配 chunk_index 之前使用的内部切片对象。"""

    content: str  # 最终放入 LangChain Document.page_content 的文本。
    h1: str | None  # 切片所属的一级标题。
    h2: str | None  # 切片只属于一个二级标题时保存标题，否则为 None。
    section_indexes: list[int]  # 切片覆盖的 section 编号；跨 H2 合并时可能有多个。
    section_paths: list[str]  # 合并短块后，一个切片可能覆盖多条标题路径。
    start_line: int  # 切片内容在原文中覆盖的最小起始行。
    end_line: int  # 切片内容在原文中覆盖的最大结束行。
    block_types: list[str]  # 切片包含的 Markdown 结构类型。
    image_refs: list[str]  # 切片中引用的 Markdown 图片地址。


@dataclass(slots=True)
class _ChunkOverlap:
    """某个切片从前一个切片继承的重叠上下文。"""

    content: str = ""  # 实际添加到当前切片开头的重叠文本。
    from_chunk_index: int | None = None  # 重叠文本来自哪个全局 chunk。
    sentence_count: int = 0  # sentence 模式下实际复制的完整句子数量。


class MarkdownChunker:
    """按照标题、结构块、段落和句子的优先级切分 Markdown。

    整体处理顺序如下：

    1. 使用 Markdown 解析器识别顶层结构块，避免把代码中的 ``##`` 当标题。
    2. 按照一级标题和二级标题，把结构块组织成多个 section。
    3. section 超长时优先在结构块边界切分；单个结构块仍然超长时，
       根据段落、代码、表格、列表等类型使用不同的细分规则。
    4. 对相邻的过短切片进行合并，但默认绝不跨越一级标题边界。

    ``length_function`` 默认使用 ``len``，此时长度表示字符数。用于向量化时，
    建议传入 Embedding 模型对应 tokenizer 的计数函数，使长度配置表示 Token 数。

    ``chunk_overlap`` 是相邻切片之间的重叠长度预算。``token`` 模式允许从
    句子中间截取上一切片的尾部；``sentence`` 模式只复制预算内的完整句子。
    """

    # 中文和英文句末标点。标点后的右引号、右括号也归入当前句子。
    _SENTENCE_END_RE = re.compile(r"[。！？!?；;]+[\"'”’）】》]*")

    # 匹配无序列表和有序列表的开头，同时捕获缩进量以识别顶层列表项。
    _LIST_MARKER_RE = re.compile(r"^(?P<indent>[ \t]*)(?:[-+*]|\d+[.)])[ \t]+")

    # Markdown 围栏代码块可以使用至少三个反引号或波浪号。
    _FENCE_RE = re.compile(r"^[ \t]*(?P<marker>`{3,}|~{3,})")

    # 从这些结构的尾部截取文本会破坏 Markdown 语法，因此不参与正文重叠。
    _OVERLAP_UNSAFE_BLOCK_TYPES = {"code", "table", "list", "image", "html"}

    # 提取 Markdown 图片地址；尖括号形式允许路径中包含空格。
    _IMAGE_RE = re.compile(
        r"!\[[^\]]*]\(\s*(?:"
        r"<(?P<angle>[^>]+)>"
        r"|(?P<plain>[^\s)]+)(?=\s*(?:[\"'][^\"']*[\"'])?\s*\))"
        r")",
        re.IGNORECASE,
    )

    def __init__(
        self,
        max_chunk_size: int = 500,
        min_chunk_size: int = 100,
        length_function: LengthFunction = len,
        *,
        chunk_overlap: int = 0,
        overlap_mode: Literal["sentence", "token"] = "token",
        merge_across_h2: bool = False,
        include_front_matter: bool = False,
    ) -> None:
        """初始化 Markdown 切分器。

        Args:
            max_chunk_size: 单个切片允许的最大长度。
            min_chunk_size: 低于该长度的切片会尝试和相邻切片合并。
            length_function: 长度计算函数，可以按字符数或 Token 数计算。
            chunk_overlap: 相邻切片重叠的长度预算，0 表示关闭重叠。
            overlap_mode: ``sentence`` 保留完整句子，``token`` 按长度截取尾部。
            merge_across_h2: 是否允许同一 H1 下不同 H2 的短切片合并。
            include_front_matter: 是否把文档开头的 YAML Front Matter 放入切片。

        Raises:
            ValueError: 最大、最小长度不合法，或长度函数返回负数。
        """

        # 先校验参数，避免在切分到一半时才出现难以定位的异常。
        if max_chunk_size <= 0:
            raise ValueError("max_chunk_size 必须大于 0")
        if min_chunk_size < 0:
            raise ValueError("min_chunk_size 不能小于 0")
        if min_chunk_size > max_chunk_size:
            raise ValueError("min_chunk_size 不能大于 max_chunk_size")
        if chunk_overlap < 0:
            raise ValueError("chunk_overlap 不能小于 0")
        if chunk_overlap >= max_chunk_size:
            raise ValueError("chunk_overlap 必须小于 max_chunk_size")
        if overlap_mode not in {"sentence", "token"}:
            raise ValueError("overlap_mode 只能是 sentence 或 token")
        if length_function("") < 0:
            raise ValueError("length_function 不能返回负数")

        # 切分正文时提前预留重叠空间，使“正文 + 重叠”仍不超过最大长度。
        content_max_size = max_chunk_size - chunk_overlap
        if min_chunk_size > content_max_size:
            raise ValueError("min_chunk_size 不能大于扣除 chunk_overlap 后的可用长度")

        self.max_chunk_size = max_chunk_size
        self._content_max_size = content_max_size
        self.min_chunk_size = min_chunk_size
        self.length_function = length_function
        self.chunk_overlap = chunk_overlap
        self.overlap_mode = overlap_mode
        self.merge_across_h2 = merge_across_h2
        self.include_front_matter = include_front_matter

        # CommonMark 负责识别围栏代码块、引用、列表等嵌套结构。
        # 表格属于常见扩展语法，需要单独启用 table 规则。
        self._parser = MarkdownIt("commonmark", {"html": True}).enable("table")

    def split_file(
        self,
        md_path: str | Path,
        *,
        encoding: str = "utf-8",
        extra_metadata: dict[str, Any] | None = None,
    ) -> list[Document]:
        """读取并切分一个 Markdown 文件。

        文件的绝对路径会自动写入每个 Document 的 ``source`` metadata。

        Args:
            md_path: Markdown 文件路径。
            encoding: 文件编码，默认使用 UTF-8。
            extra_metadata: 需要附加到每个切片的公共 metadata。

        Returns:
            按原文顺序排列的 LangChain Document 列表。
        """

        path = Path(md_path)
        content = path.read_text(encoding=encoding)

        # source 由工具统一生成；调用方仍可通过 extra_metadata 覆盖它。
        metadata = {"source": str(path.resolve())}
        if extra_metadata:
            metadata.update(extra_metadata)
        return self.split_text(content, extra_metadata=metadata)

    def split_text(
        self,
        md_content: str,
        add_overlap_to_content: bool = False,
        *,
        extra_metadata: dict[str, Any] | None = None,
    ) -> list[Document]:
        """切分 Markdown 字符串并返回 LangChain ``Document`` 对象。

        Args:
            md_content: 完整的 Markdown 原文。
            extra_metadata: 写入每一个切片的公共 metadata。

        Returns:
            切分后的 Document 列表；空字符串返回空列表。
        """

        # 空文档没有可向量化内容，直接返回，避免解析器做无效工作。
        if not md_content or not md_content.strip():
            return []

        # 第一步：将 Markdown 原文解析成结构块。
        blocks = self._parse_blocks(md_content)

        # 第二步：按 H1/H2 标题路径组织成 section。
        sections = self._group_sections(blocks)

        # 第三步：对每个 section 做“结构块优先”的超长切分。
        drafts: list[_ChunkDraft] = []
        for section in sections:
            drafts.extend(self._split_section(section))

        # 第四步：在不跨 H1、不超过最大长度的前提下合并短切片。
        drafts = self._merge_short_chunks(drafts)
        common_metadata = extra_metadata or {}

        # 第五步：为同一 section 中的相邻切片补充上一切片的尾部上下文。
        overlaps = self._build_chunk_overlaps(drafts)

        # 预先计算每个 chunk 在所属标题 section 中的位置。
        # 使用 section_index 分组，可以正确区分文档中路径相同的重复标题。
        section_positions = self._build_section_positions(drafts, sections)

        # 最后将内部对象转换为 LangChain Document，并统一分配连续索引。
        documents: list[Document] = []
        for index, draft in enumerate(drafts):
            positions = section_positions[index]
            overlap = overlaps[index]

            # 普通 chunk 只属于一个 section，可以直接暴露单值位置字段。
            # 跨 H2 合并的 chunk 没有唯一的最近标题，完整归属放在 section_positions 中。
            single_position = positions[0] if len(positions) == 1 else None
            metadata = dict(common_metadata)
            metadata.update(
                {
                    "chunk_index": index,
                    "h1": draft.h1,
                    "h2": draft.h2,
                    "nearest_heading": (
                        (single_position["nearest_heading"] if single_position else None) or "无标题"
                    ),
                    "nearest_heading_level": (
                        single_position["nearest_heading_level"] if single_position else None
                    ),
                    "section_chunk_index": (
                        single_position["section_chunk_index"] if single_position else None
                    ),
                    "section_chunk_count": (
                        single_position["section_chunk_count"] if single_position else None
                    ),
                    "chunk_overlap": self.chunk_overlap,
                    "overlap_mode": self.overlap_mode,
                    "overlap_from_chunk_index": overlap.from_chunk_index,
                    "overlap_length": self._length(overlap.content),
                    "overlap_sentence_count": overlap.sentence_count,
                    "overlap_content": overlap.content + "\n\n" if overlap.content else "",
                    "section_positions": positions,
                    "section_path": " | ".join(draft.section_paths),
                    "section_paths": list(draft.section_paths),
                    "start_line": draft.start_line,
                    "end_line": draft.end_line,
                    "block_types": list(draft.block_types),
                    "image_refs": list(draft.image_refs),
                    "chunk_length": self._length(draft.content),
                }
            )
            h1= draft.h1 + "\n\n" if draft.h1 else ""
            h2= draft.h2 + "\n\n" if draft.h2 else ""
            overlap_content = metadata.get('overlap_content') if add_overlap_to_content else ""
            page_content = h1 + h2 + f"position:{metadata.get('section_chunk_index')}\n\n" + overlap_content + draft.content
            documents.append(Document(page_content=page_content, metadata=metadata))

        return documents

    def _build_chunk_overlaps(
        self,
        drafts: Sequence[_ChunkDraft],
    ) -> list[_ChunkOverlap]:
        """为每个切片计算来自同一 section 前一个切片的重叠文本。"""

        overlaps = [_ChunkOverlap() for _ in drafts]
        if self.chunk_overlap == 0:
            return overlaps

        for index in range(1, len(drafts)):
            previous = drafts[index - 1]
            current = drafts[index]

            # 只有两个普通切片唯一且明确地属于同一个 section 时才添加重叠。
            # 跨 H2 合并的切片包含多个 section，不继续传播可能混杂的上下文。
            if (
                len(previous.section_indexes) != 1
                or len(current.section_indexes) != 1
                or previous.section_indexes[0] != current.section_indexes[0]
            ):
                continue

            # 标题已经通过 h1/h2 注入每个切片，无需把标题正文再次当作 overlap。
            if set(previous.block_types) <= {"heading"}:
                continue

            # 避免截断代码围栏、表格、列表等完整 Markdown 结构。
            if self._OVERLAP_UNSAFE_BLOCK_TYPES.intersection(previous.block_types):
                continue

            # 原子结构块可能因语法完整性而允许超过正文预算，此时不能再追加重叠。
            available = self.max_chunk_size - self._length(current.content)
            overlap_budget = min(self.chunk_overlap, max(available, 0))
            if overlap_budget == 0:
                continue

            content, sentence_count = self._extract_overlap_tail(
                previous.content,
                overlap_budget,
            )
            if not content:
                continue

            overlaps[index] = _ChunkOverlap(
                content=content,
                from_chunk_index=index - 1,
                sentence_count=sentence_count,
            )

        return overlaps

    def _extract_overlap_tail(self, text: str, budget: int) -> tuple[str, int]:
        """按照配置模式提取不超过长度预算的上一切片尾部。"""

        if not text or budget <= 0:
            return "", 0

        if self.overlap_mode == "sentence":
            sentences = self._sentence_units(text)
            selected: list[str] = []

            # 从最后一句向前选择，保证取得的是离当前切片最近的完整句子。
            for sentence in reversed(sentences):
                candidate = "".join([sentence, *selected])
                if self._length(candidate) > budget:
                    break
                selected.insert(0, sentence)

            content = "".join(selected).strip()
            return content, len(selected) if content else 0

        # token 模式允许从句子中间开始，通过二分查找取得预算内最长文本后缀。
        start = self._smallest_fitting_suffix_start(text, budget)
        return text[start:].strip(), 0

    def _smallest_fitting_suffix_start(self, text: str, budget: int) -> int:
        """寻找使文本后缀不超过长度预算的最小字符起始下标。"""

        low = 0
        high = len(text)
        best = len(text)

        while low <= high:
            middle = (low + high) // 2
            if self._length(text[middle:]) <= budget:
                best = middle
                high = middle - 1
            else:
                low = middle + 1
        return best

    @staticmethod
    def _build_section_positions(
        drafts: Sequence[_ChunkDraft],
        sections: Sequence[_Section],
    ) -> list[list[dict[str, Any]]]:
        """计算每个切片在其所属标题 section 中的顺序和总数。

        返回列表与 drafts 一一对应。普通切片只有一条位置记录；跨 H2 合并的
        切片会为每个真实来源 section 保存一条记录。
        """

        section_lookup = {
            section.section_index: section
            for section in sections
        }

        # 一个跨 section 的合并切片，会分别算作每个来源 section 中的一个 chunk。
        section_chunk_counts: dict[int, int] = {}
        for draft in drafts:
            for section_index in draft.section_indexes:
                section_chunk_counts[section_index] = (
                    section_chunk_counts.get(section_index, 0) + 1
                )

        # 按原文顺序累计当前 section 已经出现了多少个 chunk。
        section_seen_counts: dict[int, int] = {}
        all_positions: list[list[dict[str, Any]]] = []

        for draft in drafts:
            draft_positions: list[dict[str, Any]] = []
            for section_index in draft.section_indexes:
                section = section_lookup[section_index]
                section_seen_counts[section_index] = (
                    section_seen_counts.get(section_index, 0) + 1
                )

                # 有 H2 时离正文最近的是 H2，否则回退到 H1；文档开头可能都没有。
                nearest_heading = section.h2 or section.h1
                nearest_heading_level = 2 if section.h2 else (1 if section.h1 else None)

                draft_positions.append(
                    {
                        "section_path": MarkdownChunker._section_path(
                            section.h1,
                            section.h2,
                        ),
                        "h1": section.h1,
                        "h2": section.h2,
                        "nearest_heading": nearest_heading,
                        "nearest_heading_level": nearest_heading_level,
                        # section 内的位置使用 1 基编号，直接对应“第几个 chunk”。
                        "section_chunk_index": section_seen_counts[section_index],
                        "section_chunk_count": section_chunk_counts[section_index],
                    }
                )
            all_positions.append(draft_positions)

        return all_positions

    def _parse_blocks(self, md_content: str) -> list[_MarkdownBlock]:
        """把原文解析为顶层结构块，同时保留原始 Markdown 和行号。"""

        # keepends=True 可以保留原始换行符，之后才能尽量还原代码和列表格式。
        lines = md_content.splitlines(keepends=True)
        front_matter_end = self._find_front_matter_end(lines)
        blocks: list[_MarkdownBlock] = []

        # Front Matter 默认只充当文档元信息；显式开启后才作为正文结构块输出。
        if front_matter_end and self.include_front_matter:
            blocks.append(
                _MarkdownBlock(
                    block_type="front_matter",
                    content="".join(lines[:front_matter_end]).rstrip("\r\n"),
                    start_line=1,
                    end_line=front_matter_end,
                )
            )

        # MarkdownIt 不负责这里自定义识别的 Front Matter，所以只解析正文部分。
        body_lines = lines[front_matter_end:]
        tokens = self._parser.parse("".join(body_lines))

        for index, token in enumerate(tokens):
            # 顶层打开 Token 或叶子 Token 已经拥有完整的原文范围。
            # 它们的子 Token 会和父范围重叠，不能重复输出，否则列表、引用块等
            # 嵌套结构会在结果中出现两次。
            if token.level != 0 or token.map is None or token.nesting == -1:
                continue

            # token.map 使用“起始行包含、结束行不包含”的 0 基下标。
            local_start, local_end = token.map
            if local_end <= local_start:
                continue

            # 解析正文时去掉了 Front Matter，这里需要加回偏移量得到原文行号。
            source_start = front_matter_end + local_start
            source_end = front_matter_end + local_end
            raw_content = "".join(lines[source_start:source_end]).rstrip("\r\n")
            if not raw_content:
                continue

            # 先把 MarkdownIt 的 Token 类型转换成业务更容易理解的结构类型。
            block_type = self._block_type(token)
            heading_level: int | None = None
            heading_text: str | None = None

            # 标题还要额外保存级别和去掉 Markdown 标记后的可见文本。
            if token.type == "heading_open":
                heading_level = int(token.tag[1:])
                heading_text = self._read_heading_text(tokens, index)
                block_type = "heading"
            # 包含图片的段落标为 image，方便后续提取图片引用或做多模态处理。
            elif token.type == "paragraph_open" and self._contains_image(tokens, index):
                block_type = "image"

            blocks.append(
                _MarkdownBlock(
                    block_type=block_type,
                    content=raw_content,
                    start_line=source_start + 1,
                    end_line=source_end,
                    heading_level=heading_level,
                    heading_text=heading_text,
                )
            )

        return blocks

    @staticmethod
    def _find_front_matter_end(lines: Sequence[str]) -> int:
        """识别文档开头的 YAML Front Matter，返回结束行的排他下标。

        ---
        title: 设备维护手册
        author: 张三
        category: SOP
        ---

        返回 0 表示没有合法的 Front Matter。排他下标意味着
        ``lines[:返回值]`` 正好可以取得完整 Front Matter。
        """

        # YAML Front Matter 必须从文件第一行的三个短横线开始。
        if not lines or lines[0].strip() != "---":
            return 0

        # 从第二行开始寻找结束标记，结束标记可以是 --- 或 ...。
        for index in range(1, len(lines)):
            if lines[index].strip() not in {"---", "..."}:
                continue

            # 至少包含一个“键: 值”形态，避免把两个普通分隔线误判成 YAML。
            body = lines[1:index]
            if any(":" in line for line in body):
                return index + 1
            return 0
        return 0

    @staticmethod
    def _block_type(token: Token) -> str:
        """把 MarkdownIt 的 Token 类型转换为统一的业务结构类型。"""

        # 同一种结构在 MarkdownIt 中可能存在 open、close、leaf 等 Token，
        # 这里只需要顶层结构的统一名称，后续切分规则无需了解解析器细节。
        mapping = {
            "paragraph_open": "paragraph",
            "fence": "code",
            "code_block": "code",
            "bullet_list_open": "list",
            "ordered_list_open": "list",
            "blockquote_open": "blockquote",
            "table_open": "table",
            "html_block": "html",
            "hr": "thematic_break",
        }
        return mapping.get(token.type, token.type.removesuffix("_open"))

    @staticmethod
    def _read_heading_text(tokens: Sequence[Token], heading_index: int) -> str:
        """读取标题中用户真正看到的文本，忽略强调、链接等控制标记。"""

        # heading_open 后面正常应该紧跟一个 inline Token。
        if heading_index + 1 >= len(tokens):
            return ""

        inline = tokens[heading_index + 1]
        if inline.type != "inline":
            return ""
        if not inline.children:
            return inline.content.strip()

        # link_open、strong_open 等 Token 没有可见文字，不应写入标题 metadata。
        visible_parts: list[str] = []
        for child in inline.children:
            if child.type in {"text", "code_inline", "html_inline", "image"}:
                visible_parts.append(child.content)
        return "".join(visible_parts).strip()

    @staticmethod
    def _contains_image(tokens: Sequence[Token], paragraph_index: int) -> bool:
        """判断一个段落的行内子 Token 中是否包含 Markdown 图片。"""

        if paragraph_index + 1 >= len(tokens):
            return False
        inline = tokens[paragraph_index + 1]
        return bool(
            inline.type == "inline"
            and inline.children
            and any(child.type == "image" for child in inline.children)
        )

    def _group_sections(self, blocks: Sequence[_MarkdownBlock]) -> list[_Section]:
        """按照 H1/H2 标题路径，把连续结构块组织成多个 section。"""

        sections: list[_Section] = []
        current_h1: str | None = None
        current_h2: str | None = None
        current_blocks: list[_MarkdownBlock] = []

        def flush() -> None:
            """保存当前 section，并清空缓冲区准备接收下一个 section。"""

            nonlocal current_blocks
            if current_blocks:
                sections.append(
                    _Section(
                        h1=current_h1,
                        h2=current_h2,
                        blocks=current_blocks,
                        section_index=len(sections),
                    )
                )
                current_blocks = []

        for block in blocks:
            # 遇到新的 H1 时，上一 section 结束，同时 H2 上下文必须清空。
            if block.heading_level == 1:
                flush()
                current_h1 = block.heading_text
                current_h2 = None
            # 遇到新的 H2 时，只更新二级标题，继续继承当前 H1。
            elif block.heading_level == 2:
                flush()
                current_h2 = block.heading_text

            # 标题本身也保留在 section 中，使第一个切片仍包含原始标题语法。
            current_blocks.append(block)

        flush()
        return sections

    def _split_section(self, section: _Section) -> list[_ChunkDraft]:
        """优先按完整结构块组装切片，只有单块超长时才破坏块边界。"""

        drafts: list[_ChunkDraft] = []
        buffered: list[_MarkdownBlock] = []

        def flush_buffer() -> None:
            """把当前缓冲区转换为一个切片草稿并清空缓冲区。"""

            nonlocal buffered
            if buffered:
                drafts.append(self._draft_from_blocks(section, buffered))
                buffered = []

        for block in section.blocks:
            # 单个结构块已经超长，无法再和其他块打包，需要按类型细分。
            if self._length(block.content) > self._content_max_size:
                flush_buffer()
                for piece in self._split_oversized_block(block):
                    drafts.append(self._draft_from_piece(section, block, piece))
                continue

            # 先试放当前块；若组合后超长，就先结算已有缓冲区。
            candidate = self._join_blocks([*buffered, block])
            if buffered and self._length(candidate) > self._content_max_size:
                flush_buffer()
            # 当前块本身没有超长，因此一定可以作为新缓冲区的第一个块。
            buffered.append(block)

        flush_buffer()
        return drafts

    def _split_oversized_block(self, block: _MarkdownBlock) -> list[str]:
        """根据 Markdown 结构类型选择不会轻易破坏语法的细分策略。"""

        # 代码块切开后必须为每片重新补齐代码围栏。
        if block.block_type == "code":
            return self._split_code_block(block.content)

        # 表格切开后必须在每片重复表头和分隔行。
        if block.block_type == "table":
            return self._split_table(block.content)

        # 列表优先按照同级列表项切分，避免从嵌套列表中间断开。
        if block.block_type == "list":
            return self._split_list(block.content)

        # 引用块先在空行形成的段落边界切分。
        if block.block_type == "blockquote":
            return self._split_blank_line_groups(block.content)
        if block.block_type in {"image", "html"}:
            # 图片语法或 HTML 元素从中间断开通常会直接失效。
            # 因此极少数超长原子块允许暂时超过上限，优先保证语法完整。
            return [block.content]

        # 普通段落、标题等文本结构最终按句子切分。
        return self._split_sentences(block.content)

    def _split_sentences(self, text: str) -> list[str]:
        """按中英文句末标点拆句，并把句末标点保留在原句中。"""

        units: list[str] = []
        start = 0
        # 使用 match.end() 作为切点，所以标点和紧随其后的右引号不会丢失。
        for match in self._SENTENCE_END_RE.finditer(text):
            end = match.end()
            units.append(text[start:end])
            start = end
        if start < len(text):
            units.append(text[start:])

        # 拆句后还要重新打包，尽量让切片接近上限而不是“一句一个切片”。
        return self._pack_plain_units(units or [text])

    def _split_blank_line_groups(self, text: str) -> list[str]:
        """按照空行形成的段落边界切分引用块等多段文本。"""

        # 正向预查保留空行本身，尽量不改变原 Markdown 的段落形态。
        groups = re.split(r"(?=\n[ \t]*\n)", text)
        return self._pack_plain_units(groups)

    def _split_list(self, text: str) -> list[str]:
        """按照最小缩进层级的列表项切分列表，并保留嵌套子项。"""

        # 保留换行，确保一个列表项包含的多行正文和嵌套列表不被粘连。
        lines = text.splitlines(keepends=True)
        marker_rows: list[tuple[int, int]] = []

        # 记录所有列表标记所在行以及展开 Tab 后的实际缩进宽度。
        for index, line in enumerate(lines):
            match = self._LIST_MARKER_RE.match(line)
            if match:
                marker_rows.append((index, len(match.group("indent").expandtabs(4))))

        # 理论上 list Token 应当能找到列表标记；找不到时退回普通句子切分。
        if not marker_rows:
            return self._split_sentences(text)

        # 最小缩进代表当前列表的顶层；更深缩进属于某个顶层项的子内容。
        min_indent = min(indent for _, indent in marker_rows)
        starts = [index for index, indent in marker_rows if indent == min_indent]

        # 加入总行数作为最后一个列表项的排他结束位置。
        starts.append(len(lines))

        # 每个 item 都包含它下面所有更深缩进的嵌套内容。
        items = ["".join(lines[start:end]) for start, end in zip(starts, starts[1:])]
        expanded: list[str] = []
        for item in items:
            if self._length(item) <= self._content_max_size:
                expanded.append(item)
            else:
                # 单个列表项自身超长时，再按句子拆，并在每片重复列表标记。
                expanded.extend(self._split_oversized_list_item(item))

        # 相邻列表项仍可在不超过最大长度时重新打包到同一个切片。
        return self._pack_plain_units(expanded)

    def _split_oversized_list_item(self, item: str) -> list[str]:
        """细分单个超长列表项，并让每个子片段仍保持列表语法。"""

        match = self._LIST_MARKER_RE.match(item)
        if not match:
            return self._split_sentences(item)

        # 例如从 ``- 很长的内容`` 中分别取得 ``- `` 和正文。
        marker = match.group(0)
        body = item[match.end():]

        # renderer 会给每个拆出的正文片段重新加上原列表标记。
        return self._split_with_renderer(
            self._sentence_units(body),
            lambda payload: marker + payload.lstrip(),
        )

    def _split_table(self, text: str) -> list[str]:
        """按数据行切分 Markdown 表格，并为每片重复表头。"""

        lines = text.splitlines()

        # 标准 Markdown 表格至少包含表头、分隔行和一行数据。
        if len(lines) < 3:
            return self._pack_plain_units([line + "\n" for line in lines])

        # 前两行共同组成表头，后续每一行都是可独立打包的数据行。
        header = "\n".join(lines[:2])
        rows = lines[2:]

        # 每生成一个切片，都通过 render 在数据行前重新加上完整表头。
        render = lambda payload: f"{header}\n{payload}" if payload else header

        # 如果表头自己已经达到上限，再切会破坏表格语法。
        # 此时优先保留完整表格，允许它暂时超过配置长度。
        if self._length(render("")) >= self._content_max_size:
            return [text]
        return self._split_with_renderer(rows, render, separator="\n")

    def _split_code_block(self, text: str) -> list[str]:
        """按代码行切分代码块，并给每片重新补齐开、关围栏。"""

        # source_lines 用于无围栏代码的回退处理，必须保留原始换行。
        source_lines = text.splitlines(keepends=True)

        # 围栏代码的 render 会主动添加换行，所以识别围栏时先去掉行尾换行。
        lines = [line.rstrip("\r\n") for line in source_lines]
        if not lines:
            return []

        # 缩进代码块没有 ``` 或 ~~~，直接按保留换行的代码行打包。
        opening_match = self._FENCE_RE.match(lines[0])
        if not opening_match or len(lines) < 2:
            return self._pack_plain_units(source_lines)

        marker = opening_match.group("marker")
        closing_index = None

        # 从末尾向前寻找相同字符、且长度不少于开围栏的关围栏。
        for index in range(len(lines) - 1, 0, -1):
            stripped = lines[index].lstrip()
            if stripped.startswith(marker[0] * len(marker)):
                closing_index = index
                break

        # 围栏不完整时不擅自补语法，退回按原始行切分。
        if closing_index is None:
            return self._pack_plain_units(source_lines)

        # opening 中会保留语言标记，例如 ```python。
        opening = lines[0]
        closing = lines[closing_index]
        body_lines = lines[1:closing_index]

        # 每个代码切片都渲染成一个可以独立显示的完整代码块。
        render = lambda payload: f"{opening}\n{payload}\n{closing}"

        # 仅开关围栏就达到长度上限时，无法在不破坏语法的情况下继续切。
        if self._length(render("")) >= self._content_max_size:
            return [text]

        # 空代码块无需切分。
        if not body_lines:
            return [text]
        return self._split_with_renderer(body_lines, render, separator="\n")

    def _split_with_renderer(
        self,
        units: Sequence[str],
        render: Callable[[str], str],
        *,
        separator: str = "",
    ) -> list[str]:
        """打包正文单元，并通过 renderer 为每个切片重复必要的语法外壳。

        表格使用 renderer 重复表头，代码块使用 renderer 重复开关围栏，
        超长列表项则使用 renderer 重复列表标记。
        """

        chunks: list[str] = []
        buffered: list[str] = []

        def flush() -> None:
            """渲染并保存当前正文缓冲区，然后清空缓冲区。"""

            nonlocal buffered
            if buffered:
                chunks.append(render(separator.join(buffered)).strip("\r\n"))
                buffered = []

        for unit in units:
            # 先模拟把当前单元放入缓冲区后的最终渲染长度。
            candidate = render(separator.join([*buffered, unit]))

            # 加入后超长时，先结算已有单元，再单独处理当前单元。
            if buffered and self._length(candidate) > self._content_max_size:
                flush()

            # 当前单元套上语法外壳后仍然超长，只能进入字符级兜底切分。
            if self._length(render(unit)) > self._content_max_size:
                flush()
                chunks.extend(render(part) for part in self._hard_split_payload(unit, render))
            else:
                # 当前单元可以安全放入新的或已有的缓冲区。
                buffered.append(unit)

        flush()
        return [chunk for chunk in chunks if chunk]

    def _pack_plain_units(self, units: Sequence[str]) -> list[str]:
        """在不超过最大长度的前提下，贪心打包普通文本单元。"""

        chunks: list[str] = []
        buffered = ""

        for unit in units:
            # 空单元既没有语义，也会干扰长度判断，直接跳过。
            if not unit:
                continue

            # 尝试把新单元追加到当前缓冲区。
            candidate = buffered + unit
            if buffered and self._length(candidate) > self._content_max_size:
                # 组合后超长，先保存旧缓冲区，再从空缓冲区处理当前单元。
                chunks.append(buffered.strip("\r\n"))
                buffered = ""

            # 一个单元自己就超长时，句子或行级边界已经不够，只能字符兜底。
            if self._length(unit) > self._content_max_size:
                parts = self._hard_split_text(unit)

                # 除最后一片外都已经接近上限，可以直接写入结果。
                chunks.extend(part.strip("\r\n") for part in parts[:-1] if part)

                # 最后一片可能较短，先留在缓冲区，争取和下一个单元合并。
                buffered = parts[-1] if parts else ""
            else:
                buffered += unit

        # 循环结束后不要遗漏最后一个尚未结算的缓冲区。
        if buffered:
            chunks.append(buffered.strip("\r\n"))
        return [chunk for chunk in chunks if chunk]

    def _hard_split_payload(
        self,
        payload: str,
        render: Callable[[str], str],
    ) -> list[str]:
        """对带语法外壳后仍超长的单个正文执行字符级兜底切分。"""

        parts: list[str] = []
        remaining = payload
        while remaining:
            # 二分查找当前剩余文本能放入语法外壳的最大字符前缀。
            cut = self._largest_fitting_prefix(remaining, render)
            if cut == 0:
                # 语法外壳本身已经占满上限。此时保留剩余正文，
                # 避免死循环，也避免为了满足长度强行破坏 Markdown 语法。
                parts.append(remaining)
                break
            parts.append(remaining[:cut])
            remaining = remaining[cut:]
        return parts

    def _hard_split_text(self, text: str) -> list[str]:
        """把无法按句子或行切开的超长纯文本按最大可容纳字符数切分。"""

        parts: list[str] = []
        remaining = text
        while remaining:
            # identity renderer 表示没有代码围栏、表头等额外长度开销。
            cut = self._largest_fitting_prefix(remaining, lambda value: value)
            if cut == 0:
                # 防御异常长度函数，至少前进一个字符，保证循环一定结束。
                cut = 1
            parts.append(remaining[:cut])
            remaining = remaining[cut:]
        return parts

    def _largest_fitting_prefix(
        self,
        text: str,
        render: Callable[[str], str],
    ) -> int:
        """使用二分查找找到渲染后不超过上限的最长文本前缀。"""

        # low 表示当前已知可行的字符数量，high 表示待搜索的最大数量。
        low = 0
        high = len(text)
        while low < high:
            # 向上取中点，避免 low 和 high 相邻时无法继续推进。
            middle = (low + high + 1) // 2
            if self._length(render(text[:middle])) <= self._content_max_size:
                low = middle
            else:
                high = middle - 1
        return low

    def _draft_from_blocks(
        self,
        section: _Section,
        blocks: Sequence[_MarkdownBlock],
    ) -> _ChunkDraft:
        """把同一 section 中的一组完整结构块转换为切片草稿。"""

        # 块之间使用空行连接，避免标题、段落、列表等结构粘在一起。
        content = self._join_blocks(blocks)
        return _ChunkDraft(
            content=content,
            h1=section.h1,
            h2=section.h2,
            section_indexes=[section.section_index],
            section_paths=[self._section_path(section.h1, section.h2)],
            # blocks 按原文顺序排列，所以首尾块直接给出覆盖行号。
            start_line=blocks[0].start_line,
            end_line=blocks[-1].end_line,
            block_types=self._unique(block.block_type for block in blocks),
            image_refs=self._extract_image_refs(content),
        )

    def _draft_from_piece(
        self,
        section: _Section,
        block: _MarkdownBlock,
        piece: str,
    ) -> _ChunkDraft:
        """把一个超长结构块拆出的子片段转换为切片草稿。"""

        # 句子级子片段没有单独重新计算精确行号，因此仍记录其来源块的行号范围。
        return _ChunkDraft(
            content=piece,
            h1=section.h1,
            h2=section.h2,
            section_indexes=[section.section_index],
            section_paths=[self._section_path(section.h1, section.h2)],
            start_line=block.start_line,
            end_line=block.end_line,
            block_types=[block.block_type],
            image_refs=self._extract_image_refs(piece),
        )

    def _merge_short_chunks(self, drafts: list[_ChunkDraft]) -> list[_ChunkDraft]:
        """合并相邻短切片，优先选择标题路径完全相同的邻居。

        每成功合并一次，列表长度都会减少 1，因此循环一定会结束。
        无法安全合并的短切片会被保留，而不会强行跨 H1 或突破最大长度。
        """

        # 复制列表，避免直接修改调用方持有的草稿列表对象。
        merged = list(drafts)
        while len(merged) > 1:
            # 每轮从左到右寻找第一个低于最小长度的切片。
            short_index = next(
                (
                    index
                    for index, draft in enumerate(merged)
                    if self._length(draft.content) < self.min_chunk_size
                ),
                None,
            )
            if short_index is None:
                # 已经没有短切片，合并工作完成。
                break

            # 候选元组依次保存：跨 section 惩罚、合并后长度、邻居下标。
            # 使用元组排序可以优先同 section，其次选择合并后更短的方案。
            candidates: list[tuple[int, int, int]] = []

            # 只允许和紧邻的前一个或后一个切片合并，保证原文顺序不被打乱。
            for neighbor_index in (short_index - 1, short_index + 1):
                if not 0 <= neighbor_index < len(merged):
                    continue
                # _can_merge 会同时检查标题边界和最大长度。
                if not self._can_merge(merged[short_index], merged[neighbor_index]):
                    continue

                combined_length = self._combined_length(
                    merged[min(short_index, neighbor_index)],
                    merged[max(short_index, neighbor_index)],
                )
                # 标题路径不同记为 1，相同记为 0；min() 会优先选择 0。
                same_section_penalty = int(
                    merged[short_index].section_paths
                    != merged[neighbor_index].section_paths
                )
                candidates.append((same_section_penalty, combined_length, neighbor_index))

            if not candidates:
                # 当前短切片无法安全合并，但后面可能还有可合并的短切片。
                remaining_short = [
                    index
                    for index in range(short_index + 1, len(merged))
                    if self._length(merged[index].content) < self.min_chunk_size
                ]
                if not remaining_short:
                    break

                # 跳过当前不可合并项，从后续短项中尝试完成一次有效合并。
                # 辅助方法只遍历有限下标，不会因为同一个短项陷入死循环。
                if not self._merge_one_reachable_short(merged, remaining_short):
                    break
                continue

            # 元组按“同 section 优先、合并后更短优先、下标更小优先”排序。
            _, _, neighbor_index = min(candidates)

            # 无论选择前邻居还是后邻居，都按原文的左右顺序拼接。
            left_index = min(short_index, neighbor_index)
            right_index = max(short_index, neighbor_index)
            merged[left_index] = self._combine_drafts(
                merged[left_index], merged[right_index]
            )
            del merged[right_index]

        return merged

    def _merge_one_reachable_short(
        self,
        drafts: list[_ChunkDraft],
        indexes: Sequence[int],
    ) -> bool:
        """从指定短切片中完成一次可行合并，成功返回 True。"""

        for index in indexes:
            for neighbor in (index - 1, index + 1):
                if not 0 <= neighbor < len(drafts):
                    continue
                if not self._can_merge(drafts[index], drafts[neighbor]):
                    continue
                # 始终用较小下标作为左侧，保持 Markdown 原文顺序。
                left = min(index, neighbor)
                right = max(index, neighbor)
                drafts[left] = self._combine_drafts(drafts[left], drafts[right])
                del drafts[right]
                return True
        return False

    def _can_merge(self, first: _ChunkDraft, second: _ChunkDraft) -> bool:
        """判断两个相邻切片是否满足标题边界和长度约束。"""

        # 一级标题通常代表独立主题，默认绝不跨越 H1 合并。
        if first.h1 != second.h1:
            return False

        # 调用方可以关闭跨 H2 合并，使二级标题也成为硬边界。
        if not self.merge_across_h2 and first.h2 != second.h2:
            return False

        # 合并后仍然必须不超过最大切片长度。
        return self._combined_length(first, second) <= self._content_max_size

    def _combined_length(self, first: _ChunkDraft, second: _ChunkDraft) -> int:
        """计算两个切片按 Markdown 段落形式连接后的实际长度。"""

        return self._length(f"{first.content}\n\n{second.content}")

    def _combine_drafts(
        self,
        first: _ChunkDraft,
        second: _ChunkDraft,
    ) -> _ChunkDraft:
        """按照原文顺序合并两个切片及其全部 metadata。"""

        # 两段之间保留一个空行，避免前后 Markdown 结构粘连。
        content = f"{first.content.rstrip()}\n\n{second.content.lstrip()}"

        # 只有两个切片属于同一 H2 时，合并结果才能保留单一 h2 值。
        # 跨 H2 合并时使用 section_paths 表达多个真实来源，h2 设置为 None。
        h2 = first.h2 if first.h2 == second.h2 else None
        return _ChunkDraft(
            content=content,
            h1=first.h1,
            h2=h2,
            section_indexes=list(dict.fromkeys([
                *first.section_indexes,
                *second.section_indexes,
            ])),
            section_paths=self._unique([*first.section_paths, *second.section_paths]),
            start_line=min(first.start_line, second.start_line),
            end_line=max(first.end_line, second.end_line),
            block_types=self._unique([*first.block_types, *second.block_types]),
            image_refs=self._unique([*first.image_refs, *second.image_refs]),
        )

    @classmethod
    def _extract_image_refs(cls, content: str) -> list[str]:
        """提取切片中的 Markdown 图片地址，并按首次出现顺序去重。"""

        refs: list[str] = []
        for match in cls._IMAGE_RE.finditer(content):
            # 尖括号地址和普通无空格地址位于两个互斥捕获组中。
            ref = match.group("angle") or match.group("plain")
            if ref and ref not in refs:
                refs.append(ref)
        return refs

    @staticmethod
    def _section_path(h1: str | None, h2: str | None) -> str:
        """把 H1/H2 转换为适合检索和展示的标题路径字符串。"""

        path = [heading for heading in (h1, h2) if heading]
        return " > ".join(path) if path else "文档开头"

    @staticmethod
    def _join_blocks(blocks: Sequence[_MarkdownBlock]) -> str:
        """用一个空行连接完整结构块，同时清理块首尾多余换行。"""

        return "\n\n".join(block.content.strip("\r\n") for block in blocks).strip()

    @staticmethod
    def _unique(values: Sequence[str] | Any) -> list[str]:
        """在保持首次出现顺序的前提下去重。"""

        # Python 字典保持插入顺序，dict.fromkeys 可实现稳定去重。
        return list(dict.fromkeys(values))

    def _sentence_units(self, text: str) -> list[str]:
        """把文本拆成保留句末标点的句子单元，但暂时不进行长度打包。"""

        units: list[str] = []
        start = 0
        for match in self._SENTENCE_END_RE.finditer(text):
            units.append(text[start:match.end()])
            start = match.end()
        if start < len(text):
            units.append(text[start:])
        return units or [text]

    def _length(self, text: str) -> int:
        """统一调用长度函数，并防御运行过程中出现负数结果。"""

        value = self.length_function(text)
        if value < 0:
            raise ValueError("length_function 不能返回负数")
        return value
