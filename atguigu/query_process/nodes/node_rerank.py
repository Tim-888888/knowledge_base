'''
@Author  :61022
@Time    :2026/8/9
@Desc    :
'''
from typing import Tuple, List, Dict

from atguigu.query_process.base import NodeBase
from atguigu.query_process.state import QueryGraphState
from atguigu.tool.json_format_util import parse_json
from atguigu.tool.logger import logger
from atguigu.tool.reranker_utils import get_reranker_result


class NodeRerank(NodeBase):
    """
    节点功能：(LLM精排)使用 Cross-Encoder 模型对 RRF 后的结果进行精确打分重排。
    """

    # 覆盖基类的 name 属性，标识节点名称
    name: str = "node_rerank"

    def process(self, state: QueryGraphState) -> QueryGraphState:
        """
        节点逻辑
        :param state: 工作流状态对象
        :return: 更新后的状态对象
        """

        # 融合rrk和web两路数据
        rewritten_query, final_docs = self.init_prams(state)

        # 调用Reranker模型进行重排序
        sorted_docs = self.reranking_docs(final_docs, rewritten_query)

        # 分数断崖检测
        reranked_docs = self.score_cutoff(sorted_docs)

        return {"reranked_docs": reranked_docs}

    def reranking_docs(self, final_docs: list[dict], rewritten_query: str) -> list[dict]:
        # 拿正文
        chunks = [final_doc.get("content") for final_doc in final_docs]

        # 用qwen3-reranker进行交叉编码重排序
        scores = get_reranker_result(rewritten_query, chunks)

        # 组装分数和docs
        final_docs = [{**final_docs[idx], "score": score} for idx, score in enumerate(scores)]

        # 按分数倒序排序
        sorted_docs = sorted(final_docs, key=lambda doc: doc.get("score"), reverse=True)
        return sorted_docs

    def init_prams(self, state) -> Tuple[str, List[Dict]]:
        rewritten_query = state.get("rewritten_query")
        if not rewritten_query:
            ValueError(f"rewritten_query 不能为空")

        rrf_chunks = state.get("rrf_chunks")
        if not rrf_chunks:
            ValueError(f"rrf_chunks 不能为空")

        web_search_docs = state.get("web_search_docs")
        if not web_search_docs:
            ValueError(f"web_search_docs 不能为空")

        rrf_chunks = [
            {
                "title": rrf_chunk.get("item_name"),
                "content": rrf_chunk.get("content"),
                "chunk_id": rrf_chunk.get("chunk_id"),
                "url": None,
                "source": rrf_chunk.get("source")
            } for rrf_chunk in rrf_chunks
        ]

        web_search_docs = [{
            "title": web_search_doc.get("title"),
            "content": web_search_doc.get("snippet"),
            "chunk_id": None,
            "url": web_search_doc.get("url"),
            "source": web_search_doc.get("source")
        } for web_search_doc in web_search_docs]

        rrf_chunks.extend(web_search_docs)

        return rewritten_query, rrf_chunks

    def score_cutoff(self, sorted_docs: list[dict]) -> list[dict]:
        # -----------------------------
        # Rerank / TopK 全局常量（不从 state 读取）
        # -----------------------------
        # 动态 TopK 硬上限：最多取前 N 条（<=10）
        RERANK_MAX_TOPK: int = 10
        # 最小 TopK：至少保留前 N 条（>=1，且 <= RERANK_MAX_TOPK）
        RERANK_MIN_TOPK: int = 3  # 总数最少条数

        # 断崖阈值（相对）
        RERANK_GAP_RATIO: float = 0.25
        # 断崖阈值（绝对）
        RERANK_GAP_ABS: float = 0.10

        upper_bound = min(RERANK_MAX_TOPK, len(sorted_docs))
        lower_bound = min(RERANK_MIN_TOPK, upper_bound)

        for idx in range(lower_bound - 1, upper_bound - 1):
            current_score = sorted_docs[idx]['score']
            next_score = sorted_docs[idx + 1]['score']

            abs_diff = abs(current_score - next_score)
            ratio_diff = abs_diff / (current_score + 1e-6)

            if abs_diff >= RERANK_GAP_ABS or ratio_diff >= RERANK_GAP_RATIO:
                sorted_docs = sorted_docs[:idx + 1]
                break
        else:
            # for循环完成之后, 执行这个逻辑
            sorted_docs = sorted_docs[:upper_bound]

        return sorted_docs


if __name__ == '__main__':
    mock_state = {
        "rewritten_query": "请问hak180烫金机怎么使用？",
        "rrf_chunks": [
            {
                "title": "HAK 180 烫金机",
                "file_title": "hak180产品安全手册",
                "content": "二级标题:HAK 180 烫金机\n\n这个chunk所在最近标题的位置:1\n\n上一个chunk的结尾内容:\nchunk内容:## HAK 180 烫金机\n\n产品安全手册（简体中文）\n\n感谢您购买 HAK 180 烫金机。\n\n在使用本设备之前，请先阅读本手册，包括所有预防措施。阅读本手册后，请妥善保管。\n\n有关使用本设备的更多信息，请参阅使用说明书，其可在兄弟 (中国)商业有限公司技 术服务支持网站 http://www.95105369.com/Web/Manuals.aspx 上找到。建议您先通读使用说明书，再使用本设备。\n\n如需获得常见问题解答、故障排除和说明书，请访问\n\nhttp://www.95105369.com。\n\n对于本设备所有者不遵守本指南中规定的说明操作而导致的损害，Brother 不承担任何责任。\n\n•\t对于保养、调整或维修事宜，请联系 Brother 呼叫中心或您当地的Brother 经销商。\n\n•\t如果本设备工作不正常或发生任何错误，请关闭本设备，拔下所有电缆，然后联系 Brother 呼叫中心或您当地的 Brother 经销商。\n\n•\t本文档中提供的信息可能会随时更改，恕不另行通知。\n\n•\t严禁未经授权擅自复制或重制本文档的任何部分或全部内容。",
                "item_name": "BrotherHAK180烫金机",
                "chunk_id": 468318939296846902,
                "score": 1.8114157709919039,
                "source": "local"
            },
            {
                "title": "设备",
                "file_title": "hak180产品安全手册",
                "content": "二级标题:设备\n\n这个chunk所在最近标题的位置:1\n\n上一个chunk的结尾内容:\nchunk内容:## 设备\n\n•\t将本设备放置在平整、水平且稳定的表面上（如桌面），避免震动和冲击。\n\n•\t将本设备放置在通风良好的环境中。\n\n•\t为了防止人员受伤，请谨慎操作，避免将手指放置在图中所示的区域中。\n\n![禁止将手指伸入设备内部指定区域以防受伤](http://192.168.10.100:9000/knowledge-base/upload-images/hak180产品安全手册/c61a7f4e923881679f747508ae309c39dc221685344b068009256b1b3a40cc00.jpg)\n\n![禁止将手指伸入设备指定区域以防受伤](http://192.168.10.100:9000/knowledge-base/upload-images/hak180产品安全手册/5067b2891ca4f761e2874921e0eb433aa742afbf38ca8dc509afecbf0aa6a6b5.jpg)",
                "item_name": "BrotherHAK180烫金机",
                "chunk_id": 468318939296846911,
                "score": 1.7650581761561353,
                "source": "local"
            },
            {
                "title": "设备",
                "file_title": "hak180产品安全手册",
                "content": "二级标题:设备\n\n这个chunk所在最近标题的位置:1\n\n上一个chunk的结尾内容:\nchunk内容:## 设备\n\n•\t请先阅读这本手册，再尝试操作本设备或尝试进行任何维护。不按照这些说明操作可能会提高发生人员受伤或财产损坏（包括火灾、触电、烧伤或窒息所致）的风险。对于本设备所有者不遵守本指南中规定的说明操作而导致的损害，Brother 不承担任何责任。\n\n•\t请勿在未去除所有包装材料的情况下使用本设备，包括本设备内部的任何附加的包装材料。否则可能会产生火灾的风险。\n\n•\t请勿拆解本设备。拆解本设备可能会导致火灾或触电。\n\n•\t请勿尝试 自行维修本设备。打开或拆下盖子可能使您接触到危险电压点以及带来其他风险，并且可能使您的保修失效。对于所有维修事宜，请联系 Brother 呼叫中心或您当地的 Brother 经销商。\n\n•\t请在以 下环境使用本设备：温度保持在 10 °C 和 32 °C 之间，湿度保持在 20% 和 80% 之间，无冷凝。\n\n•\t请勿使本设备受到阳光直射、过热、接触明火、腐蚀性气体、湿气或灰尘。否则可能产生触电、 短路或火灾的风险，从而导致损坏设备和/或导致设备无法运行。\n\n•\t请勿将设备放在加热器、空调、电风扇或水附近。\n\n否则当水（包括加热 空调 通风设备所产生的冷凝水）接触本设备时可能产生短路或火灾的风险。",
                "item_name": "BrotherHAK180烫金机",
                "chunk_id": 468318939296846905,
                "score": 1.763036266449959,
                "source": "local"
            },
            {
                "title": "设备",
                "file_title": "hak180产品安全手册",
                "content": "二级标题:设备\n\n这个chunk所在最近标题的位置:3\n\n上一个chunk的结尾内容:或除臭剂）可能导致塑料盖和/或电缆溶解或分解，从而产生火灾或触电的风险。这些化学品 或其他化学品可能导致本设备故障或褪色。\n\n\nchunk内容:•\t本设备的包装中使用了塑料袋。塑料袋并不是玩具。为避免窒息的危险，请将这些塑料袋远离婴儿和儿童，并正确弃置这些塑料袋。\n\n•\t对于使用起搏器的用户：\n\n本设备可能会产生弱磁场。如果您在本设备附近感觉到起搏器工作不正常，请远离本设备，并立即咨询医生。\n\n•\t使用本设备之后短时间内，本设备的一些内部零件仍 然处于极热状态。打开前盖时，请勿触摸以灰色标记的区域。存在烧伤的风险。先等待设备冷却下来，再触摸设备的内部零件。\n\n![hak180产品安全手册：高温部件防烫伤警示及电源线使用规范](http://192.168.10.100:9000/knowledge-base/upload-images/hak180产品安全手册/f3349cded08d6686a93d0a81b9a64ec1e50d9a82cbb88541b37027f085813a15.jpg)  \n儎⑟ഴḽ䆜઀ᛞ࠽व䀜᪮儎⑟Ⲻ䇴༽䜞ԬȾ",
                "item_name": "BrotherHAK180烫金机",
                "chunk_id": 468318939296846907,
                "score": 1.7457569847031245,
                "source": "local"
            },
            {
                "title": "为设备选择一个安全的位置",
                "file_title": "hak180产品安全手册",
                "content": "二级标题:为设备选择一个安全的位置\n\n这个chunk所在最近标题的位置:1\n\n上一个chunk的结尾内容:\nchunk内容:## 为设备选择一个安全的位置\n\n•\t提起本设备时，请使用双手抓稳本设备的两侧。如果抓住的是进纸托板和出纸盒，它们可能会掉下来。必须通过将双手放在本设备下面来搬运本设备。\n\n![正确与错误的设备搬运姿势示意图](http://192.168.10.100:9000/knowledge-base/upload-images/hak180产品安全手册/cc5ee1ac24ebb2707d40dc7a234a8b243f55f5bf08fabc683859be6fdf096ffa.jpg)  \n确保本设备的任何部位均未伸出设备所在的桌面或支架。特别是当本设备位于桌面、支架等边缘时，请勿让出纸盒打开。确保本设备位于平整、水平且稳定的表面上，避免震动。不遵守这些预防措施可能导致设备跌落，从而导致用户的人身伤害以及设备严重损坏。\n\n![禁止将设备置于桌面边缘且打开出纸盒](http://192.168.10.100:9000/knowledge-base/upload-images/hak180产品安全手册/8e839864036a7326885565163d99117ea943ecd29a656c85e7aa4052a9b9d28d.jpg)",
                "item_name": "BrotherHAK180烫金机",
                "chunk_id": 468318939296846913,
                "score": 1.7405086448698333,
                "source": "local"
            },
            {
                "title": "无标题",
                "file_title": "hak180产品安全手册",
                "content": "这个chunk所在最近标题的位置:1\n\n上一个chunk的结尾内容:\nchunk内容:![HAK 180烫金机产品安全手册条形码](http://192.168.10.100:9000/knowledge-base/upload-images/hak180产品安全手册/677a08ee041965bbbdb6b483d6c17d5aaa36a26b6dc96870a2019f0307b8616f.jpg)  \nD01WD7001-00\n\nSCHN",
                "item_name": "BrotherHAK180烫金机",
                "chunk_id": 468318939296846901,
                "score": 1.7375930681431198,
                "source": "local"
            },
            {
                "title": "电源线",
                "file_title": "hak180产品安全手册",
                "content": "二级标题:电源线\n\n这个chunk所在最近标题的位置:1\n\n上一个chunk的结尾内容:\nchunk内容:## 电源线\n\n•\t本设备通过 AC 220 V-240 V 50/60 Hz 电源供电。\n\n请 勿将本设备连接到直流电源或逆变器（直流交流变换器）。存在火灾或触电的风险。\n\n•\t请勿用湿手触摸插头。这样可能导致触电。如果不确定您拥有哪种类型的电源，请联系合格的电工。\n\n•\t始终确保插头已完全插入。如果电源线磨损或损坏，请勿使用设备或用手触摸电源线。\n\n•\t设备内部有高压电极。\n\n先拔掉电源线，再清洁设备内部。拔出电源线时，不要拉电线，而是捏住插头往外 拔。存在发生火灾、触电或设备故障的风险。\n\n•\t请勿将任何物体压在电源线上。\n\n•\t请勿将本设备放在人们可能踏过电源线的位置。\n\n•\t请勿将本设备放置在会使得拉伸或拉紧电源线的位置 ，否则电源线可能会磨损或损坏。\n\n•\t始终确保插头已完全插入。如果电源线磨损或损坏，请勿使用设备或用手触摸电源线。如果拔出设备的电源插头，请勿触摸损坏 磨损的部分。\n\n•\t请勿让设 备压在电源线上。\n\n•\t请勿在雷暴天气期间使用本设备。存在闪电导致触电的潜在风险。\n\n•\t请勿使用任何非指定的电缆。否则可能导致火灾或人员受伤。必须按照 正确安装。\n\n•\t请勿让任何金属硬件或任何类型的液体落在设备的电源插头上。否则可能导致触电或火灾。",
                "item_name": "BrotherHAK180烫金机",
                "chunk_id": 468318939296846909,
                "score": 1.7203778012788955,
                "source": "local"
            },
            {
                "title": "设备",
                "file_title": "hak180产品安全手册",
                "content": "二级标题:设备\n\n这个chunk所在最近标题的位置:4\n\n上一个chunk的结尾内容:\nchunk内容:![设备高温部件警示及禁止触摸区域示意图](http://192.168.10.100:9000/knowledge-base/upload-images/hak180产品安全手册/501bb8d2d681e4502d87badb15a68939eadfa086d309c3599f1c36b0bc559177.jpg)",
                "item_name": "BrotherHAK180烫金机",
                "chunk_id": 468318939296846908,
                "score": 1.7136379374669133,
                "source": "local"
            },
            {
                "title": "HAK 180 烫金机",
                "file_title": "hak180产品安全手册",
                "content": "二级标题:HAK 180 烫金机\n\n这个chunk所在最近标题的位置:2\n\n上一个chunk的结尾内容:。\n\n•\t本文档中提供的信息可能会随时更改，恕不另行通知。\n\n•\t严禁未经授权擅自复制或重制本文档的任何部分或全部内容。\n\n\nchunk内容:•\t请注意，对于使用通过本设备制作的产品造成的任何损坏或利润损失，或者故障、维修导致的数据消失或更改，或者第三方提出 的任何索赔，我们不承担任何责任。",
                "item_name": "BrotherHAK180烫金机",
                "chunk_id": 468318939296846903,
                "score": 1.5618776398700671,
                "source": "local"
            },
            {
                "title": "设备",
                "file_title": "hak180产品安全手册",
                "content": "二级标题:设备\n\n这个chunk所在最近标题的位置:1\n\n上一个chunk的结尾内容:\nchunk内容:## 设备\n\n如果遵守了操作说明进行操作，但是设备不能正确运行，请仅调整 操作说明中涵盖的控制。错误调整其他控制可能导致损坏并且通常需要合格技术进行全面工作以将本设备恢复到正常操作。Brother不建议使用 Brother 正品烫金膜盒以外的其他品牌烫金膜盒。如果使用与本设备不兼容的耗材导致损坏本设备的任何零件，由此导致的任何维修可能不在保修范围内。",
                "item_name": "BrotherHAK180烫金机",
                "chunk_id": 468318939296846915,
                "score": 1.3977356894169288,
                "source": "local"
            }
        ],
        "web_search_docs": [
            {
                "title": "兄弟(中国)发布烫金机,满足高端文印需求",
                "url": "https://www.thepaper.cn/newsDetail_forward_15996505",
                "snippet": "兄弟(中国)发布烫金机,满足高端文印需求 在邀请函、贺卡或者是红包上轻松呈现出流光溢彩的烫金效果,随着兄弟(中国)23日正式推出的Brother HAK180烫金机而成为现实。 这款体形轻巧烫印机的面市,改变了以往需要到工厂定制才能实现烫金文印品的历史。个人用户只需在纸张介质上使用激光打印机打印好内容,再放入兄弟烫金机HAK180中,即可实现一键烫印,省去繁杂的软件编辑、电脑连接过程。 近年来,文印市场逐渐呈现精细化发展趋势,拥有核心技术的兄弟烫金机HAK180则恰好是可以满足高端文印需求的一款产品。为了让更多用户体验这股“金色能量”,兄弟(中国)携全新烫金机Brother HAK180,以“引领鎏金岁月,创新成就JIN界”为发布会主题,于12月23日,带来了一场线上多平台直播,线下多地共享的双线联动发布会,向中国用户全方位展示烫金之美。 兄弟(中国)商业有限公司董事长兼总裁尹炳新先生在当天的发布会上介绍,相关数据显示,中国是全球烫金文印市场规模最大的国家,占据全球60%的份额,其次是德国和日本。基于中国庞大的市场潜力,为满足高端文印市场对于个性化烫金需求,解决繁杂制版工序及成本高企等诸多困扰,兄弟集团决定在中国市场推出 “便捷使用”和“精品烫印”于一身的烫金机。 尹炳新先生介绍,以往实现烫金效果需要把产品送往工厂,交由大型专业设备进行处理。而兄弟(中国)推出的HAK180烫金机体积小巧,无需制版,避免环境污染,操作简便。据了解,这款产品可瞬间实现烫金效果,可广泛适用于各类场景,如精美邀请函,高档菜单与座位卡,激励学子的金色奖状等各类需要高品质,个性化定制的场景,满足学校,商务公司,高档宴会酒店等多用户需求。 据介绍,HAK180烫金机广泛支持各类纸张,胜任各式复杂情况,可高效便捷地为用户完成繁重任务。另外,HAK180烫金机在无版烫金与读秒烫金的基础之上,采用“金”“银”“红”三色烫金薄膜设计,让烫金效果达到纤毫毕现的水准。无论是纤细线条,亦或微小字体,都能精准呈现。清晰的烫印效果,杜绝棱角、毛边、断线、模糊等恼人问题。同时烫印的内容耐得住长期保存,即便用手指刮抠也不会掉色或脱落,高品质烫金将为用户带来无可替代的体验。 整体上,做为凝聚着百年企业核心技术的Brother HAK180烫金机集合了兄弟集团始终坚持的高质量与高性价比的产品力,赋能其“无版烫印”、“多页连续烫金”、“纤毫毕现品质呈现”多重创新技术,以提升用户烫金体验,为高端文印与商务交流提供更优质、更创新的 解决方案。 顺应市场的需求,兄弟(中国)面向中国市场",
                "source": "web"
            },
            {
                "title": "HAK180",
                "url": "https://www.brother.cn/hak/hak180",
                "snippet": "HAK180 烫金机 零售价 面议 最大15PPM烫金速度  可选7PPM烫金速度  无版烫印  配备最大44页标准ADF进纸器  支持省膜模式  10字符x2行LCD液晶屏  HAK180烫金机,凭借其高速、高品质、以及出色的细节小字烫印效果,成为定制化专属机型。可烫印90g/m²~350g/m²的A4各类型纸张,支持各类广泛的应用领域。 高效、稳定的进纸结构 配备44页标准ADF进纸器,支持90g/m²~350g/m²的各类纸张(普通纸、薄纸、再生纸、厚纸等),进纸通道结构稳定可靠,支持连续烫印。 * 350g/m²支持12页自动进纸 * 最大支持44页进纸容量(90g/m²)烫印面朝下 高速连续烫金 HAK180针对不同厚 度、介质的纸张提供两种可选烫金速度。15ppm满足普通规格纸张的高效烫金需求,7ppm适合稍厚纸张的烫金。 10字符×2行LCD液晶屏 10字符×2行LCD液晶屏,2个自定义按键,操作直观,方便快捷。 产品规 格  一般参数  正常工作环境(温度): 10 ~ 32 摄氏度(50 ~ 90 华氏度) 正常工作环境(相对湿度): 20 % ~ 80 % 机器尺寸: W 384.2mm×D 330.2mm×H 356.2mm 重量(含包装箱): 16.9kg 电源: 220~240 V 消费电力(烫印中): 少于340W 消费电力(待机中): 少于7W 消费电力(关机): 少于0.04W LCD液晶屏尺寸: 48.0mm×10.9mm 节省烫金膜功能: 支持(在省膜模式中“跳过”和“中间”功能, 仅适用全幅烫金膜盒) 烫印参数  最大烫印速度 (A4): 最高达15 ppm 可选烫印速度(A4): 7 ppm 视频 烫金机-HAK180-烫印速度调整-7PPM 烫金机-HAK180-安装耗材 烫金机-HAK180-更换耗材 兄弟机床公众号 数码打印机公众号 创意标签P-touch Candy",
                "source": "web"
            },
            {
                "title": "无版烫金+连续烫印?论一台优秀烫金机的自我修养!兄弟HAK180烫金机评测",
                "url": "https://www.163.com/dy/article/HBO219SA05118VMB.html",
                "snippet": "兄弟HAK180烫金机评测 作为高端文印设备,烫金机并没有太高的“知名度”,大部分人可能从未听过,而且它售价较为昂贵,一般只会在高端文印店(或工厂)才能见到。不过,由烫金机实现的作品,相信大家都接触过,甚至是“得到过”,比如入户门上的金色福字、商务会议的邀请函、礼品店/花店的祝福贺卡、高档酒店/餐厅的菜单酒单,以及代表荣誉和认可的获奖奖状等等。 去年底,Brother在进博会发布HAK180烫金机,作为Brother旗下新品类,烫金机是其在打印机、一体机、标签机、条码机、扫描仪等之后,布局的又一办公文印设备品。作为一款主要针对高端文印店推出的产品,HAK180的问世,令烫金品在文印店中即可完成,无需再像以前跑到制作工厂去定制,简化流程提升效率;对于烫金需求方而言,也就是企业、学校、花店等,无需频繁的确认,减少了制作流程,向文印店提出需求后,在 文印店中就可完成,简单的烫金需求甚至可以做到“立等可取”,一改了传统需要在“需求方,供应商,制作工厂”间频繁沟通、确认、修改的流程,HAK180让烫金流程更省时、更省力、更省沟通。那么,烫金机究竟如何工作,长相又如何,且随着笔者一同去认识这款产品! 我们先观看一段视频,了解下烫金机的用途 细分市场需求,灵巧机身,任性安置 近年来,随着文印市场逐渐呈现精细化发展趋势,高端文印的需求 逐渐增加,大势之下兄弟HAK180烫金机应运而生。烫金机,顾名思义,可以简单理解为,在纸张表面烫印一层金色,当然,此“金”非彼“金”,就像上面提到的奖状、春联,只是在技术上有些特殊。 第一眼看到兄 弟HAK180烫金机,如非提前知晓这是一台烫金机,可能会让人误以为是一台馈纸式扫描仪,毕竟从外观来看,兄弟HAK180烫金机与扫描仪有着相似的外观,尤其是进纸、出纸托盘的设计,都有着一定相似度。  机身顶部的进纸托盘可以存放大量用于烫印的纸张,HAK180支持多种纸张质量规格,像办公常用70g/m²的A4纸张,以及更厚更重350g/m²的A4纸张都是可以正常实现烫印的,其中90g/m²纸张可以同时存放44张,350g/m²纸张可以同时存放12张,并可实现纸张自动、连续进纸烫金(如文章起始视频所示),这得益于其采用的“多页连续烫金”技术,可以处理批量烫印任务。 兄弟HAK180烫金机还支持“无版烫金”,整个烫印过程无需提前制版。如上图所示,比如我们需要制作一张用于表彰员工,或是学生的荣誉证书/奖状,只需提前制作一张《荣誉证书》的样式(设计图),利用激光打印机,将样式内容打印出来,再将带有内容的 一侧,面向HAK180放入到进纸托盘中,点击启动键",
                "source": "web"
            },
            {
                "title": "高速高品质 定制化专属,兄弟HAK180烫金机让你的文印店抢占先机",
                "url": "https://www.163.com/dy/article/HC5ISR9H05119GO7.html",
                "snippet": "高速高品质 定制化专属,兄弟HAK180烫金机让你的文印店抢占先机   高速高品质 定制化专属,兄弟HAK180烫金机让你的文印店抢占先机   说起打印店,我们对这个现象肯定记 忆深刻:打字、复印、打印 ……曾几何时,小型文印店如雨后春笋般地布满城市的大些小巷。后来随着家用打印机和无纸化办公的普及,文印店的用户不断遭受流失,开始面临着极大的生存危机:同行价格战、人力成本、维护成本高、设备功能落后……原来必须去文印店的打印工作,现在家里就能完成,普通的打印已经在文印店中失去优势,因此文印店业主不得不拓展更多的业务, 谋求转型升级。   随着社会经济的不断发展,人们对商品包装外观样式以及质量要求也越来越高。烫印工艺以其色彩亮丽、图案清晰、美观醒目的装饰效果被人们喜爱。这种烫印工艺也逐渐延申到了文印行业,在传统黑白、彩色的文印产品上加入烫印的点缀,对于客户来说,产品的美观、档次程度都能得到有效地提升,而对于文印店来说,客户的回顾率以及利润也能得到突破性地增长。   因此,文印店也开始“卷起来”了。除了满足客户最基本的黑白、彩色打印,如今金色奖状,烫金的升职任命书、贺卡的制作需求也越来越多了,越来越多的用户开始选择定制烫印。但是传统的烫金机都需要制版,不仅麻烦,还费时,顾客也等不了这么长时间。所以,怎样才能既方便又快捷地就能拥有烫印的效果呢?     Brother兄弟(以下简称“兄弟”)推出的HAK180烫金机凭借其高速、高品质、以及出色的细节小字烫印效果,成为定制化专属机型,专业实力为邀请函、贺卡、请柬等个性化定制需求提供了更多的便利,最终帮助用户实现产业升级、促进文印服务往高端化发展。同时,无版烫印、支持省膜模式,大幅降低运营成本,使用效率更高,免去使用者的顾虑,为业务保驾护航。   紧凑体积,简约外观 外观方面,这款HAK180烫金机产品给人以沉稳扎实的感觉。产品颜色为黑色,磨砂的质感使得产品在使用时不易留下指纹,更具耐磨性。一体机整体观感棱角分明,但机身 边角处均采用了圆润的设计,很大程度避免了用户在使用时发生不必要的磕碰。烫金机正面采用斜面设计,使得操作更加便捷舒适,摁键设置不用半蹲操作。并且外观还获得了2021年的日本GOOD DESIGN奖。       操作面板采用经济性和操作性适中的10字符*2行LCD液晶屏+按键的方式,操作直观,方便快捷。对于打印店快速、效率的工作环境来说,简洁明了的直观显示非常友好。",
                "source": "web"
            },
            {
                "title": "第四届进博会圆满收官 Brother首发新品成展位亮点",
                "url": "https://city.cri.cn/chinanews/20211116/c464b2cc-34d2-14b4-1827-219896c878ea.html",
                "snippet": "此外,Brother在这场盛大的展览中首发了最新HAK180烫金机,协同一系列明星解决方案,为进博会增益添彩。 Brother HAK180烫金机于第四届进博会首发 Brother HAK180 -- 烫金机中的优等生 11月8日,Brother首发了HAK180烫金机。HAK180烫金机搭载“多页连续烫金”技术和“无版烫金”技术,大大简化了烫印流程。与此同时,HAK180烫金机的烫印效果也十分优异,无论是纤细的线 条,还是微小的字体,都能清晰呈现。HAK180烫金机所具备的高速、高品质、定制化的属性赢得了大量关注,成了Brother展台中稳居C位的又一拳头产品。 此外,Brother根据当下用户的使用习惯呈现了众多个性化、智能化与便捷化的生活与工作方式,比如:可实现医院自助终端设备各项单据、自助报告单等快速输出的Brother 高速双面网络激光打印机;方便移动执法、移动商务人群的Brother A4幅面便携式 打印机PJ系列;为物流仓储、零售等多行业的标识标记难题提供解决方案的热敏/热转印标签打印机及集团旗下知名品牌多米诺(Domino)的喷码机;开启“私人订制”的商用绣花机等。",
                "source": "web"
            }
        ]
    }

    node_rerank = NodeRerank()
    result = node_rerank(mock_state)
    logger.info(parse_json(result))
