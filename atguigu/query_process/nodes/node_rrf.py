'''
@Author  :61022
@Time    :2026/8/9
@Desc    :
'''
from typing import List, Dict

from atguigu.query_process.base import NodeBase
from atguigu.query_process.state import QueryGraphState
from atguigu.tool.json_format_util import parse_json
from atguigu.tool.logger import logger


class NodeRrf(NodeBase):
    """
    节点功能：(算法粗排) 将多路召回的结果（向量、HyDE、Web）进行加权融合排序。
    Reciprocal Rank Fusion
    """

    # 覆盖基类的 name 属性，标识节点名称
    @property
    def name(self) -> str:
        return "node_rrf"

    def process(self, state: QueryGraphState) -> QueryGraphState:
        """
        节点逻辑
        :param state: 工作流状态对象
        :return: 更新后的状态对象
        """

        embedding_chunks = state.get("embedding_chunks")
        if not embedding_chunks:
            raise ValueError("embedding_chunks 不能为空")
        hyde_embedding_chunks = state.get("hyde_embedding_chunks")
        if not hyde_embedding_chunks:
            raise ValueError("hyde_embedding_chunks 不能为空")

        # 通路权重
        chunks_weight = [
            (embedding_chunks, 1),
            (hyde_embedding_chunks, 1)
        ]
        # score

        # {chunk_id , chunk}
        rrf_chunks:Dict[str, Dict] = {}

        for chunks, weight in chunks_weight:
            for idx, chunk in enumerate(chunks, start=1):
                chunk_id = chunk.get("chunk_id")
                score = chunk.get("score")
                rrf_score = score + weight / (60 + idx)
                if chunk_id in rrf_chunks:
                    rrf_chunks[chunk_id]['score'] += rrf_score
                else:
                    chunk['score'] = rrf_score
                    rrf_chunks[chunk_id] = chunk
        rrf_result_chunks = sorted(rrf_chunks.values(), key=lambda d: d.get("score"), reverse=True)[:10]
        return {"rrf_chunks":rrf_result_chunks}


if __name__ == '__main__':
    node = NodeRrf()

    init_state = {
        "embedding_chunks": [
            {
                "title": "HAK 180 烫金机",
                "file_title": "hak180产品安全手册",
                "content": "二级标题:HAK 180 烫金机\n\n这个chunk所在最近标题的位置:1\n\n上一个chunk的结尾内容:\nchunk内容:## HAK 180 烫金机\n\n产品安全手册（简体中文）\n\n感谢您购买 HAK 180 烫金机。\n\n在使用本设备之前，请先阅读本手册，包括所有预防措施。阅读本手册后，请妥善保管。\n\n有关使用本设备的更多信息，请参阅使用说明书，其可在兄弟 (中国)商业有限公司技 术服务支持网站 http://www.95105369.com/Web/Manuals.aspx 上找到。建议您先通读使用说明书，再使用本设备。\n\n如需获得常见问题解答、故障排除和说明书，请访问\n\nhttp://www.95105369.com。\n\n对于本设备所有者不遵守本指南中规定的说明操作而导致的损害，Brother 不承担任何责任。\n\n•\t对于保养、调整或维修事宜，请联系 Brother 呼叫中心或您当地的Brother 经销商。\n\n•\t如果本设备工作不正常或发生任何错误，请关闭本设备，拔下所有电缆，然后联系 Brother 呼叫中心或您当地的 Brother 经销商。\n\n•\t本文档中提供的信息可能会随时更改，恕不另行通知。\n\n•\t严禁未经授权擅自复制或重制本文档的任何部分或全部内容。",
                "item_name": "BrotherHAK180烫金机",
                "chunk_id": 468318939296846902,
                "score": 0.8804964423179626,
                "source": "local"
            },
            {
                "title": "设备",
                "file_title": "hak180产品安全手册",
                "content": "二级标题:设备\n\n这个chunk所在最近标题的位置:1\n\n上一个chunk的结尾内容:\nchunk内容:## 设备\n\n•\t将本设备放置在平整、水平且稳定的表面上（如桌面），避免震动和冲击。\n\n•\t将本设备放置在通风良好的环境中。\n\n•\t为了防止人员受伤，请谨慎操作，避免将手指放置在图中所示的区域中。\n\n![禁止将手指伸入设备内部指定区域以防受伤](http://192.168.10.100:9000/knowledge-base/upload-images/hak180产品安全手册/c61a7f4e923881679f747508ae309c39dc221685344b068009256b1b3a40cc00.jpg)\n\n![禁止将手指伸入设备指定区域以防受伤](http://192.168.10.100:9000/knowledge-base/upload-images/hak180产品安全手册/5067b2891ca4f761e2874921e0eb433aa742afbf38ca8dc509afecbf0aa6a6b5.jpg)",
                "item_name": "BrotherHAK180烫金机",
                "chunk_id": 468318939296846911,
                "score": 0.8632898330688477,
                "source": "local"
            },
            {
                "title": "设备",
                "file_title": "hak180产品安全手册",
                "content": "二级标题:设备\n\n这个chunk所在最近标题的位置:3\n\n上一个chunk的结尾内容:或除臭剂）可能导致塑料盖和/或电缆溶解或分解，从而产生火灾或触电的风险。这些化学品 或其他化学品可能导致本设备故障或褪色。\n\n\nchunk内容:•\t本设备的包装中使用了塑料袋。塑料袋并不是玩具。为避免窒息的危险，请将这些塑料袋远离婴儿和儿童，并正确弃置这些塑料袋。\n\n•\t对于使用起搏器的用户：\n\n本设备可能会产生弱磁场。如果您在本设备附近感觉到起搏器工作不正常，请远离本设备，并立即咨询医生。\n\n•\t使用本设备之后短时间内，本设备的一些内部零件仍 然处于极热状态。打开前盖时，请勿触摸以灰色标记的区域。存在烧伤的风险。先等待设备冷却下来，再触摸设备的内部零件。\n\n![hak180产品安全手册：高温部件防烫伤警示及电源线使用规范](http://192.168.10.100:9000/knowledge-base/upload-images/hak180产品安全手册/f3349cded08d6686a93d0a81b9a64ec1e50d9a82cbb88541b37027f085813a15.jpg)  \n儎⑟ഴḽ䆜઀ᛞ࠽व䀜᪮儎⑟Ⲻ䇴༽䜞ԬȾ",
                "item_name": "BrotherHAK180烫金机",
                "chunk_id": 468318939296846907,
                "score": 0.8514158129692078,
                "source": "local"
            },
            {
                "title": "设备",
                "file_title": "hak180产品安全手册",
                "content": "二级标题:设备\n\n这个chunk所在最近标题的位置:1\n\n上一个chunk的结尾内容:\nchunk内容:## 设备\n\n•\t请先阅读这本手册，再尝试操作本设备或尝试进行任何维护。不按照这些说明操作可能会提高发生人员受伤或财产损坏（包括火灾、触电、烧伤或窒息所致）的风险。对于本设备所有者不遵守本指南中规定的说明操作而导致的损害，Brother 不承担任何责任。\n\n•\t请勿在未去除所有包装材料的情况下使用本设备，包括本设备内部的任何附加的包装材料。否则可能会产生火灾的风险。\n\n•\t请勿拆解本设备。拆解本设备可能会导致火灾或触电。\n\n•\t请勿尝试 自行维修本设备。打开或拆下盖子可能使您接触到危险电压点以及带来其他风险，并且可能使您的保修失效。对于所有维修事宜，请联系 Brother 呼叫中心或您当地的 Brother 经销商。\n\n•\t请在以 下环境使用本设备：温度保持在 10 °C 和 32 °C 之间，湿度保持在 20% 和 80% 之间，无冷凝。\n\n•\t请勿使本设备受到阳光直射、过热、接触明火、腐蚀性气体、湿气或灰尘。否则可能产生触电、 短路或火灾的风险，从而导致损坏设备和/或导致设备无法运行。\n\n•\t请勿将设备放在加热器、空调、电风扇或水附近。\n\n否则当水（包括加热 空调 通风设备所产生的冷凝水）接触本设备时可能产生短路或火灾的风险。",
                "item_name": "BrotherHAK180烫金机",
                "chunk_id": 468318939296846905,
                "score": 0.8513296842575073,
                "source": "local"
            },
            {
                "title": "HAK 180 烫金机",
                "file_title": "hak180产品安全手册",
                "content": "二级标题:HAK 180 烫金机\n\n这个chunk所在最近标题的位置:2\n\n上一个chunk的结尾内容:。\n\n•\t本文档中提供的信息可能会随时更改，恕不另行通知。\n\n•\t严禁未经授权擅自复制或重制本文档的任何部分或全部内容。\n\n\nchunk内容:•\t请注意，对于使用通过本设备制作的产品造成的任何损坏或利润损失，或者故障、维修导致的数据消失或更改，或者第三方提出 的任何索赔，我们不承担任何责任。",
                "item_name": "BrotherHAK180烫金机",
                "chunk_id": 468318939296846903,
                "score": 0.85114586353302,
                "source": "local"
            },
            {
                "title": "为设备选择一个安全的位置",
                "file_title": "hak180产品安全手册",
                "content": "二级标题:为设备选择一个安全的位置\n\n这个chunk所在最近标题的位置:1\n\n上一个chunk的结尾内容:\nchunk内容:## 为设备选择一个安全的位置\n\n•\t提起本设备时，请使用双手抓稳本设备的两侧。如果抓住的是进纸托板和出纸盒，它们可能会掉下来。必须通过将双手放在本设备下面来搬运本设备。\n\n![正确与错误的设备搬运姿势示意图](http://192.168.10.100:9000/knowledge-base/upload-images/hak180产品安全手册/cc5ee1ac24ebb2707d40dc7a234a8b243f55f5bf08fabc683859be6fdf096ffa.jpg)  \n确保本设备的任何部位均未伸出设备所在的桌面或支架。特别是当本设备位于桌面、支架等边缘时，请勿让出纸盒打开。确保本设备位于平整、水平且稳定的表面上，避免震动。不遵守这些预防措施可能导致设备跌落，从而导致用户的人身伤害以及设备严重损坏。\n\n![禁止将设备置于桌面边缘且打开出纸盒](http://192.168.10.100:9000/knowledge-base/upload-images/hak180产品安全手册/8e839864036a7326885565163d99117ea943ecd29a656c85e7aa4052a9b9d28d.jpg)",
                "item_name": "BrotherHAK180烫金机",
                "chunk_id": 468318939296846913,
                "score": 0.8498690724372864,
                "source": "local"
            },
            {
                "title": "无标题",
                "file_title": "hak180产品安全手册",
                "content": "这个chunk所在最近标题的位置:1\n\n上一个chunk的结尾内容:\nchunk内容:![HAK 180烫金机产品安全手册条形码](http://192.168.10.100:9000/knowledge-base/upload-images/hak180产品安全手册/677a08ee041965bbbdb6b483d6c17d5aaa36a26b6dc96870a2019f0307b8616f.jpg)  \nD01WD7001-00\n\nSCHN",
                "item_name": "BrotherHAK180烫金机",
                "chunk_id": 468318939296846901,
                "score": 0.8464744687080383,
                "source": "local"
            },
            {
                "title": "电源线",
                "file_title": "hak180产品安全手册",
                "content": "二级标题:电源线\n\n这个chunk所在最近标题的位置:1\n\n上一个chunk的结尾内容:\nchunk内容:## 电源线\n\n•\t本设备通过 AC 220 V-240 V 50/60 Hz 电源供电。\n\n请 勿将本设备连接到直流电源或逆变器（直流交流变换器）。存在火灾或触电的风险。\n\n•\t请勿用湿手触摸插头。这样可能导致触电。如果不确定您拥有哪种类型的电源，请联系合格的电工。\n\n•\t始终确保插头已完全插入。如果电源线磨损或损坏，请勿使用设备或用手触摸电源线。\n\n•\t设备内部有高压电极。\n\n先拔掉电源线，再清洁设备内部。拔出电源线时，不要拉电线，而是捏住插头往外 拔。存在发生火灾、触电或设备故障的风险。\n\n•\t请勿将任何物体压在电源线上。\n\n•\t请勿将本设备放在人们可能踏过电源线的位置。\n\n•\t请勿将本设备放置在会使得拉伸或拉紧电源线的位置 ，否则电源线可能会磨损或损坏。\n\n•\t始终确保插头已完全插入。如果电源线磨损或损坏，请勿使用设备或用手触摸电源线。如果拔出设备的电源插头，请勿触摸损坏 磨损的部分。\n\n•\t请勿让设 备压在电源线上。\n\n•\t请勿在雷暴天气期间使用本设备。存在闪电导致触电的潜在风险。\n\n•\t请勿使用任何非指定的电缆。否则可能导致火灾或人员受伤。必须按照 正确安装。\n\n•\t请勿让任何金属硬件或任何类型的液体落在设备的电源插头上。否则可能导致触电或火灾。",
                "item_name": "BrotherHAK180烫金机",
                "chunk_id": 468318939296846909,
                "score": 0.8400719165802002,
                "source": "local"
            },
            {
                "title": "设备",
                "file_title": "hak180产品安全手册",
                "content": "二级标题:设备\n\n这个chunk所在最近标题的位置:4\n\n上一个chunk的结尾内容:\nchunk内容:![设备高温部件警示及禁止触摸区域示意图](http://192.168.10.100:9000/knowledge-base/upload-images/hak180产品安全手册/501bb8d2d681e4502d87badb15a68939eadfa086d309c3599f1c36b0bc559177.jpg)",
                "item_name": "BrotherHAK180烫金机",
                "chunk_id": 468318939296846908,
                "score": 0.8359363675117493,
                "source": "local"
            },
            {
                "title": "设备",
                "file_title": "hak180产品安全手册",
                "content": "二级标题:设备\n\n这个chunk所在最近标题的位置:1\n\n上一个chunk的结尾内容:\nchunk内容:## 设备\n\n如果遵守了操作说明进行操作，但是设备不能正确运行，请仅调整 操作说明中涵盖的控制。错误调整其他控制可能导致损坏并且通常需要合格技术进行全面工作以将本设备恢复到正常操作。Brother不建议使用 Brother 正品烫金膜盒以外的其他品牌烫金膜盒。如果使用与本设备不兼容的耗材导致损坏本设备的任何零件，由此导致的任何维修可能不在保修范围内。",
                "item_name": "BrotherHAK180烫金机",
                "chunk_id": 468318939296846915,
                "score": 0.6837016940116882,
                "source": "local"
            }
        ],
        "hyde_embedding_chunks": [
            {
                "title": "HAK 180 烫金机",
                "file_title": "hak180产品安全手册",
                "content": "二级标题:HAK 180 烫金机\n\n这个chunk所在最近标题的位置:1\n\n上一个chunk的结尾内容:\nchunk内容:## HAK 180 烫金机\n\n产品安全手册（简体中文）\n\n感谢您购买 HAK 180 烫金机。\n\n在使用本设备之前，请先阅读本手册，包括所有预防措施。阅读本手册后，请妥善保管。\n\n有关使用本设备的更多信息，请参阅使用说明书，其可在兄弟 (中国)商业有限公司技 术服务支持网站 http://www.95105369.com/Web/Manuals.aspx 上找到。建议您先通读使用说明书，再使用本设备。\n\n如需获得常见问题解答、故障排除和说明书，请访问\n\nhttp://www.95105369.com。\n\n对于本设备所有者不遵守本指南中规定的说明操作而导致的损害，Brother 不承担任何责任。\n\n•\t对于保养、调整或维修事宜，请联系 Brother 呼叫中心或您当地的Brother 经销商。\n\n•\t如果本设备工作不正常或发生任何错误，请关闭本设备，拔下所有电缆，然后联系 Brother 呼叫中心或您当地的 Brother 经销商。\n\n•\t本文档中提供的信息可能会随时更改，恕不另行通知。\n\n•\t严禁未经授权擅自复制或重制本文档的任何部分或全部内容。",
                "item_name": "BrotherHAK180烫金机",
                "chunk_id": 468318939296846902,
                "score": 0.8981324434280396,
                "source": "local"
            },
            {
                "title": "设备",
                "file_title": "hak180产品安全手册",
                "content": "二级标题:设备\n\n这个chunk所在最近标题的位置:1\n\n上一个chunk的结尾内容:\nchunk内容:## 设备\n\n•\t请先阅读这本手册，再尝试操作本设备或尝试进行任何维护。不按照这些说明操作可能会提高发生人员受伤或财产损坏（包括火灾、触电、烧伤或窒息所致）的风险。对于本设备所有者不遵守本指南中规定的说明操作而导致的损害，Brother 不承担任何责任。\n\n•\t请勿在未去除所有包装材料的情况下使用本设备，包括本设备内部的任何附加的包装材料。否则可能会产生火灾的风险。\n\n•\t请勿拆解本设备。拆解本设备可能会导致火灾或触电。\n\n•\t请勿尝试 自行维修本设备。打开或拆下盖子可能使您接触到危险电压点以及带来其他风险，并且可能使您的保修失效。对于所有维修事宜，请联系 Brother 呼叫中心或您当地的 Brother 经销商。\n\n•\t请在以 下环境使用本设备：温度保持在 10 °C 和 32 °C 之间，湿度保持在 20% 和 80% 之间，无冷凝。\n\n•\t请勿使本设备受到阳光直射、过热、接触明火、腐蚀性气体、湿气或灰尘。否则可能产生触电、 短路或火灾的风险，从而导致损坏设备和/或导致设备无法运行。\n\n•\t请勿将设备放在加热器、空调、电风扇或水附近。\n\n否则当水（包括加热 空调 通风设备所产生的冷凝水）接触本设备时可能产生短路或火灾的风险。",
                "item_name": "BrotherHAK180烫金机",
                "chunk_id": 468318939296846905,
                "score": 0.8799525499343872,
                "source": "local"
            },
            {
                "title": "设备",
                "file_title": "hak180产品安全手册",
                "content": "二级标题:设备\n\n这个chunk所在最近标题的位置:1\n\n上一个chunk的结尾内容:\nchunk内容:## 设备\n\n•\t将本设备放置在平整、水平且稳定的表面上（如桌面），避免震动和冲击。\n\n•\t将本设备放置在通风良好的环境中。\n\n•\t为了防止人员受伤，请谨慎操作，避免将手指放置在图中所示的区域中。\n\n![禁止将手指伸入设备内部指定区域以防受伤](http://192.168.10.100:9000/knowledge-base/upload-images/hak180产品安全手册/c61a7f4e923881679f747508ae309c39dc221685344b068009256b1b3a40cc00.jpg)\n\n![禁止将手指伸入设备指定区域以防受伤](http://192.168.10.100:9000/knowledge-base/upload-images/hak180产品安全手册/5067b2891ca4f761e2874921e0eb433aa742afbf38ca8dc509afecbf0aa6a6b5.jpg)",
                "item_name": "BrotherHAK180烫金机",
                "chunk_id": 468318939296846911,
                "score": 0.8697662949562073,
                "source": "local"
            },
            {
                "title": "设备",
                "file_title": "hak180产品安全手册",
                "content": "二级标题:设备\n\n这个chunk所在最近标题的位置:3\n\n上一个chunk的结尾内容:或除臭剂）可能导致塑料盖和/或电缆溶解或分解，从而产生火灾或触电的风险。这些化学品 或其他化学品可能导致本设备故障或褪色。\n\n\nchunk内容:•\t本设备的包装中使用了塑料袋。塑料袋并不是玩具。为避免窒息的危险，请将这些塑料袋远离婴儿和儿童，并正确弃置这些塑料袋。\n\n•\t对于使用起搏器的用户：\n\n本设备可能会产生弱磁场。如果您在本设备附近感觉到起搏器工作不正常，请远离本设备，并立即咨询医生。\n\n•\t使用本设备之后短时间内，本设备的一些内部零件仍 然处于极热状态。打开前盖时，请勿触摸以灰色标记的区域。存在烧伤的风险。先等待设备冷却下来，再触摸设备的内部零件。\n\n![hak180产品安全手册：高温部件防烫伤警示及电源线使用规范](http://192.168.10.100:9000/knowledge-base/upload-images/hak180产品安全手册/f3349cded08d6686a93d0a81b9a64ec1e50d9a82cbb88541b37027f085813a15.jpg)  \n儎⑟ഴḽ䆜઀ᛞ࠽व䀜᪮儎⑟Ⲻ䇴༽䜞ԬȾ",
                "item_name": "BrotherHAK180烫金机",
                "chunk_id": 468318939296846907,
                "score": 0.8628431558609009,
                "source": "local"
            },
            {
                "title": "无标题",
                "file_title": "hak180产品安全手册",
                "content": "这个chunk所在最近标题的位置:1\n\n上一个chunk的结尾内容:\nchunk内容:![HAK 180烫金机产品安全手册条形码](http://192.168.10.100:9000/knowledge-base/upload-images/hak180产品安全手册/677a08ee041965bbbdb6b483d6c17d5aaa36a26b6dc96870a2019f0307b8616f.jpg)  \nD01WD7001-00\n\nSCHN",
                "item_name": "BrotherHAK180烫金机",
                "chunk_id": 468318939296846901,
                "score": 0.8608086109161377,
                "source": "local"
            },
            {
                "title": "为设备选择一个安全的位置",
                "file_title": "hak180产品安全手册",
                "content": "二级标题:为设备选择一个安全的位置\n\n这个chunk所在最近标题的位置:1\n\n上一个chunk的结尾内容:\nchunk内容:## 为设备选择一个安全的位置\n\n•\t提起本设备时，请使用双手抓稳本设备的两侧。如果抓住的是进纸托板和出纸盒，它们可能会掉下来。必须通过将双手放在本设备下面来搬运本设备。\n\n![正确与错误的设备搬运姿势示意图](http://192.168.10.100:9000/knowledge-base/upload-images/hak180产品安全手册/cc5ee1ac24ebb2707d40dc7a234a8b243f55f5bf08fabc683859be6fdf096ffa.jpg)  \n确保本设备的任何部位均未伸出设备所在的桌面或支架。特别是当本设备位于桌面、支架等边缘时，请勿让出纸盒打开。确保本设备位于平整、水平且稳定的表面上，避免震动。不遵守这些预防措施可能导致设备跌落，从而导致用户的人身伤害以及设备严重损坏。\n\n![禁止将设备置于桌面边缘且打开出纸盒](http://192.168.10.100:9000/knowledge-base/upload-images/hak180产品安全手册/8e839864036a7326885565163d99117ea943ecd29a656c85e7aa4052a9b9d28d.jpg)",
                "item_name": "BrotherHAK180烫金机",
                "chunk_id": 468318939296846913,
                "score": 0.8603365421295166,
                "source": "local"
            },
            {
                "title": "电源线",
                "file_title": "hak180产品安全手册",
                "content": "二级标题:电源线\n\n这个chunk所在最近标题的位置:1\n\n上一个chunk的结尾内容:\nchunk内容:## 电源线\n\n•\t本设备通过 AC 220 V-240 V 50/60 Hz 电源供电。\n\n请 勿将本设备连接到直流电源或逆变器（直流交流变换器）。存在火灾或触电的风险。\n\n•\t请勿用湿手触摸插头。这样可能导致触电。如果不确定您拥有哪种类型的电源，请联系合格的电工。\n\n•\t始终确保插头已完全插入。如果电源线磨损或损坏，请勿使用设备或用手触摸电源线。\n\n•\t设备内部有高压电极。\n\n先拔掉电源线，再清洁设备内部。拔出电源线时，不要拉电线，而是捏住插头往外 拔。存在发生火灾、触电或设备故障的风险。\n\n•\t请勿将任何物体压在电源线上。\n\n•\t请勿将本设备放在人们可能踏过电源线的位置。\n\n•\t请勿将本设备放置在会使得拉伸或拉紧电源线的位置 ，否则电源线可能会磨损或损坏。\n\n•\t始终确保插头已完全插入。如果电源线磨损或损坏，请勿使用设备或用手触摸电源线。如果拔出设备的电源插头，请勿触摸损坏 磨损的部分。\n\n•\t请勿让设 备压在电源线上。\n\n•\t请勿在雷暴天气期间使用本设备。存在闪电导致触电的潜在风险。\n\n•\t请勿使用任何非指定的电缆。否则可能导致火灾或人员受伤。必须按照 正确安装。\n\n•\t请勿让任何金属硬件或任何类型的液体落在设备的电源插头上。否则可能导致触电或火灾。",
                "item_name": "BrotherHAK180烫金机",
                "chunk_id": 468318939296846909,
                "score": 0.8506746292114258,
                "source": "local"
            },
            {
                "title": "设备",
                "file_title": "hak180产品安全手册",
                "content": "二级标题:设备\n\n这个chunk所在最近标题的位置:4\n\n上一个chunk的结尾内容:\nchunk内容:![设备高温部件警示及禁止触摸区域示意图](http://192.168.10.100:9000/knowledge-base/upload-images/hak180产品安全手册/501bb8d2d681e4502d87badb15a68939eadfa086d309c3599f1c36b0bc559177.jpg)",
                "item_name": "BrotherHAK180烫金机",
                "chunk_id": 468318939296846908,
                "score": 0.8485029339790344,
                "source": "local"
            },
            {
                "title": "设备",
                "file_title": "hak180产品安全手册",
                "content": "二级标题:设备\n\n这个chunk所在最近标题的位置:1\n\n上一个chunk的结尾内容:\nchunk内容:## 设备\n\n如果遵守了操作说明进行操作，但是设备不能正确运行，请仅调整 操作说明中涵盖的控制。错误调整其他控制可能导致损坏并且通常需要合格技术进行全面工作以将本设备恢复到正常操作。Brother不建议使用 Brother 正品烫金膜盒以外的其他品牌烫金膜盒。如果使用与本设备不兼容的耗材导致损坏本设备的任何零件，由此导致的任何维修可能不在保修范围内。",
                "item_name": "BrotherHAK180烫金机",
                "chunk_id": 468318939296846915,
                "score": 0.6852555274963379,
                "source": "local"
            },
            {
                "title": "HAK 180 烫金机",
                "file_title": "hak180产品安全手册",
                "content": "二级标题:HAK 180 烫金机\n\n这个chunk所在最近标题的位置:2\n\n上一个chunk的结尾内容:。\n\n•\t本文档中提供的信息可能会随时更改，恕不另行通知。\n\n•\t严禁未经授权擅自复制或重制本文档的任何部分或全部内容。\n\n\nchunk内容:•\t请注意，对于使用通过本设备制作的产品造成的任何损坏或利润损失，或者故障、维修导致的数据消失或更改，或者第三方提出 的任何索赔，我们不承担任何责任。",
                "item_name": "BrotherHAK180烫金机",
                "chunk_id": 468318939296846903,
                "score": 0.6810614466667175,
                "source": "local"
            }
        ]
    }

    logger.info(parse_json(node(init_state)))

