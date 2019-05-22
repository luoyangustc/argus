# -*- coding: utf-8 -*-
import sys
from regular import Regular
from dfa import SensitiveDictLoader
from config import LabelType
from config import Config


if sys.version_info.major == 2:
    reload(sys)
    sys.setdefaultencoding('utf-8')
elif sys.version_info.major == 3:
    pass


class FilterEngine():
    def __init__(self, regular=None, dfa=None):
        if regular is None:
            self.regular = Regular()
        else:
            self.regular = Regular(regular)
        self.dfas = {}
        for t in LabelType:
            if t is LabelType.Normal:
                continue
            self.dfas[t] = SensitiveDictLoader(key_type=t)

        self.stop_keys = set()
        with open(Config.STOP_WORDS_PATH, "r") as r:
            for line in r:
                self.stop_keys.add(line.strip())

    def check(self, msgs, types=None):
        filter_msgs = []
        for msg in msgs:
            msg = "".join([c.lower() for c in msg if c not in self.stop_keys])
            filter_msgs.append(msg)

        result = {}
        check_types = []
        if types is None:
            check_types = [t for t in LabelType if t is not LabelType.Normal]
        else:
            check_types = [LabelType(t) for t in types]

        for t in check_types:
            t_map = {}
            summary = {}
            confidences = []
            t_count = 0

            t_max_score = 0.0

            for msg in filter_msgs:

                confidence = {}
                dfa_keys, dfa_type, dfa_score = self.dfas[t].query(msg)
                confidence["keys"] = dfa_keys
                confidence["label"] = dfa_type.value
                confidence["score"] = dfa_score

                if t is LabelType.Ads:
                    reg_keys, reg_type, reg_score = self.regular.check(msg)
                    if reg_type is t and dfa_type is t:
                        dfa_keys.extend(reg_keys)
                        confidence["keys"] = list(set(dfa_keys))
                        if dfa_score >= 0.95:
                            confidence["score"] = 1.0
                        if (dfa_score >= 0.8 and reg_score < 0.8):
                            score = 0.9 + float(len(msg)) / 120.0
                        if (dfa_score < 0.8 and reg_score >= 0.8):
                            score = 0.9 + float(len(msg)) / 240.0
                        elif dfa_score >= 0.8 and reg_score >= 0.8:
                            score = 1.0
                        elif dfa_score < 0.8 and reg_score < 0.8:
                            score = 0.9 + float(len(msg)) / 400.0
                        else:
                            score = 1.0
                        if score > 1.0:
                            score = 1.0
                        confidence["score"] = score
                    elif reg_type is t and dfa_type is not t:
                        confidence["keys"] = reg_keys
                        confidence["label"] = reg_type.value
                        confidence["score"] = reg_score

                if LabelType(confidence["label"]) is t:
                    t_count += 1
                    if t_max_score < confidence["score"]:
                        t_max_score = confidence["score"]
                confidences.append(confidence)

            if 0 < t_max_score < 0.8 and 2 < t_count < 10:
                t_max_score = 0.9 + float(t_count) / 100.0
            elif 0 < t_max_score < 0.8 and t_count >= 10:
                t_max_score = 1.0
            elif t_max_score >= 0.8 and t_count >= 2:
                t_max_score = 1.0
            if t_count > 0 and t_max_score > 0:
                summary["label"] = t.value
                summary["score"] = t_max_score
            else:
                summary["label"] = LabelType.Normal.value
                score = 1.0
                if len(msgs) > 15:
                    score = 1.0 - float(len(msgs)) / 135.0
                if score < 0.9:
                    score = 0.9
                summary["score"] = score
            t_map["summary"] = summary
            t_map["confidences"] = confidences
            result[t.value] = t_map
        return result

    def review(self, score):

        if score > 0.5 and score < 0.9:
            return True
        else:
            return False


if __name__ == "__main__":
    msgs = ["向你推荐 微信:abd2345453443",
            "互看身体加我微信",
            "27745156637",
            "想看大鸡吧",
            "13487846841",
            "想看大鸡鸡，我微信",
            "18334390332",
            "想看大鸡吧",
            "26659196301",
            "13852729142",
            "给你看你想看的",
            "加我微信",
            "这里不方便加我微信170",
            "63430451我等你哦",
            "互看身体",
            "26659197711",
            "这个平台不会玩加我微信",
            "233206059406",
            "一个离婚的女人等你",
            "是我微信841213947",
            "望在那里可以见到你",
            "037，妹妹好急，希",
            "这个平台不会用，这",
            "13452575409",
            "互看身体加我",
            "微信",
            "给你看你想看的",
            "加我微信",
            "18795964678",
            "加我微信",
            "给你看你想看的",
            "17746902097",
            "这个平台不好玩，加我微信8490",
            "4535301好痒，陪陪我，等你哦。",
            "13165253748",
            "给你看你想看的",
            "加我微信",
            "互看身体",
            "27648555894",
            "给你撩，给你看",
            "13113320830",
            "加我微信",
            "哥哥加我微信在线等你哦",
            "连信不方便哥哥",
            "19172594350",
            "伽V心互看下面",
            "18339545657",
            "CJQ13411775534",
            "加我微信：",
            "233267212026",
            "离过婚的女人",
            "我的微信。",
            "遇到渣男被甩陪我",
            "13059743134",
            "想和你69式",
            "加我微信",
            "17734867215",
            "给你看你想看的",
            "加我微信",
            "15158036124",
            "给你撩，给你看",
            "18323386155",
            "加我微信",
            "66917693519",
            "想看大鸡鸡，我微信",
            "13406144260",
            "伽V心互看下面",
            "这个平台不会玩，加我微信",
            "离婚的女人等你",
            "841289699356",
            "加我微信15210414378",
            "我好通过",
            "只进入身体不进入生活",
            "加了记得跟我说网名喔",
            "想看大鸡吧",
            "18116265549",
            "微信号：26826525",
            "非诚勿扰，骗子勿扰",
            "妹妹本人一对一视频",
            "有惊喜哦！你懂得！",
            "加我微信：66627930447，看表演",
            "给你看你想看的",
            "加我微信",
            "15052524297",
            "这个平台不长上，加我",
            "微信吧：18083736005",
            "微信",
            "互看身体加我",
            "13452659311",
            "想认识我吗加微信",
            "752105899等你呦！",
            "16189993674",
            "加我微信",
            "15152476844",
            "给你看你想看的",
            "晚上无聊等你来玩互动",
            "妹妹薇信：27627890084",
            "点我",
            "头像",
            "13393937185",
            "互看伽V信",
            "加我微信",
            "给你撩，给你看",
            "ye19780406",
            "77023285417",
            "互看身体加微信",
            "加我哦，给你撩，给你看",
            "私人微信：17339403883",
            "互看身体",
            "18302969989",
            "加我微信",
            "13195734856",
            "给你看你想看的",
            "互看身体",
            "18015230749",
            "免费送一部黄片",
            "搜索连信好友",
            "nana9505",
            "12697932312",
            "交朋友+微信",
            "27628254185",
            "互看身体+V",
            "18807313819",
            "想看大鸡鸡，我微信",
            "给你撩，给你看",
            "加我微信",
            "15868434406",
            "16189993674",
            "互看身体",
            "233238901214",
            "互看身体",
            "233238901214",
            "加微信裸聊oww890",
            "26659707403",
            "想看大鸡吧",
            "想看大鸡吧",
            "66628867743",
            "互看身体加",
            "微信",
            "15882042985",
            "加我微信13681580035",
            "只进入身体不进入生活",
            "加了记得跟我说网名喔",
            "我好通过",
            "想看大鸡鸡，我微信",
            "26659718966",
            "想看大鸡鸡微信",
            "27747325449",
            "愿意的话可以加我",
            "睡不着了，有人陪我聊天吗",
            "微信qqu22222聊一会，",
            "希望能陪我多聊一会",
            "94726041625",
            "想看大鸡吧",
            "26659718240",
            "想看大鸡吧",
            "26659718240",
            "想看大鸡吧",
            "代充Q币",
            "微信：abdcd02dfg3434545"]

    engine = FilterEngine()
    print(engine.check(msgs, types=[LabelType.Ads]))
    msg2 = ["菌苗已由该所投产", "代充Q币"]
    print(engine.check(msg2, types=[LabelType.Ads]))
    msg3=["123232344","加我微信"]
    print(engine.check(msg3,types=[LabelType.Ads]))
