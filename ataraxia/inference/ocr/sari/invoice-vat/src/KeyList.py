# -*- coding: utf-8 -*-

# 所有应该出现在xml中的key
key_list = ['_DaLeiMingCheng', '_XiaoLeiMingCheng', '_FaPiaoDaiMa_YinShua', '_FaPiaoHaoMa_YinShua', '_DanZhengMingCheng',
            '_FaPiaoLianCi', '_DaiKaiBiaoShi', '_FaPiaoJianZhiZhang', '_FaPiaoDaiMa_DaYin', '_FaPiaoHaoMa_DaYin', '_KaiPiaoRiQi',
            '_JiQiBianHao', '_JiaoYanMa', '_GouMaiFangMingCheng', '_GouMaiFangNaShuiShiBieHao', '_GouMaiFangDiZhiJiDianHua',
            '_GouMaiFangKaiHuHangJiZhangHao', '_MiMa', '_XiaoShouMingXi', '_HeJiJinE_BuHanShui', '_HeJiShuiE',
            '_JiaShuiHeJi_XiaoXie', '_JiaShuiHeJi_DaXie', '_DaiKaiJiGuanMingCheng', '_DaiKaiJiGuanHaoMa',
            '_DaiKaiJiGuanDiZhiJiDianHua', '_WanShuiPingZhengHao', '_XiaoShouFangMingCheng', '_XiaoShouFangNaShuiRenShiBieHao',
            '_XiaoShouFangDiZhiJiDianHua', '_XiaoShouFangKaiHuHangJiZhangHao', '_BeiZhu', '_FaPiaoYinZhiPiHanJiYinZhiGongSi',
            '_ShouKuanRen', '_FuHeRen', '_KaiPiaoRen', '_XiaoShouDanWeiGaiZhangLeiXing', '_GaiZhangDanWeiMingCheng',
            '_GaiZhangDanWeiShuiHao', '_DaiKaiJiGuanGaiZhang'] #'_HuoWuHuoYingShuiLaoWuMingCheng', '_GuiGeXingHao', 
            #'_DanWei', '_ShuLiang', '_DanJia','_ShuiLv', '_ShuiE', '_JinE_BuHanShui',


# 所有可以合并处理的key
oridinary_key_list = ['_DanZhengMingCheng',
                      '_FaPiaoDaiMa_YinShua', '_FaPiaoHaoMa_YinShua',
                      '_FaPiaoDaiMa_DaYin', '_FaPiaoHaoMa_DaYin',
                      '_KaiPiaoRiQi',
                      '_GouMaiFangMingCheng', '_GouMaiFangNaShuiShiBieHao', '_GouMaiFangDiZhiJiDianHua', '_GouMaiFangKaiHuHangJiZhangHao',
                      '_MiMa',
                      '_HeJiJinE_BuHanShui', '_HeJiShuiE',
                      '_JiaShuiHeJi_XiaoXie', '_JiaShuiHeJi_DaXie',
                      '_XiaoShouFangMingCheng', '_XiaoShouFangNaShuiRenShiBieHao', '_XiaoShouFangDiZhiJiDianHua', '_XiaoShouFangKaiHuHangJiZhangHao',
                      '_BeiZhu',
                      '_FaPiaoYinZhiPiHanJiYinZhiGongSi',
                      '_ShouKuanRen', '_FuHeRen', '_KaiPiaoRen']

# 需要返回图片的key
image_key_list = ['_FaPiaoJianZhiZhang', '_XiaoShouDanWeiGaiZhangLeiXing',
                  '_GaiZhangDanWeiMingCheng', '_GaiZhangDanWeiShuiHao']

# 需要对图片进行分类的key
to_cls_key_list = ['_FaPiaoLianCi', '_DaiKaiBiaoShi']

# 发票中间的表格相关的key
table_key_list = ['_XiaoShouMingXi',
                  '_HuoWuHuoYingShuiLaoWuMingCheng', '_GuiGeXingHao', '_DanWei', '_ShuLiang', '_DanJia', '_JinE_BuHanShui', '_ShuiLv', '_ShuiE']

# 依赖于其他关键字的key, 二选一
other_key_list = ['_JiQiBianHao', '_JiaoYanMa']

# 代开独有的key, 返回的是字符串
daiKai_key_list = ['_DaiKaiJiGuanMingCheng', '_DaiKaiJiGuanHaoMa',
                   '_DaiKaiJiGuanDiZhiJiDianHua', '_WanShuiPingZhengHao']

# 代开独有的key, 返回的是图片（目前还没有处理）
daiKai_image_key_list = ['_DaiKaiJiGuanGaiZhang']
