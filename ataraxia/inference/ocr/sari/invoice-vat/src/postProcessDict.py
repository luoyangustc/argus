# -*- coding: utf-8 -*-
def reviseDanZhengMingCheng(input_dict):
    '''
    纠正单证名称字段
    :param input_dict:待纠正的字段
    :return:纠正后的字段
    '''
    mingcheng = input_dict['_DanZhengMingCheng']
    if (not mingcheng == None) and (not mingcheng == []):
        c = mingcheng[-1]
        ret = mingcheng[0:2]
        if c == u'票':
            if mingcheng[-3] == u'普' or mingcheng[-3] == u'通':
                ret += u'增值税普通发票'
            else:
                ret += u'增值税专用发票'
        else:
            if mingcheng[-2] == u'普' or mingcheng[-2] == u'通':
                ret += u'增值税普通发票'
            else:
                ret += u'增值税专用发票'
        input_dict['_DanZhengMingCheng'] = ret
    return input_dict

def reviseShuiLv(input_dict):
    '''
    纠正识别后的字段
    :param input_dict:待纠正的字段
    :return:纠正销售明细字段
    '''
    mingxibiao = input_dict['_XiaoShouMingXi']
    if len(mingxibiao) > 0:
        for mingxi in mingxibiao:
            content = mingxi[-2]
            if content != None:
                ret = ''
                for c in content:
                    if ord(c) >= ord('0') and ord(c) <= ord('9'):
                        ret = ret + c
                if len(ret) > 2:
                    ret = ret[0:2]
                ret = ret + '%'
                mingxi[-2] = ret
    return input_dict

def reviseKaiPiaoRen(input_dict):
    '''
    纠正收款人,复核人,开票人字段
    :param input_dict:待纠正的字段
    :return:纠正后的字段
    '''
    persons = ['_ShouKuanRen', '_FuHeRen', '_KaiPiaoRen']
    for person in persons:
        _person = input_dict[person]
        if len(_person) > 0 and _person[0] == u'自':
            input_dict[person] = u'自助代开'
    return input_dict

def reviseMima(input_dict):
    '''
    纠正识别后的字段
    :param input_dict:待纠正的字段
    :return:纠正后的字段
    '''
    def replace(s):
        s = s.replace(u'￥', '*')
        s = s.replace(u'Y', '*')
        s = s.replace(u'十', '+')
        s = s.replace(u'米', '*')
        s = s.replace(u'一', '-')
        s = s.replace(u']', '1')

        s = s.replace(u'[', '1')
        s = s.replace(u'金', '>')
        s = s.replace(u'大', '*')
        s = s.replace(u'水', '*')
        s = s.replace(u'称', '*')
        s = s.replace(u'之', '>')
        s = s.replace(u'P', '>')
        s = s.replace(u'C', '<')
        s = s.replace(u'c', '<')
        s = s.replace(u'火', '*')
        s = s.replace(u'七', '+')
        return s
    mima = input_dict['_MiMa']
    input_dict['_MiMa'] = replace(mima)
    return input_dict

def reviseDaXieJinE(input_dict):
    '''
    纠正识别后的字段
    :param input_dict:待纠正的字段
    :return:纠正大写金额字段
    '''
    text = input_dict['_JiaShuiHeJi_DaXie']
    if text != None:
        text = ''.join(text.split(' ')[:])
        if len(text)>0:
            text = text[:-1]
            if text[0] != u'ⓧ':
                text = u'ⓧ' + text
            if text[-1] != u'整' and text[-1] != u'分' and len(text) > 1:
                if text[-1] == u'角' or text[-1] == u'圆':
                    text += u'整'
            input_dict['_JiaShuiHeJi_DaXie'] = text
    return input_dict

def reviseDK_XiaoshouFang(input_dict):
    '''
    纠正识别后的字段
    :param input_dict:待纠正的字段
    :return:纠正销售方开户行记账号，销售方名称，销售方纳税人识别号
    '''
    xiaoshoufang = ['_XiaoShouFangKaiHuHangJiZhangHao',
                    '_XiaoShouFangMingCheng', '_XiaoShouFangNaShuiRenShiBieHao']
    for _xiaoshoufang in xiaoshoufang:
        comps = input_dict[_xiaoshoufang].split(' ')
        if len(comps)>0:
            _xiaoshoufang_text = comps[-1]
            if(len(_xiaoshoufang_text)<=2):
                continue
            c1 = _xiaoshoufang_text[-1]
            c2 = _xiaoshoufang_text[-2]
            c3 = _xiaoshoufang_text[1]

            if c3 == u'代' or c1 == u'关' or c2 == u'关' or c1 == u'机' or c2 == u'机':
                out_text = ''
                for i in range(len(comps) - 1):
                    out_text += comps[i]
                out_text += u' (代开机关)'
                input_dict[_xiaoshoufang] = out_text
                input_dict['_DaiKaiBiaoShi'] = u'代开'
                input_dict['_XiaoLeiMingCheng'] = u'增值税代开发票'
            if c3 == u'完' or c1 == u'号' or c2 == u'号' or c1 == u'证' or c2 == u'证':
                out_text = ''
                for i in range(len(comps) - 1):
                    out_text += comps[i]
                out_text += u' (完税凭证号)'
                input_dict[_xiaoshoufang] = out_text[:-7]
    return input_dict

def reviseFaPiaoHaoMa(input_dict):
    '''
    纠正识别后的字段
    :param input_dict:待纠正的字段
    :return:纠正发票号码印刷字段
    '''
    text = input_dict['_FaPiaoHaoMa_YinShua']
    if text != None:
        out = ''
        for c in text:
            if c>='0' and c<='9':
                out= out+c
        input_dict['_FaPiaoHaoMa_YinShua'] = out
    return input_dict

def reviseKaiPiaoRiQi(input_dict):
    '''
    纠正识别后的字段
    :param input_dict:待纠正的字段
    :return:纠正开票日期字段
    '''
    text = input_dict['_KaiPiaoRiQi']
    if text!=None:
        if text.find(u'日')==-1:
            input_dict['_KaiPiaoRiQi'] = text+u'日'
    return input_dict

def reviseJiaoYanMa(input_dict):
    text = input_dict['_JiaoYanMa']
    if text!=None:
        text = text.replace(u'校', '')
        text = text.replace(u'验', '')
        input_dict['_JiaoYanMa'] = text.replace(u'码', '').strip()
    return input_dict

def postProcess(input_dict):
    '''
    纠正识别后的字段
    :param input_dict:待纠正的字段
    :return:纠正后的字段
    '''
    input_dict = reviseMima(input_dict)
    input_dict = reviseDaXieJinE(input_dict)
    input_dict = reviseShuiLv(input_dict)
    input_dict = reviseDanZhengMingCheng(input_dict)
    input_dict = reviseKaiPiaoRen(input_dict)
    input_dict = reviseDK_XiaoshouFang(input_dict)
    input_dict = reviseFaPiaoHaoMa(input_dict)
    input_dict = reviseKaiPiaoRiQi(input_dict)
    input_dict = reviseJiaoYanMa(input_dict)
    return input_dict
