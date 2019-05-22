export RUN_MODE=standalone

cd python/evals
wget http://pbqb5ctvq.bkt.clouddn.com/YBZZS_01488003.jpg

import cv2
import eval
from evals.src.invoice import ZenZhuiShui_reco
from evals.src.postProcessDict import postProcess
mod = eval.create_net({'batch_size':1})
daikai_model = mod[0]['daikai_model']
fapaiolian_model = mod[0]['fapaiolian_model']
img = cv2.imread('YBZZS_01488003.jpg')
vat = ZenZhuiShui_reco()
rect_boxes,boxes_dict = vat.gen_img_dict(img)


rec_result = [u'\u7ef5\u9633\u5c71\u91d1\u673a\u7535\u8bbe\u5907\u6709\u9650\u516c\u53f8', u'2015\u5e7412\u670815\u65e5', u'01488003', u'\uffe511230.77', u'510015213', u'\u7ef5\u9633\u9ad8\u65b0\u533a\u666e\u660e\u5357\u8def\u4e1c\u6bb521\u4e16\u7eaa\u4e8c\u671f', u'0816-2530378', u'17%', u'', u'KFR-72LW/', u'\u5de5\u884c\u7ef5\u9633\u4e34\u56ed\u652f\u884c2308412409024849172', u'\u7a7a\u8c03', u'\u58f9\u4e07\u53c1\u4edf\u58f9\u4f70\u8086\u62fe\u5706\u6574', u'\u5957', u'1', u'11230.769231', u'\u56db\u5ddd\u7ef5\u9633\u79d1\u521b\u56ed\u533a\u56ed\u827a\u4e1c\u88578\u53f70816-2536680', u'1909.2', u'510798551022257', u'\u56db\u5ddd\u5149\u53d1\u79d1\u6280\u6709\u9650\u516c\u53f8', u'11230.77', u'6', u'510790567644042', u'\u5f20\u5168\u534e', u'\u4e2d\u56fd\u94f6\u884c\u80a1\u4efd\u6709\u9650\u516c\u53f8\u7ef5\u9633\u6daa\u57ce\u652f\u884c', u'123912372612', u'\uffe51909.2', u'\uffe513140.00', u'\u519c\u62c9\u80a1', u'\u9644', u'476<*51+5485+563459018135>8', u'5*/425446>370892**107*<65', u'790+09+5825>5499+*315191381', u'-1/4324-0457431992*<9883', u'01488003', u'5100152130', u'\u56db\u5ddd\u589e\u503c\u7a0e\u4e13\u7528\u53d1\u7968']

vat = ZenZhuiShui_reco()    # 没法传递image_dict参数，只能新建实例然后重新生成参数
vat.gen_img_dict_base(img)
vat.predict_oridinary(boxes_dict,rec_result)
vat.predict_other(boxes_dict,rec_result)
vat.predict_XiaoShouMingXi(boxes_dict,rec_result)
vat.predict_svm(daikai_model)
vat.predict_FaPiaoLianCi(fapaiolian_model)
vat.predict_XiaoLeiMingCheng()
res = postProcess(vat.out_dict)
