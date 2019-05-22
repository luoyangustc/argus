# -*- coding: utf-8 -*-
from config import config
from evals.utils import CTX
import json
#
#```
#{
#	"status": < int >,
#"id_res":{
#			 "address": < string >,
#"id_number": < string >, /
#"name": < string >, //
#"people": < string >, //
#"sex": < string >, //
#"type": < string > // "第二代身份证"
#}
#}
#```
import sys
reload(sys)
sys.setdefaultencoding('utf-8')
class idcard_post(object):
	def __init__(self):
		return

	def gender_filter(self, predict):
		if predict in config.POST.GENDER_FILTER_MALE:
			predict = "男"
		elif predict in config.POST.GENDER_FILTER_FEMALE:
			predict = "女"
		else:
			predict = ""
		return predict


	def postProcessing(self,predicts):
                punctuation = [',','。','<','>','@',':','“','”','"','\'','~',
                              '《','》','[',']','#','=','±','+','_','-','°',
                              '!','！',';','.','/','%','*','(',')','?','？',
                              '、','￡','￥','$','〈','〉','{','}','//','\\','|']
		punctuation1 = [',','。','<','>','@',':','“','”','"','\'','~',
                              '《','》','[',']','#','=','±','+','_','°',
                              '!','！',';','.','/','%','*','(',')','?','？',
                              '、','￡','￥','$','〈','〉','{','}','//','\\','|']      
		if len(predicts) != len(config.SEGMENT.TEMPLATE_FIELDS_LIST):
			id_res = {}
			id_res["status"] = -1
			return json.dumps(id_res)
		returnres ={}
		for i in range(len(config.SEGMENT.TEMPLATE_FIELDS_LIST)):
			key = config.SEGMENT.TEMPLATE_FIELDS_LIST[i]
#			print(len(predicts))
#			print(predicts)
			if key == 'name':
				nname = predicts[i]
				#print(nname)
				#print(type(nname))
				if len(nname) == 4 and nname[-1] == nname[-2]:
					nname = nname[:-1]
                                for fuhao in punctuation:
        			    if fuhao in nname:
				        nname = nname.replace(fuhao,"")
				returnres['name'] = nname

			if key == 'gender':
				idresult = predicts[config.SEGMENT.TEMPLATE_FIELDS_LIST.index('id')]
				if len(idresult) == 18:
					gender = idresult[-2]
					if gender != 'X':
						gender = int(gender)
						if gender % 2 == 0:
							returnres['sex'] = u'女'
						else:
							returnres['sex'] = u'男'
					else:
						returnres['sex'] = self.gender_filter(predicts[i])
				else:
					returnres['sex'] = self.gender_filter(predicts[i])

			if key == 'nation':
				nation_result = predicts[i]
				for fuhao in punctuation:
					if fuhao in nation_result:
						nation_result = nation_result.replace(fuhao,"")		
				if u'改' in nation_result or u'汉' in nation_result:
					returnres['people'] = u'汉'
				else:
					returnres['people'] = nation_result

			if key == 'id':
				returnres['id_number'] = predicts[i]
			if key == "address1":
 				address1 = predicts[i]
				address1s = predicts[i]
                                for fuhao in punctuation1:
                                    if fuhao in address1:
                                        address1s = address1s.replace(fuhao,"")
				#print("address1:"+str(predicts_new))
				returnres['address'] =  address1s
			if key == "address2":
				address2 = predicts[i]
				address2s = predicts[i]
                                for fuhao in punctuation1:
                                    if fuhao in address2:
                                        address2s = address2s.replace(fuhao,"")
				returnres['address'] += address2s
			if key == "address3":
				address3 = predicts[i]
				address3s = predicts[i]
				for fuhao in punctuation1:
                                    if fuhao in address3:
                                        address3s = address3s.replace(fuhao,"")
				returnres['address'] += address3s
		id_res ={}
		none_num =0
#		print(returnres)

		for res in returnres:
#s			print(res)
			if returnres[res] == '':
				none_num+=1
#		print(len(returnres))
#		print(none_num)

		CTX.logger.debug("returnres:%s",returnres)
		if none_num < len(returnres)/2 -1:
			id_res["status"] =0
			id_res['id_res'] = returnres
		else:
			id_res["status"] = -1

		return id_res

if __name__ == '__main__':
	tests =["name","zhejiang","c","july","1990","hanzhu","330xxx123456","5th","女","10","11"]
	post = idcard_post()
	resjson  = post.postProcessing(tests)
	print(resjson)

	tests =[None,"zhejiang","c","july","1990",None,None,"5th",None,"10","11"]
	post = idcard_post()
	resjson  = post.postProcessing(tests)
	print(resjson)

	tests =[]
	post = idcard_post()
	resjson  = post.postProcessing(tests)
	print(resjson)
