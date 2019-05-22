#!/usr/bin/python
# coding:utf-8
import cv2
import os
from .Machersolution import Matcher
import numpy as np 
import json
import base64
import requests
import sys
import config

Debug = True

class AlignerIDCard(object):
    def __init__(self,templateImg,templateLabel):
        templateimg = cv2.imread(templateImg)
        self.size = templateimg.shape		
        self.matcher = Matcher()
        self.KP,self.DES = self.matcher.sift_fet(templateimg)
        f = open(templateLabel)
        lines = f.readlines()
        self.num = 0
        f.close()
        self.rects = []
        self.names = []
        self.imageRoot = None
        for line in lines:
            self.names.append(line.strip().split(' ')[0])
            self.rects.append(list(map(int,line.strip().split(' ')[1:])))

    def align(self,im):
        kp, des = self.matcher.sift_fet(im)
        alignedImg, M = self.matcher.post_match(self.KP, self.DES, kp, des, im, input_roi=None, outputsize=(self.size[1],self.size[0]),offset=(0, 0), threshold=0.7, good_num=10)
        result = {}


        for idx,rect in enumerate(self.rects):		
            pos =  np.array([[rect[0],rect[0],rect[2],rect[2]],[rect[1],rect[3],rect[3],rect[1]]]).T
            cut = alignedImg[rect[1]:rect[3],rect[0]:rect[2]]
            name = self.names[idx]
            result[name] = [cut,pos]
        return result,alignedImg

    def det(self,im):
        address = config.MODEL_UNIVERSAL_EAST_IDCARD_API #MODEL_UNIVERSAL_EAST_DRIVER_API  MODEL_UNIVERSAL_EAST_API
        img_encoded = base64.b64encode(cv2.imencode('.png',im)[1].tostring())
        result = requests.post(address,data=img_encoded)
        boxes = json.loads(result.text,encoding='utf-8')
        return boxes

    def rec(self,im,boxes):
        address = config.MODEL_CRANN_BOXES_API
        img_encoded = base64.b64encode(cv2.imencode('.png',im)[1].tostring())
        result = requests.post(address,data=json.dumps({'img':img_encoded.decode('utf-8'),'bboxes':boxes}))
        result = json.loads(result.text,encoding='utf-8')
        return result
    

    def within(self,boxes,regions,th=0.4):
        ret = []
        for region in regions:
            res = []
            region = np.array(region)
            x11 = np.min(region[:,0])
            y11 = np.min(region[:,1])
            x12 = np.max(region[:,0])
            y12 = np.max(region[:,1])
            for i,box in enumerate(np.array(boxes)):
                x21 = np.min(box[:,0])
                y21 = np.min(box[:,1])
                x22 = np.max(box[:,0])
                y22 = np.max(box[:,1])
                area = (x22 - x21+1)*(y22-y21+1)
                xx1 = max(x11,x21)
                yy1 = max(y11,y21)
                xx2 = min(x12,x22)
                yy2 = min(y12,y22)
                w = max(0.0, xx2 - xx1 + 1)
                h = max(0.0, yy2 - yy1 + 1)
                inter = w * h
                ioi = inter*1.0/area
                if ioi >th:
                    res.append(i)
            ret.append(res)
        return ret

    def AinB(self,A,B):
        x11,y11,x12,y12 = A[0],A[1],A[2],A[3]
        x21,y21,x22,y22 = B[0],B[1],B[2],B[3]
        xx1 = max(x11,x21)
        yy1 = max(y11,y21)
        xx2 = min(x12,x22)
        yy2 = min(y12,y22)
        w = max(0.0, xx2 - xx1 + 1)
        h = max(0.0, yy2 - yy1 + 1)
        inter = w * h
        return inter*1.0/((x12-x11+1)*(y12-y11+1))

    def filtersmall(self,boxes,minH):
        boxes2 = []
        for box in boxes:
            h = min(np.linalg.norm([box[1][0] - box[0][0],box[1][1] - box[0][1]]),np.linalg.norm([box[2][0] - box[1][0],box[2][1] - box[1][1]]))
            if h >minH:
                boxes2.append(box)
        return boxes2

    def trimBoundary(self,boxes,regions):
        #match K->V
        withinIdx = self.within(boxes,regions)
        for regionIdx,indices in enumerate(withinIdx):
            region = regions[regionIdx]
            l,r,t,b = np.min(region[:,0]),np.max(region[:,0]),np.min(region[:,1]),np.max(region[:,1])
            for boxIdx in indices:
                box = np.array(boxes[boxIdx])
                box[0,0] = box[1,0] = l
                box[2,0] = box[3,0] = r
                box[0,1] = box[3,1] = max(max(box[0,1],box[3,1]),t)
                box[1,1] = box[2,1] = min(min(box[1,1],box[2,1]),b)
                boxes[boxIdx] = box.tolist()
        return boxes

    def cvtResults(self,boxes,texts,regions,names):
        #match K->V
        withinIdx = self.within(boxes,regions)
        #organize result
        ret = {}
        for idx,name in enumerate(names):
            rindices = withinIdx[idx]
            res = []
            for index in rindices:
                pred = np.array(boxes[index])
                L = np.min(pred[:,0])
                T = np.min(pred[:,1])
                R = np.max(pred[:,0])
                B = np.max(pred[:,1])
                text = texts[index]
                res.append({'pos':[L,T,R,B],'text':text})
            ret[name] = res
        return ret

    def Chinese(self,ch):
        if ord(ch) >= 0x4e00 and ord(ch)<= 0x9fa5:
            return True
        else:
            return False
    

    def merge(self,ret):
        ret2 = {}
        for key in ret:
            text = ''.join(ret[key])
            ret2[key] = text
        return ret2
    
    def resizebox(self,bboxes,image):
        newboxes = []
        for box in bboxes:
            box = np.array(box)
            # print(box)
            centx = np.mean(box[:,0])
            centy = np.mean(box[:,1])
            cent = np.array([centx,centy])
            # cv2.circle(image,(int(centx),int(centy)),5,(0,0,255),5)
            centpoint = []
            for i in range(4):
                centpoint.append((box[i]+box[(i+1)%4])/2)
                # cv2.circle(image,(int(((box[i]+box[(i+1)%4])/2)[0]),int(((box[i]+box[(i+1)%4])/2)[1])),5,(0,0,255),5)
            centpoint = np.array(centpoint)
            dis = np.sqrt((centpoint[:,0]-centx)*(centpoint[:,0]-centx)+(centpoint[:,1]-centy)*(centpoint[:,1]-centy))
            newbox = []
            if(dis[0]+dis[2]>dis[1]+dis[3]):
                h = dis[1]+dis[3]
                centpoint[0]=(centpoint[0]-np.array([centx,centy]))*(h*0.5/dis[0]+1)+np.array([centx,centy])
                centpoint[2]=(centpoint[2]-np.array([centx,centy]))*(h*0.5/dis[2]+1)+np.array([centx,centy])
            else:
                h = dis[0]+dis[2]
                centpoint[3]=(centpoint[3]-np.array([centx,centy]))*(h*0.5/dis[3]+1)+np.array([centx,centy])
                centpoint[1]=(centpoint[1]-np.array([centx,centy]))*(h*0.5/dis[1]+1)+np.array([centx,centy])
            # for i in range(4):
            #     cv2.circle(image,(int(centpoint[i][0]),int(centpoint[i][1])),5,(0,255,255),5)
            for i in range(4):
                newbox.append(((centpoint[i]-cent)+(centpoint[(i+1)%4]-cent)+cent).astype(np.int32).tolist())
                # cv2.circle(image,(int(newbox[i][0]),int(newbox[i][1])),5,(0,255,255),5)
            newboxes.append(newbox)
            # cv2.imshow('image',image)
            # cv2.waitKey(0)
        return newboxes

    def preferredSize(self,im,maxlong):
        h,w = im.shape[:2]
        if h>w:
            H = maxlong
            W = int(maxlong*1.0/h*w)
        else:
            W = maxlong
            H = int(maxlong*1.0/w*h)
        return cv2.resize(im,(W,H))
    
    def tightBoundary(self,hitList,boxes,detectedBoxes,names,image):
        for idx,name in enumerate(names):
            if name in ['住址1','住址2','住址3','公民身份号码']:
                hl = hitList[idx]
                if len(hl)==1:
                    box = detectedBoxes[hl[0]]
                    if name == '公民身份号码':
                        boxes[idx] = box
                        continue
                    l,t,r,b= 32767,32767,-1,-1
                    for i in range(4):
                        l = min(box[i][0],l)
                        t = min(box[i][1],t)
                        r = max(box[i][0],r)
                        b = max(box[i][1],b)
                    # print(l,t,r,b)
                    
                    cut = image[t:b,l:r,:]
                    offsetx = l
                    begin,end = self.projectMethod(cut)
                    # cv2.imshow('cut',cut)
                    
                    # print(begin,end)
                    if begin<end:
                        l = offsetx+begin
                        r = offsetx+end
                        box[0][0],box[0][1]=l,t
                        box[1][0],box[1][1]=r,t
                        box[2][0],box[2][1]=r,b
                        box[3][0],box[3][1]=l,b
                    boxes[idx] = box
                    # for i in range(4):
                    #     cv2.line(image,tuple(box[i]),tuple(box[(i+1)%4]),(0,0,255),3)
                    # # cv2.imshow('image',image)
                    # # cv2.waitKey(0)
                    # # print(boxes)
                elif len(hl)>1:
                    print('xxxxxxxx',hl)
                    L,T,R,B= 32767,32767,-1,-1
                    for i in hl:
                        box = detectedBoxes[hl[0]]
                        l,t,r,b= 32767,32767,-1,-1
                        for i in range(4):
                            l = min(box[i][0],l)
                            t = min(box[i][1],t)
                            r = max(box[i][0],r)
                            b = max(box[i][1],b)
                        boxes[idx] = [[l,t],[r,t],[r,b],[l,b]]
                        
                elif name != '住址1' and name!= '公民身份号码':
                    boxes[idx] =  [[0,0],[0,1],[1,1],[1,0]]
        return np.array(boxes).tolist()

    def projectMethod(self,cut,th = 0.0):
        gray_img = cv2.cvtColor(cut,cv2.COLOR_BGR2GRAY)
        _, binary_img = cv2.threshold(gray_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        binary_img = 255 - binary_img

        verticalProj = np.sum(binary_img,axis=0)*1.0/binary_img.shape[1]
        begin = np.min(np.where(verticalProj>th)[0])
        end = np.max(np.where(verticalProj>th)[0])
        return begin,end

    def predet(self,img,num=0,filename=None):
        # img = cv2.imdecode(np.fromstring(base64.b64decode(img), dtype=np.uint8), 1)
        img = self.preferredSize(img,2000)
        result,alignedImg = self.align(img)

        if alignedImg is None:
            return []
        regions = []
        names = []
        for name in result:
            cut = result[name][0]			
            pos = result[name][1]
            if name == '姓名':
                offsetx = pos[0][0]
                begin,end = self.projectMethod(cut)
                if begin<end:
                    pos[0][0] = pos[1][0] = offsetx+begin
                    pos[2][0] = pos[3][0] = offsetx+end
            regions.append(pos.tolist())
            names.append(name)
        
        boxes = np.array(regions).tolist()
        return alignedImg,names,regions,boxes
    
    def prerecog(self,detectedBoxes,alignedImg,names,regions,boxes):
        detectedBoxes = self.resizebox(detectedBoxes,alignedImg)
        
        showimg = alignedImg.copy()
        
        hitList = self.within(detectedBoxes,regions,0.7)

        boxes = self.tightBoundary(hitList,boxes,detectedBoxes,names,alignedImg)
        boxes = self.filtersmall(boxes,10)
        # handle name issue
        boxes[-1][0][0] -= 20
        boxes[-1][1][0] -= 20
        boxes[-1][2][0] += 20
        boxes[-1][3][0] += 20
        return boxes


    def postprocess(self,boxes,texts,regions,names):
        ret = self.cvtResults(boxes,texts,regions,names)
        ret2 = {}
        for key in ret:
            if not '住址' in key:
                text = ret[key][0]['text'].decode('utf8')
                if key == '性民':
                    gender = '男'
                    nation = '汉'
                    if '女' in text.decode('utf8'):
                        gender = '女'
                    pos = text.find('族')
                    if pos > -1:
                        nation = text[pos+1:]
                        newnation = ''
                        for i in range(len(nation)):
                            if nation[i]>='0' and nation[i]<='9':
                                continue
                            newnation+=nation[i]
                        nation = newnation
                        if nation == '汊':
                            nation = '汉'
                    ret2['性别'] = gender
                    ret2['民族'] = nation
                elif key == '出生':
                    text2 = ''
                    for ch in text.decode('utf8'):
                        if ch.isdigit() or ch in ['年','月','日']:
                            text2 += ch
                    if text2.find('日' )==-1:
                        text2+='日'
                    accNum = 0
                    for i in range(len(text2)):
                        if text2[i].isdigit():
                            accNum+=1
                    prefix = {1:'199',2:'19',3:'1'}
                    if accNum<4 and accNum>=1:
                        text2 = prefix[accNum]+text2
                    if not ('年' in text2) and len(text2) > 4 and not (text2[4]=='年'):
                        text2 = text2[:4]+'年'+text2[4:]
                    ret2[key] = text2
                elif key == '姓名':
                    text2 = ''
                    for ch in text.decode('utf8'):
                        if self.Chinese(ch):
                            text2+=ch
                    ret2[key] = text2
                elif key =='公民身份号码':
                    text2 = ''
                    for ch in text.decode('utf8'):
                        if ch.isdigit():
                            text2+=ch
                        elif ch == 'x' or ch == 'X' or ch == 'Ｘ' or 'x':
                            text2+='X'
                    ret2[key] = text2
            else:
                tmp = ''
                ret2['住址'] = ''
                for zhuzhi in ['住址1','住址2','住址3']:
                    if len(ret[zhuzhi])>0:
                        zhuzhiziduan = ret[zhuzhi][0]['text']
                        zhuzhiziduan2 = ''
                        for ch in zhuzhiziduan.decode('utf8'):
                            if ch.isdigit()or self.Chinese(ch):
                                zhuzhiziduan2+=ch
                        tmp+=zhuzhiziduan2
                    else:
                        break
                for ch in tmp.decode('utf8'):
                    if ch != '住' and ch != '址':
                        ret2['住址']+=ch
        return ret2

    def test(self):
        f = open('output2.txt','w+')
        for filename in os.listdir(self.imageRoot):
            print(filename)
            f.write('\n')
            f.write(filename)
            f.write('\n')
            im = cv2.imread(self.imageRoot+'/'+filename)
            ret = self.run(im,filename)
            if Debug:
                for key in ret:
                    val = ret[key]
                    print(key,val)
                    f.write(key)
                    f.write(':')
                    f.write(val)
                    f.write('\n')
                print('----------------------------------')
                cv2.waitKey(30)
        f.close()

class AlignerJiaZhao(AlignerIDCard):
    def __init__(self):
        super(AlignerJiaZhao, self).__init__('template/1.jpg','template/1.txt')
        self.imageRoot = 'image'

if __name__ == '__main__':
    aligner = AlignerJiaZhao()
    aligner.test()

