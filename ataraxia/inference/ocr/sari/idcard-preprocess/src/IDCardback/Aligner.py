#!/usr/bin/python
# -*- coding: utf-8 -*-
import cv2
import os
from .Machersolution import Matcher
import numpy as np 
import json
import base64
import requests
import sys
import config
import logging

Debug = True

class AlignerIDCardBack(object):
    def __init__(self,templateImg,templateLabel):
        templateimg = cv2.imread(templateImg)
        self.size = templateimg.shape		
        self.matcher = Matcher()
        self.KP,self.DES = self.matcher.sift_fet(templateimg)
        f = open(templateLabel)
        lines = f.readlines()
        f.close()
        self.num = 0
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
                centpoint[0]=(centpoint[0]-np.array([centx,centy]))*(h*0.8/dis[0]+1)+np.array([centx,centy])
                centpoint[2]=(centpoint[2]-np.array([centx,centy]))*(h*0.8/dis[2]+1)+np.array([centx,centy])
            else:
                h = dis[0]+dis[2]
                centpoint[3]=(centpoint[3]-np.array([centx,centy]))*(h*0.8/dis[3]+1)+np.array([centx,centy])
                centpoint[1]=(centpoint[1]-np.array([centx,centy]))*(h*0.8/dis[1]+1)+np.array([centx,centy])
            # for i in range(4):
            #     cv2.circle(image,(int(centpoint[i][0]),int(centpoint[i][1])),5,(0,255,255),5)
            for i in range(4):
                newbox.append(((centpoint[i]-cent)+(centpoint[(i+1)%4]-cent)+cent).astype(np.int32).tolist())
                # cv2.circle(image,(int(newbox[i][0]),int(newbox[i][1])),5,(0,255,255),5)
            newboxes.append(newbox)
            # cv2.imshow('image',image)
            # cv2.waitKey(0)
        return newboxes

    def merge(self,ret):
        ret2 = {}
        for key in ret:
            text = ''.join(ret[key])
            ret2[key] = text
        return ret2

    def preferredSize(self,im,maxlong):
        h,w = im.shape[:2]
        if h>w:
            H = maxlong
            W = int(maxlong*1.0/h*w)
        else:
            W = maxlong
            H = int(maxlong*1.0/w*h)
        return cv2.resize(im,(W,H))

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
        #region of interests
        result,alignedImg = self.align(img)

        if alignedImg is None:
            return []

        regions = []
        names = []
        for name in result:
            cut = result[name][0]			
            pos = result[name][1]
            
            offsetx = pos[0][0]
            begin,end = self.projectMethod(cut)
            margin = 10
            if begin<end:
                pos[0][0] = pos[1][0] = offsetx+begin - margin
                pos[2][0] = pos[3][0] = offsetx+end + margin
            

            regions.append(pos.tolist())
            names.append(name)
        
        boxes = np.array(regions).tolist()
        return alignedImg,names,regions,boxes
    
    def prerecog(self,detectedBoxes,alignedImg,names,regions,boxes):
        detectedBoxes = self.resizebox(detectedBoxes,alignedImg)

        hitList = self.within(detectedBoxes,regions,0.7)
        boxes = self.filtersmall(boxes,30)
        return boxes

    def postprocess(self,boxes,texts,regions,names):
        ret = self.cvtResults(boxes,texts,regions,names)
        ret2 = {}
        for key in ret:
            text = ret[key][0]['text']
            # logging.critical(text)
            newtext = ''
            if key == '有效期限':
                num = 0
                for i in range(len(text)):
                    if(text[i]>='0' and text[i]<='9'):
                        newtext+=text[i]
                        num+=1
                    elif(text[i]=='.'):
                        continue
                    elif(text[i]=='-'):
                        continue
                    else:
                        newtext+=text[i:]
                        break
                    if num>16:
                        break
                    if(num%8==4 or num%8==6):
                        newtext+='.'
                    elif(num%8==0 and num<16):
                        newtext+='-'
                    if(len(newtext)==21):
                        break
                ret2[key] = newtext
            else:
                ret2[key] = text
        return ret2



if __name__ == '__main__':
    aligner = AlignerJiaZhao()
    aligner.test()

