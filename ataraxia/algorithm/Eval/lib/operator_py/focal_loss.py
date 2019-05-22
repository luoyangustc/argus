# --------------------------------------------------------
# Focal loss
# Licensed under The Apache-2.0 License [see LICENSE for details]
# Written by unsky https://github.com/unsky/
# --------------------------------------------------------

"""
Focal loss 
"""

import mxnet as mx
import numpy as np

class FocalLossOperator(mx.operator.CustomOp):
    def __init__(self,  gamma, alpha,use_ignore,ignore_label):
        super(FocalLossOperator, self).__init__()
        self._gamma = gamma
        self._alpha = alpha 
        self.use_ignore = use_ignore
        self.ignore_label = ignore_label
        # print('Focalloss params: ',self._gamma,self._alpha,self.use_ignore,self.ignore_label)

    def _SoftmaxActivation(in_data,out_data):
        shape = in_data.shape


    def forward(self, is_train, req, in_data, out_data, aux):
        # print('----forward----')
      
        cls_score = in_data[0]
        # ----adapt----
        self.shape = cls_score.shape
        cls_score = mx.nd.reshape(cls_score,(-3,-3))
        cls_score = mx.nd.swapaxes(cls_score,0,1)
        # ----adapt----
        # print('cls_score.shape: ',cls_score.shape)
        # print('cls_score[0]: ',in_data[0].asnumpy()[0])

        # print('labels: ',in_data[1].asnumpy())

        # labels = in_data[1].asnumpy()[0]
        labels = mx.nd.swapaxes(in_data[1],0,1).asnumpy()
        # labels = in_data[1].asnumpy().reshape((-1,1))
        # print('labels.shape: ',labels.shape)
    
        self._labels = labels

        pro_ =(mx.nd.SoftmaxActivation(cls_score) + 1e-14).asnumpy()
        # pro_ =(_SoftmaxActivation(cls_score) + 1e-14).asnumpy()

        self.pro_ = pro_
        # print('pro_[0]: ',pro_[0])
        # for index,label in enumerate(labels.astype('int').tolist()):
        #     if label != 0:
        #         print('pro_[{}]: {}'.format(index,pro_[index]))
        # print('pro_.shape: ',self.pro_.shape)
       
        # self._pt = pro_[np.arange(pro_.shape[0],dtype = 'int'), labels[:,0].astype('int')].reshape((-1,1))
        self._pt = pro_[np.arange(pro_.shape[0],dtype = 'int'), labels[0].astype('int')]
        # print('_pt: ',self._pt)
        # print('_pt.shape: ',self._pt.shape)
 
        ### note!!!!!!!!!!!!!!!!
        # focal loss value is not used in this place we should forward the cls_pro in this layer, 
        # the focal vale should be calculated in metric.py
        # the method is in readme
        #  focal loss (batch_size,num_class)
        
        # loss_ = -1 * np.power(1 - pro_, self._gamma) * np.log(pro_)
        # print('loss_: ',loss_)

        # ----adapt----
        # recover data structure
        pro = mx.nd.array(pro_)
        pro = mx.nd.swapaxes(pro,0,1)
        pro = mx.nd.reshape(pro,(-4,self.shape[0],self.shape[1],-4,self.shape[2],self.shape[3]))
        # print('pro.shape: ',pro.shape)
        # ----adapt----
 
        self.assign(out_data[0],req[0],pro)

    def backward(self, req, out_grad, in_data, out_data, in_grad, aux):
        # print('----backward----')
        
        labels = self._labels.astype('int')
        # print('labels.T: ',labels.T)
        # print('labels.shape: ',labels.shape)
        pro_ = self.pro_
        
        # i!=j
        pt = self._pt + 1e-14
        # pt = self._pt
    
        pt = pt.reshape(len(pt),1)
        # pt.reshape(-1,1)
        # print('pt.shape: ',pt.shape)

        # dx = 0.75*np.power(1 - pt, self._gamma - 1) * (self._gamma * (-1 * pt * pro_) * np.log(pt) + pro_ * (1 - pt)) * 1.0 
        dx = (1-self._alpha) * np.power(1-pt, self._gamma-1) * (self._gamma * (-1*pt*pro_) * np.log(pt) + pro_ * (1-pt)) * 1.0
        
        # ----test softmax----
        # dx = pro_ 
        # ----test softmax----

        # print('dx.shape: ',dx.shape)
        # print('dx1: ',dx)
        dx1 = np.copy(dx)

        # i==j 
        # reload pt
        pt = self._pt + 1e-14
        # pt = self._pt

        ig_inds = []
        negative_inds = np.where(labels>-100)
        # negative_inds is a rank 1 array
        # negative_inds = np.where(labels>-100)[0]
        # print('negative_inds: ',negative_inds)

        if self.use_ignore:
            print('ignore_label: ',self.ignore_label)
            ig_inds= np.where(labels==self.ignore_label)
            dx[ig_inds,:] = 0
            negative_inds = np.where(labels!=self.ignore_label)
            print('negative_inds.shape: ',negative_inds[0].shape)
            pt = pt[negative_inds]
        # print('labels[negative_inds]: ',labels[negative_inds].astype('int'))
        # dx[negative_inds, labels[negative_inds].astype('int')]  = 0.25*np.power(1 - pt, self._gamma) * (self._gamma * pt * np.log(pt) + pt -1)
        tmp = self._alpha * np.power(1 - pt, self._gamma) * (self._gamma * pt * np.log(pt) + pt - 1)
        # print('tmp: ',tmp)
        # print('tmp.shape: ',tmp.shape)
        # print('i=j before: ',dx[negative_inds, labels[negative_inds].astype('int')])
        dx[negative_inds, labels[negative_inds].astype('int')] = tmp
        
        # ----test softmax----
        # dx[negative_inds, labels[negative_inds].astype('int')] -= 1.0 
        # ----test softmax----

        # print('i=j after: ',dx[negative_inds, labels[negative_inds].astype('int')])
        # dx[negative_inds.tolist(), labels[negative_inds].astype('int').tolist()]  = 0.25 * np.power(1 - pt, self._gamma) * (self._gamma * pt * np.log(pt) + pt - 1)
        # print('dx2: ',dx)
        # print('dx1!=dx2: ',dx1[np.where(dx!=dx1)],dx[np.where(dx!=dx1)])
        # print('dx.shape: ',dx.shape)
        # print('(dx1!=dx2).shape: ',np.where(dx!=dx1)[0].shape,np.where(dx!=dx1)[1].shape)

        norm_scale = np.where((labels!=0)&(labels!=self.ignore_label)) 
        # print('norm_scale: ',norm_scale)

        if len(norm_scale[0])==0:
            scale = 1
        else:
            scale = len(norm_scale[0])
        # print('scale: ',scale)
        
        # dx /= 500.0 ##batch 
        dx /= dx.shape[0]
     

        ig_labels = np.where(labels==-1)
        # print('ig_labels: ',ig_labels)
        dx[ig_labels,:]=0

        # print('dx[0]: ',dx[0])
        # ----adapt----
        # recover data structure
        # print('dx.shape: ',dx.shape)
        dx = mx.nd.array(dx)
        dx = mx.nd.swapaxes(dx,0,1)
        dx = mx.nd.reshape(dx,(-4,self.shape[0],self.shape[1],-4,self.shape[2],self.shape[3]))
        # print('dx.shape: ',dx.shape)
        # ----adapt----

        self.assign(in_grad[0], req[0], mx.nd.array(dx))
        self.assign(in_grad[1], req[1], 0)
 
        # assert False,'Debugging.'
         

@mx.operator.register('FocalLoss')
class FocalLossProp(mx.operator.CustomOpProp):
    def __init__(self, gamma,alpha,use_ignore,ignore_label):
        super(FocalLossProp, self).__init__(need_top_grad=False)
        # self.use_ignore = bool(use_ignore)
        self.use_ignore = False if use_ignore == 'False' else True
        print('use_ignore: ',self.use_ignore)
        self.ignore_label = int(ignore_label)

        self._gamma = float(gamma)
        self._alpha = float(alpha)

    def list_arguments(self):
        return ['data', 'labels']

    def list_outputs(self):
        return ['focal_loss']

    def infer_shape(self, in_shape):
        data_shape = in_shape[0]
        labels_shape = in_shape[1]
        out_shape = data_shape

        return  [data_shape, labels_shape],[out_shape]

    def create_operator(self, ctx, shapes, dtypes):
        return FocalLossOperator(self._gamma,self._alpha,self.use_ignore,self.ignore_label)

    def declare_backward_dependency(self, out_grad, in_data, out_data):
        return []
