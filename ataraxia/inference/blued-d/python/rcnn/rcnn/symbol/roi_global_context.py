"""ROI Global Context Operator enlarges the rois with its surrounding areas, to provide contextual information
"""

from __future__ import print_function
import mxnet as mx

DEBUG = False


class ROIGlobalContextOperator(mx.operator.CustomOp):
    def __init__(self, global_context_scale):
        super(ROIGlobalContextOperator, self).__init__()
        self._global_context_scale = global_context_scale


    def forward(self, is_train, req, in_data, out_data, aux):
        x = in_data[0].copy()  # rois=[cls, x1, y1, x2, y2]
        im_info = in_data[1].copy()   # im_info=(height,width)
        #im_info = mx.ndarray.slice_axis(im_info, axis=0, begin=0, end=1)
        if DEBUG:
            print('im_info:{},rois_shape:{}'.format(im_info.asnumpy(), x.shape))
        y = out_data[0]

        rois_cls = mx.ndarray.slice_axis(x, axis = 1, begin=0, end=1)
        rois_x1 = mx.ndarray.slice_axis(x, axis = 1, begin=1, end=2)
        rois_x2 = mx.ndarray.slice_axis(x, axis = 1, begin=3, end=4)
        rois_y1 = mx.ndarray.slice_axis(x, axis = 1, begin=2, end=3)
        rois_y2 = mx.ndarray.slice_axis(x, axis = 1, begin=4, end=5)

        rois_ctr_x = 0.5 * (rois_x1 + rois_x2)
        rois_ctr_y = 0.5 * (rois_y1 + rois_y2)
        rois_w_half = rois_ctr_x - rois_x1
        rois_h_half = rois_ctr_y - rois_y1
        rois_w_half_new = self._global_context_scale * rois_w_half
        rois_h_half_new = self._global_context_scale * rois_h_half

        y[:,0] = rois_cls
        y[:,1] = rois_ctr_x - rois_w_half_new
        y[:,2] = rois_ctr_y - rois_h_half_new
        y[:,3] = rois_ctr_x + rois_w_half_new
        y[:,4] = rois_ctr_y + rois_h_half_new
        y = self.clip_boxes(y, im_info)
        if DEBUG:
            print('y.shape:',y.shape)



    def backward(self, req, out_grad, in_data, out_data, in_grad, aux):
        self.assign(in_grad[0], req[0], 0)


    @staticmethod
    def clip_boxes(box, im_shape):
        """
        Clip boxes to image boundaries.
        :param boxes: [1, 5]
        :param im_shape: tuple of 2
        :return: [1, 5]
        """
        im_shape = im_shape[0]
        rois_x1 = mx.ndarray.slice_axis(box, axis=1, begin=1, end=2)
        rois_x2 = mx.ndarray.slice_axis(box, axis=1, begin=3, end=4)
        rois_y1 = mx.ndarray.slice_axis(box, axis=1, begin=2, end=3)
        rois_y2 = mx.ndarray.slice_axis(box, axis=1, begin=4, end=5)
        # x1 >= 0
        box[:,1] = mx.nd.maximum(mx.nd.minimum(rois_x1, im_shape[1] - 1), 0)
        # y1 >= 0
        box[:,2] = mx.nd.maximum(mx.nd.minimum(rois_y1, im_shape[0] - 1), 0)
        # x2 < im_shape[1]
        box[:,3] = mx.nd.maximum(mx.nd.minimum(rois_x2, im_shape[1] - 1), 0)
        # y2 < im_shape[0]
        box[:,4] = mx.nd.maximum(mx.nd.minimum(rois_y2, im_shape[0] - 1), 0)
        return box




@mx.operator.register('roi_global_context')
class ROIGlobalContextProp(mx.operator.CustomOpProp):
    def __init__(self, global_context_scale = '1.2'):
        super(ROIGlobalContextProp, self).__init__(need_top_grad=False)
        self._global_context_scale =float(global_context_scale)

    def list_arguments(self):
        return ['rois','im_info']

    def list_outputs(self):
        return ['rois_output']

    def infer_shape(self, in_shape):
        rpn_rois_shape = in_shape[0]
        im_info_shape = in_shape[1]
        output_rois_shape = rpn_rois_shape

        return [rpn_rois_shape, im_info_shape], \
               [output_rois_shape]

    def create_operator(self, ctx, shapes, dtypes):
        return ROIGlobalContextOperator(self._global_context_scale)

    def declare_backward_dependency(self, out_grad, in_data, out_data):
        return []
