import mxnet as mx
import roi_global_context

rois = mx.sym.Variable('rois')
im_info = mx.sym.Variable('im_info')

rois_globalcontext= mx.symbol.Custom(rois = rois, im_info = im_info, global_context_scale = 1.2,\
                                     op_type='roi_global_context')
print "rois_globalcontext:", rois_globalcontext

rois_globalcontext.list_arguments()
rois_globalcontext.list_outputs()  

arg_shape, out_shape, _ = rois_globalcontext.infer_shape(rois=(1,5), im_info=(1,2))
print "arg_shape:", arg_shape
print "out_shape:", out_shape

rois_ = mx.nd.array([[0,0,0,4,4]])
im_info_ = mx.nd.array([[5,7]])
ex = rois_globalcontext.bind(ctx=mx.cpu(), args={'rois' : rois_, 'im_info' : im_info_})
ex.forward()
print 'roi_global_context-CPU:\ninput rois = \n%s \ninput im_shape = \n%s \nnumber of outputs = %d\nthe first output = \n%s' % (
           rois_.asnumpy(), im_info_.asnumpy(), len(ex.outputs), ex.outputs[0].asnumpy())
