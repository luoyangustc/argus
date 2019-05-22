import mxnet as mx

data = mx.sym.Variable('data')
rois = mx.sym.Variable('rois')

operator = mx.symbol.ROIAlign(data=data, rois=rois, pooled_size=(2,2),spatial_scale=1.0)
arg_name = operator.list_arguments()  # get the names of the inputs
out_name = operator.list_outputs()    # get the names of the outputs

# infer shape
arg_shape, out_shape, _ = operator.infer_shape(data=(1,1,6,6), rois=(1,5))
{'input' : dict(zip(arg_name, arg_shape)),
 'output' : dict(zip(out_name, out_shape))}

#Bind with Data and Evaluate
feat_map = mx.nd.array([[[[  0.,   1.,   2.,   3.,   4.,   5.],
                          [  6.,   7.,   8.,   9.,  10.,  11.],
                          [ 12.,  13.,  14.,  15.,  16.,  17.],
                          [ 18.,  19.,  20.,  21.,  22.,  23.],
                          [ 24.,  25.,  26.,  27.,  28.,  29.],
                          [ 30.,  31.,  32.,  33.,  34.,  35.],
                          [ 36.,  37.,  38.,  39.,  40.,  41.],
                          [ 42.,  43.,  44.,  45.,  46.,  47.]]]])

rois_ = mx.nd.array([[0,0,0,4,4]])

top_grad = mx.nd.empty([1,1,8,6])
top_grad_gpu = mx.nd.empty([1,1,8,6]).as_in_context(mx.gpu(0))

# ROIAlign CPU Test
ex = operator.bind(ctx=mx.cpu(), args={'data' : feat_map, 'rois' : rois_}, args_grad={'data' : top_grad}, grad_req='write')
ex.forward()
ex.backward(out_grads=[mx.nd.ones([1,1,2,2])])
print 'ROIAlign-CPU:\ninput feature map = \n%s \ninput rois = \n%s \n number of outputs = %d\nthe first output = \n%s\ngrad:\n%s' % (
           feat_map.asnumpy(), rois_.asnumpy(), len(ex.outputs), ex.outputs[0].asnumpy(), top_grad.asnumpy())

# ROIAlign GPU Test
operator2 = mx.symbol.ROIAlign(data=data, rois=rois, pooled_size=(2,2),spatial_scale=1.0)
ex2 = operator2.bind(ctx=mx.gpu(0), args={'data' : feat_map.as_in_context(mx.gpu(0)), 'rois' : rois_.as_in_context(mx.gpu(0))}, \
                    args_grad={'data': top_grad_gpu}, grad_req='write')
ex2.forward()
ex2.backward(out_grads=[mx.nd.ones([1,1,2,2])])
print 'ROIAlign-GPU:\nnumber of outputs = %d\nthe first output = \n%s\ngrad:\n%s' % (
           len(ex2.outputs), ex2.outputs[0].asnumpy(),top_grad_gpu.asnumpy())


#ROIPooling CPU Test
operator_roipool = mx.symbol.ROIPooling(data=data, rois=rois, pooled_size=(2,2),spatial_scale=1.0)
ex_roipool = operator_roipool.bind(ctx=mx.cpu(), args={'data' : feat_map, 'rois' : rois_}, \
                                   args_grad={'data': top_grad}, grad_req='write')
ex_roipool.forward()
ex_roipool.backward(out_grads=[mx.nd.ones([1,1,2,2])])
print 'ROIPooling:\nnumber of outputs = %d\nthe first output = \n%s\ngrad:\n%s' % (
           len(ex_roipool.outputs), ex_roipool.outputs[0].asnumpy(),  top_grad.asnumpy())
