import mxnet as mx
import numpy as np
from collections import namedtuple, OrderedDict
Batch = namedtuple('Batch', ['data'])


class FeatureCoding(object):
	"""
	extract features for video frames.
	Parameters
	-----------------
	featureDim: input feature dim for coding
	modelPrefix: models snapshot
	modelEpoch: models snapshot
	synset: class label file
	gpu_id: which gpu to use
	"""

	def __init__(self, featureDim, batchsize, modelPrefix, modelEpoch, synset, gpu_id=0):
		self.batchsize = batchsize
		ctx = mx.gpu(gpu_id)
		sym, arg_params, aux_params = mx.model.load_checkpoint(modelPrefix, modelEpoch)
		mod = mx.mod.Module(symbol=sym, context=ctx, label_names=None)
		mod.bind(for_training=False, data_shapes=[('data', (1, self.batchsize, featureDim))],
		         label_shapes=mod._label_shapes)
		mod.set_params(arg_params, aux_params, allow_missing=True)
		self.mod = mod
		self.featureDim = featureDim
		with open(synset, 'r') as f:
			labels = [l.strip().split('\t')[-1] for l in f]
		self.labels = labels


	def classify(self,extracted_batch_feature, topN = 5):
		feature = extracted_batch_feature.reshape(1, self.batchsize, -1)
		if feature.shape[-1] != self.featureDim:
			feature = self._cut_feature(feature)

		self.mod.forward(Batch([mx.nd.array(feature)]))
		prob = self.mod.get_outputs()[0].asnumpy()
		prob = np.squeeze(prob).tolist()
		sorted_idx = np.argsort(prob)[::-1]
		topN_result = OrderedDict()
		for i in sorted_idx[0:topN]:
			topN_result[self.labels[i]] = prob[i]

		return topN_result



	def _cut_feature(self, source_fea):
		if source_fea.shape[-1] < self.featureDim:
			raise IOError(
				("FeatureCoding error: feature dimension (%d) is smaller than cut feature dimension (%d)"
				 % (source_fea.shape[-1], self.featureDim)))

		fea_len = int(source_fea.shape[-1] / self.featureDim)
		batchsize = source_fea.shape[1]
		randidx = np.random.randint(fea_len,size=batchsize)
		crop_fea = np.empty((1, self.batchsize, self.featureDim), dtype=np.float32)
		for ind, start_dim in enumerate(randidx):
			crop_fea[0,ind,:] = source_fea[0,ind,start_dim:start_dim+self.featureDim]

		return crop_fea
