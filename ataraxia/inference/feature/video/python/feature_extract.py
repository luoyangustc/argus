import numpy as np
import caffe



class FeatureExtraction(object):
    """
    extract features for video frames.
    
    Parameters
    -----------------
    video: Video
    modelPrototxt: models architecture file
    modelFile: models snapshot
    featureLayer: which layer to be extracted as feature
    gpu_id: which gpu to use
    """

    def __init__(self, modelPrototxt, modelFile, featureLayer, gpu_id=0):
        caffe.set_mode_gpu()
        caffe.set_device(gpu_id)

        self.net = caffe.Net(modelPrototxt, modelFile, caffe.TEST)
        data_shape = self.net.blobs['data'].data.shape
        self.batchsize = data_shape[0]
        self.height = data_shape[2]
        self.width = data_shape[3]

        self.featureLayer = featureLayer
        featureDim = self.net.blobs[featureLayer].data.shape
        print "featureDim:", featureDim

        transformer = caffe.io.Transformer({'data': data_shape})
        transformer.set_transpose('data', (2, 0, 1))
        transformer.set_mean('data', np.array([103.939, 116.779, 123.68]))  # mean pixel
        transformer.set_raw_scale('data', 255)
        transformer.set_channel_swap('data', (2, 1, 0))
        self.transformer = transformer

    def ext_process(self,img_list):
        import cv2
        num_img = len(img_list)
        num_batch = int(num_img / self.batchsize)
        num_left = num_img % self.batchsize
        batch_count = [self.batchsize for _ in range(num_batch)]
        if num_left > 0:
            batch_count.append(num_left)

        img_count = 0
        im_group = np.empty((self.batchsize, 3, self.height, self.width), dtype=np.float32)
        is_first_batch = True
        for batch_num in batch_count:
            for ix in range(batch_num):
                img = img_list[img_count + ix]
                img = cv2.resize(img, (225, 225))
                img = img.astype(np.float32, copy=True)
                img -= np.array([[[103.94, 116.78, 123.68]]])
                img = img * 0.017
                img = img.transpose((2, 0, 1))
                im_group[ix] = img
            self.net.blobs['data'].data[...] = im_group
            out = self.net.forward()
            feature = np.squeeze(out[self.featureLayer])
            if is_first_batch:
                all_feature = feature[:batch_num, :]
                is_first_batch = False
            else:
                all_feature = np.vstack((all_feature, feature[:batch_num, :]))
            img_count += batch_num

        print np.shape(all_feature)
        return all_feature

    def __call__(self, video):
        for timestamps, frames in video: # frames are rgb channel-ordered
            features = self.ext_process(frames)

            yield timestamps, frames, features