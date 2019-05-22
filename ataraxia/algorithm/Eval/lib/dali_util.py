from nvidia.dali.pipeline import Pipeline 
import nvidia.dali.ops as dali_ops
import nvidia.dali.types as dali_types
from config import cfg

def get_idx_path(rec_path):
    try:
        idx_path = rec_path.replace('.rec','.idx')
    except:
        idx_path = None
    return idx_path 
    

class HybridTrainPipe(Pipeline):
    def __init__(self, batch_size, num_threads, device_id, num_gpus):
        super(HybridTrainPipe, self).__init__(batch_size, num_threads, device_id, seed = 12 + device_id)
        self.input = dali_ops.MXNetReader(
                path = [cfg.TRAIN.TRAIN_REC], 
                index_path=[get_idx_path(cfg.TRAIN.TRAIN_REC)],
                random_shuffle = cfg.TRAIN.SHUFFLE, 
                shard_id = device_id, 
                num_shards = num_gpus)
        self.decode = dali_ops.nvJPEGDecoder(device = "mixed", output_type = dali_types.RGB)
        self.rrc = dali_ops.RandomResizedCrop(device = "gpu", size = (224, 224))
        self.cmnp = dali_ops.CropMirrorNormalize(
                device = "gpu",
                output_dtype = dali_types.FLOAT,
                output_layout = dali_types.NCHW,
                crop = (224, 224),
                image_type = dali_types.RGB,
                mean = cfg.TRAIN.MEAN_RGB,
                std = cfg.TRAIN.STD_RGB)
        self.coin = dali_ops.CoinFlip(probability = 0.5)

    def define_graph(self):
        rng = self.coin()
        self.jpegs, self.labels = self.input(name = "Reader")
        images = self.decode(self.jpegs)
        images = self.rrc(images)
        output = self.cmnp(images, mirror = rng)
        return [output, self.labels]

class HybridValPipe(Pipeline):
    def __init__(self, batch_size, num_threads, device_id, num_gpus):
        super(HybridValPipe, self).__init__(batch_size, num_threads, device_id, seed = 12 + device_id)
        self.input = dali_ops.MXNetReader(
                path = [cfg.TRAIN.DEV_REC], 
                index_path=[get_idx_path(cfg.TRAIN.DEV_REC)],
                random_shuffle = False, 
                shard_id = device_id, 
                num_shards = num_gpus)
        self.decode = dali_ops.nvJPEGDecoder(device = "mixed", output_type = dali_types.RGB)
        self.cmnp = dali_ops.CropMirrorNormalize(
                device = "gpu",
                output_dtype = dali_types.FLOAT,
                output_layout = dali_types.NCHW,
                crop = (224, 224),
                image_type = dali_types.RGB,
                mean = cfg.TRAIN.MEAN_RGB,
                std = cfg.TRAIN.STD_RGB)

    def define_graph(self):
        self.jpegs, self.labels = self.input(name = "Reader")
        images = self.decode(self.jpegs)
        output = self.cmnp(images)
        return [output, self.labels]
