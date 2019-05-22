#######################################################
##################### custom info #####################
#######################################################
def get_config_custom():
    net_info = {
        'num_class': [1000],  # class numbers
        'input_name': ['data'],  # net input blob names
        'input_shape': [[1, 3, 224, 224]],  # net input blob shapes
        'mean_value': [],  # data mean values
        'scale_value': [],  # data scale values
        'arg': {},  # net arguments, must end with one of 's_i, s_f, s_s, v_i, v_f, v_s'
        'out_blob': []  # net output blob names
    }

    meta_net_info = {
        'model_type': 'mxnet',  # model type: caffe or mxnet
        'model_name': ['squeezenet_v1.1'],  # model name on disk
        'model_epoch': [0],  # Only for mxnet model
        'save_name': 'squeezenet_v1.1',  # Tron model save name
        'version': '0.0.1',  # model version
        'method': 'classification',  # model category: classification, faster, ssd, mtcnn
        'network': [net_info]  # networks
    }

    return meta_net_info

def get_config_mobilenetv2():
    mobilenetv2_info = {
        'num_class': [0],
        'input_name': ['data'],
        'input_shape': [[1,3,128,128]],
        'mean_value': [103.94, 116.78, 123.68],
        'scale_value': [0.017],
        'arg': {},
        'out_blob': ['output_points','output_aspects']
    }
    meta_net_info = {
        'model_type': 'caffe',
        'model_name': ['mobilenetv2-ld-pos-v5_iter_70000_nobn'],
        'model_epoch': [],
        'save_name': 'mobilenetv2',
        'version': '0.0.1',
        'method': 'mobilenetv2',
        'network': [mobilenetv2_info]
    }
    return meta_net_info
