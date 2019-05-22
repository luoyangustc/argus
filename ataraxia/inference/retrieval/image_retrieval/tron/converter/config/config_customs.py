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



#######################################################
################## vgg19 info ###################
#######################################################
def get_config_vgg19():
    net_info = {
        'num_class': [4096],
        'input_name': ['data'],
        'input_shape': [[1, 3, 224, 224]],
        'mean_value': [123.68, 116.779, 103.939],
        'scale_value': [],
        'arg': {},
        'out_blob': ['fc6']
    }

    meta_net_info = {
        'model_type': 'mxnet',
        'model_name': ['vgg19'],
        'model_epoch': [0],
        'save_name': 'vgg19',
        'version': '1.0.0',
        'method': 'classification',
        'network': [net_info]
    }

    return meta_net_info





