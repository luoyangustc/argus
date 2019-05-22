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
    ##################### face detection ##################
    #######################################################
def get_config_refinedet():
    net_info = {
        'num_class': [2],  # class numbers
        'input_name': ['data'],  # net input blob names
        'input_shape': [[1, 3, 512, 512]],  # net input blob shapes
        'mean_value': [104, 117, 123],  # data mean values
        'scale_value': [0.017],  # data scale values
        'arg': {'selected_labels_v_i': [1]},  # net arguments, must end with one of 's_i, s_f, s_s, v_i, v_f, v_s'
        'out_blob': ['odm_loc', 'odm_conf_flatten', 'arm_priorbox', 'arm_conf_flatten', 'arm_loc']  # net output blob names
    }

    meta_net_info = {
        'model_type': 'caffe',  # model type: caffe or mxnet
        'model_name': ['refinedet_v0.0.3'],  # model name on disk
        'model_epoch': [0],  # Only for mxnet model
        'save_name': 'refinedet_v0.0.3',  # Tron model save name
        'version': '0.0.3',  # model version
        'method': 'refinedet',  # model category: classification, faster, ssd, mtcnn
        'network': [net_info]  # networks
    }

    return meta_net_info

    #######################################################
    ##################### face quality ####################
    #######################################################
def get_config_quality():

        net_info = {
            'num_class': [0],
            'input_name': ['data'],
            'input_shape': [[16, 3, 96, 96]],
            'mean_value': [127.5],
            'scale_value': [0.0078125],
            'arg': {},
            'out_blob': ['prob', 'fc6-pts']
        }

        meta_net_info = {
            'model_type': 'caffe',
            'model_name': ['quality_v0.0.1'],
            'model_epoch': [],
            'save_name': 'quality_v0.0.1',
            'version': '0.0.1',
            'method': 'quality',
            'network': [net_info]
        }

        return meta_net_info

def get_config_quality_v2():

        net_info = {
            'num_class': [0],
            'input_name': ['data'],
            'input_shape': [[16, 3, 96, 96]],
            'mean_value': [127.5],
            'scale_value': [0.0078125],
            'arg': {},
            'out_blob': ['softmax', 'pose_softmax']
        }

        meta_net_info = {
            'model_type': 'caffe',
            'model_name': ['quality_v0.0.3'],
            'model_epoch': [],
            'save_name': 'quality_v0.0.3',
            'version': '0.0.3',
            'method': 'quality',
            'network': [net_info]
        }

        return meta_net_info

def get_config_orient():

        net_info = {
            'num_class': [0],
            'input_name': ['data'],
            'input_shape': [[16, 3, 96, 96]],
            'mean_value': [127.5],
            'scale_value': [0.0078125],
            'arg': {},
            'out_blob': ['orientation_softmax', 'pred_angle']
        }

        meta_net_info = {
            'model_type': 'caffe',
            'model_name': ['orientation_v0.0.3'],
            'model_epoch': [],
            'save_name': 'orientation_v0.0.3',
            'version': '0.0.3',
            'method': 'orientation',
            'network': [net_info]
        }

        return meta_net_info
