#######################################################
##################### custom info #####################
#######################################################
def get_config_terror_pulp():
    net_info = {
        'num_class': [48],  # class numbers
        'input_name': ['data'],  # net input blob names
        'input_shape': [[1, 3, 168, 168]],  # net input blob shapes
        'mean_value': [103.94,116.78,123.68],  # data mean values
        'scale_value': [0.017],  # data scale values
        'arg': {
            'labels_v_s': ['0_terror', '0_terror', '0_terror', '0_terror',
                           '0_terror', '0_terror', '0_terror', '2_march',
                           '2_march', '0_terror', '0_terror', '3_text',
                           '0_terror', '0_terror', '0_terror', '0_terror',
                           '0_terror', '0_terror', '0_terror', '4_normal',
                           '1_pulp', '4_normal', '4_normal', '4_normal',
                           '4_normal', '4_normal', '4_normal', '4_normal',
                           '3_text', '3_text', '4_normal', '4_normal',
                           '4_normal', '4_normal', '4_normal', '4_normal',
                           '4_normal', '4_normal', '4_normal', '4_normal',
                           '4_normal', '4_normal', '2_march', '2_march',
                           '4_normal', '4_normal', '4_normal', '4_normal']
        },  # net arguments, must end with one of 's_i, s_f, s_s, v_i, v_f, v_s'
        'out_blob': ['prob']  # net output blob names
    }

    meta_net_info = {
        'model_type': 'caffe',  # model type: caffe or mxnet
        'model_name': ['weights-0521'],  # model name on disk
        'model_epoch': [],  # Only for mxnet model
        'save_name': 'weights-0521',  # Tron model save name
        'version': '0.0.1',  # model version
        'method': 'classification',  # model category: classification, faster, ssd, mtcnn
        'network': [net_info]  # networks
    }

    return meta_net_info
