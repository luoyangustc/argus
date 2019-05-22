import csv
import json

from .error import ErrorConfig
from .check import value_check


def make_synset(path_csv):
    '''
        generate synset list from synset csv
        csv format:
        <label_index>,"<label_name>"
        ...
        synset_list: [[class_index, class_name],...]
    '''
    synset_list = []
    with open(path_csv, 'r') as file_csv:
        if file_csv:
            read = csv.reader(file_csv)
            synset_list = [r for r in read]
    return synset_list


def _make_tag(path_tag):
    '''
        generate tags dict from taglist file
    '''
    buff = ''
    with open(path_tag, 'r') as ftag:
        for line in ftag:
            buff += line.strip()
    tag = json.loads(buff)
    return tag


def infer_input_unmarshal(model, args):
    '''
        unmarshal inference input to net, reqid, reqs
    '''
    net = model["net"]
    args = json.loads(args)
    reqid = args["reqid"]
    reqs = args["reqs"]
    return net, reqid, reqs


def infer_output_marshal(ret, headers=None):
    '''
        marshal inference output to json string
    '''
    output = {"results": ret}
    if headers is not None:
        output["headers"] = headers.marshal()
    return json.dumps(output)


def parse_params_file(params_file_name):
    '''
    parse the 'params' file whose filename has
    pattern `prefix_model`-`epoch`.params
    :params params_file_name params filename
    '''
    elems = params_file_name[:-len('.params')].split('-')

    return '-'.join(elems[:-1]), int(elems[-1])


def parse_crop_size(configs,
                    model_params=None,
                    custom_values=None,
                    default_image_width=0,
                    default_image_height=0):
    '''
        RETURN
            image_width, image_height
    '''
    import re

    if model_params is None:
        model_params = value_check(configs, 'model_params', False, {})
    if custom_values is None:
        custom_values = value_check(configs, 'custom_values', False, {})

    crop_size = value_check(model_params, 'cropSize', False, "")
    if crop_size != "":
        groups = re.match('^([0-9]+)(x([0-9]+))?$', crop_size).groups()
        if len(groups) != 3:
            raise ErrorConfig("model_params.cropSize")
        if groups[2] is None:
            return int(groups[0]), int(groups[0])
        return int(groups[0]), int(groups[2])

    image_width = value_check(configs, 'image_width', False, 0)
    if image_width == 0:
        image_width = value_check(custom_values, 'image_width', False, 0)
    if image_width == 0:
        image_width = default_image_width

    image_height = value_check(configs, 'image_height', False, 0)
    if image_height == 0:
        image_height = value_check(custom_values, 'image_height', False, 0)
    if image_height == 0:
        image_height = default_image_height
    if image_height == 0:
        image_height = image_width

    return image_width, image_height
