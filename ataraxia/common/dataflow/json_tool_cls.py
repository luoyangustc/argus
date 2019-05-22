######################################################
#  upload images from folder or from list to bucket  #
#  Input Args:
#    AK SK, 
#    src(list file) dest(bucket name), 
#    type(mx/cf/det), rootpath
#    synset, dataset_label
#  Output Args:
#    jsonfile 
#  Support 3 Types:
#    cf - caffe txt for classification
#    mx - mxnet lst for classification
#    det - detection txt + xml
#  Version 1:
#    no multithread, no retry
######################################################

from StorageApi import QiniuStorage
import os
import json
import argparse


def make_ava_json_cls(bucket, key, cls, synset, dataset_label):
  '''
  url, type, <source_url>, <ops>, 
  label:
    [{
      "data": [{"class": "bomb"}], 
      "version": "1", 
      "type": "classification", 
      "name": "terror"
    }]
  '''
  # url = qiniu:///<bucket_name>/<key>
  url = "qiniu:///" + bucket + "/" + key
  label_json = {"data":[{"class": synset[cls]}], "version": "1", "type": "classification", "name": dataset_label}
  ava_json = {"url": url, "ops": "download()", "type": "image", "label":[label_json]}
  return ava_json


def decode_mxnet_line(line):
  try:
    index, label, file_key = [i for i in line.split()]
    label = int(float(label))
    # index = int(index)
    ret = 0
    return ret, file_key, label
  except Exception as e:
    ret = -1
    file_key = ''
    label = "decode_mxnet_line - str format error"
    return ret, file_key, label


def decode_caffe_line(line):
  try:
    file_key, label = [i for i in line.split()]
    label = int(label)
    ret = 0
    return ret, file_key, label
  except Exception as e:
    ret = -1
    file_key = ''
    label = "decode_caffe_line - str format error"
    return ret, file_key, label


def make_avajson_only(root, list, bucket, syn_set, dataset_label, list_type, json_file):
  '''
  CAFFE format:
  <file> <cls(int)>
  MXNET format:
  <index(int)> <label(float)> <file>
  '''
  json_array = []
  with open(list) as fo:
    while True:
      line = str.strip(fo.readline())
      if len(line) < 10:
        break
      if list_type == "mx":
        ret, file_key, label = decode_mxnet_line(line)
      elif list_type == "cf":
        ret, file_key, label = decode_caffe_line(line)
      else:
        print "list_type Not Supported"
        break
      if not ret == 0:
        print label
        break
      if list_type == "mx" or list_type == "cf":
        saj = make_ava_json_cls(bucket, file_path, label, syn_set, dataset_label) # make json
      else:
        print "list_type Not Supported"
        break
      json_array.append(saj)
  fo.close()

  if len(json_array) > 0:
    with open(json_file,"w") as fi:
      for saj in json_array:
        json.dump(saj, fi)
        fi.write("\n")
    fi.close()
  else:
    print "No AVA Json Builded."
  return


def upload_images_from_clslist(root, list, bucket, syn_set, dataset_label, list_type, json_file, ak, sk):
  '''
  CAFFE format:
  <file> <cls(int)>
  MXNET format:
  <index(int)> <label(float)> <file>
  '''
  json_array = []
  qapi = QiniuStorage(ak, sk)
  with open(list) as fo:
    while True:
      line = str.strip(fo.readline())
      if len(line) < 10:
        break
      if list_type == "mx":
        ret, file_key, label = decode_mxnet_line(line)
      elif list_type == "cf":
        ret, file_key, label = decode_caffe_line(line)
      else:
        print "list_type Not Supported"
        break
      if not ret == 0:
        print label
        break
      file_path = os.path.join(root, file_key)
      print bucket, file_key, file_path, label
      ret = qapi.upload(bucket, file_path, file_path) # upload file
      if ret == "fail":
        print "upload ", ret
        break
      if list_type == "mx" or list_type == "cf":
        saj = make_ava_json_cls(bucket, file_path, label, syn_set, dataset_label) # make json
      else:
        print "list_type Not Supported"
        break
      json_array.append(saj)
  fo.close()

  if len(json_array) > 0:
    with open(json_file,"w") as fi:
      for saj in json_array:
        json.dump(saj, fi)
        fi.write("\n")
    fi.close()
    res = qapi.upload(bucket, json_file, json_file) # upload file
    if res == "fail":
      print "upload ", res
  else:
    print "No AVA Json Builded."
  return


def load_synset(synset_file):
  syn_set = []
  with open(synset_file) as fo:
    while True:
      line = str.strip(fo.readline())
      if len(line) < 1:
        break
      syn_set.append(line)
  fo.close()
  return syn_set


def test_avajson(bucket, syn_set, dataset_label, json_file):
  file_jsons = []
  with open("cf.list") as fo:
    while True:
      line = str.strip(fo.readline())
      if len(line) < 10:
        break
      ret, file_key, label = decode_caffe_line(line)
      if not ret == 0:
        print label
        break
      print file_key, label
      saj = make_ava_json_cls(bucket, file_key, label, syn_set, dataset_label) # make json
      print saj
      file_jsons.append(saj)
  fo.close()
  with open(json_file,"w") as f:
    for saj in file_jsons:
      json.dump(saj, f)
      f.write("\n")
#  ret = qapi.upload(bucket_name, file_jsons, file_jsons) # upload file
  return


def test_decode():

  with open("mx.list") as fo:
    while True:
      line = str.strip(fo.readline())
      if len(line) < 10:
        break
      ret, file_key, label = decode_mxnet_line(line)
      if not ret == 0:
        print label
        break
      print file_key, label
  fo.close()

  with open("cf.list") as fo:
    while True:
      line = str.strip(fo.readline())
      if len(line) < 10:
        break
      ret, file_key, label = decode_caffe_line(line)
      if not ret == 0:
        print label
        break
      print file_key, label
  fo.close()


def parse_args():
  parser = argparse.ArgumentParser(description='AVA Image Uploader And AVA Json Builder')
  parser.add_argument(
      '--ak', dest='AK', default=None, type=str
  )
  parser.add_argument(
      '--sk', dest='SK', default=None, type=str
  )
  parser.add_argument(
      '--dest', dest='DST', help='destination bucket', default=None, type=str
  )
  parser.add_argument(
      '--src', dest='SRC', help='image list in caffe/mxnet/det format', default=None, type=str
  )
  parser.add_argument(
      '--type', dest='TYPE', help='list format: caffe/mxnet/det ', default=None, type=str
  )
  parser.add_argument(
      '--synset', dest='SYNSET', help='SYNSET File', default=None, type=str
  )
  parser.add_argument(
      '--datasetlabel', dest='DATASET_LABEL', help='DATASET_LABEL', default=None, type=str
  )
  parser.add_argument(
      '--jsonlist', dest='JSON_LIST', help='JSON_LIST', default=None, type=str
  )
  parser.add_argument(
      '--root', dest='ROOT_PATH', help='root path of images', default=None, type=str
  )
  parser.add_argument(
      '--op', dest='OP', help='operation type: 0 - upload images and make json/1 - make json only ', default=0, type=int
  )

  return parser.parse_args()

if __name__ == '__main__':

  '''
  bucket = "test_lyn"
  syn_set = ["z0", "z1", "z2", "z3", "z4", "z5", "z6"]
  dataset_label = "test_label"
  json_file = "ava.json"
  test_avajson(bucket, syn_set, dataset_label, json_file)
#  test_decode()

  AK = sys.argv[1]
  SK = sys.argv[2]
  SRC = sys.argv[3]
  DST = sys.argv[4]
  TYPE = sys.argv[5]
  SYNSET = sys.argv[6]
  DATASET_LABEL = sys.argv[7]
  JSON_LIST = sys.argv[8]
  '''
  args = parse_args()

  syn_set = load_synset(args.SYNSET)
  if args.OP == 1:
    make_avajson_only(args.ROOT_PATH, args.SRC, syn_set, args.DATASET_LABEL, args.TYPE, args.JSON_LIST)
  else:
    upload_images_from_clslist(args.ROOT_PATH, args.SRC, args.DST, syn_set, args.DATASET_LABEL, args.TYPE, args.JSON_LIST, args.AK, args.SK)
