from StorageApi import QiniuStorage
import xml.dom.minidom
import sys
import os
import json
import argparse


def xml2json(xml_file, image_path, bucket, dataset_label):
  result = {
      'source_url': '',
      "ops": "download()",
      'type': 'image',
      'label': [
          {
              'name': dataset_label,
	    	'type': "detection",
	    	'version': "1",
	    	'data': []
          }
      ]
  }

  try:
    DOMTree = xml.dom.minidom.parse(xml_file)
    collection = DOMTree.documentElement
    imgname = str(collection.getElementsByTagName(
        "filename")[0].childNodes[0].data)
    if imgname.split('.')[-1] not in ['jpg', 'JPEG']:
      img_list = os.listdir(image_path)
      img_tag = img_list[0].split('.')[-1]
      imgname = imgname + '.' + img_tag
    url = "qiniu:///" + bucket + "/" + os.path.join(image_path, imgname)
    result['url'] = url
    objects = collection.getElementsByTagName("object")

    # make detections
    detections = []
    for object in objects:
      bndbox = {}
      class_name = str(object.getElementsByTagName('name')
                       [0].childNodes[0].data)
      xmin = str(object.getElementsByTagName('xmin')[0].childNodes[0].data)
      ymin = str(object.getElementsByTagName('ymin')[0].childNodes[0].data)
      xmax = str(object.getElementsByTagName('xmax')[0].childNodes[0].data)
      ymax = str(object.getElementsByTagName('ymax')[0].childNodes[0].data)
      bndbox['class'] = class_name
      bndbox['bbox'] = [
    			     [float(xmin), float(ymin)],
    			     [float(xmax), float(ymin)],
        			 [float(xmax), float(ymax)],
        			 [float(xmin), float(ymax)]
      ]
      detections.append(bndbox)

    result['label'][0]['data'] = detections
    return "success", imgname, result
  except Exception as e:
    print "XML analyze ERROR"
    return "fail", imgname, result


def make_xmllist(file_path):
  xml_list = []
  if os.path.isfile(file_path):
    with open(file_path, "r") as fo:
      while True:
        line = str.strip(fo.readline())
        if len(line) < 1:
          break
        xml_file = line + ".xml"
        xml_list.append(xml_file)
    fo.close()
    print xml_list
    return "success", xml_list

  elif os.path.isdir(file_path):
    annotations_path = os.path.join(file_path, 'Annotations')
    split_fo = os.listdir(annotations_path)
    for fi in split_fo:
      if os.path.isfile(os.path.join(annotations_path, fi)):
        if not fi[0] == '.':
          xml_list.append(fi)
      else:
        file_fo = os.listdir(os.path.join(annotations_path, fi))
        for file_fo_fi in file_fo:
          if os.path.isfile(os.path.join(annotations_path, fi, file_fo_fi)) and file_fo_fi[0] != '.':
            xml_list.append(fi + '/' + file_fo_fi)
    return "success", xml_list
  else:
    print "file_path Error"
  return "fail", None


def make_avajson_only(xml_list, root_path, bucket, json_file, dataset_label, skip_class=''):
  annotations_path = os.path.join(root_path, 'Annotations')
  images_path = os.path.join(root_path, 'JPEGImages')
  json_array = []
  for xml in xml_list:
    xml_file = os.path.join(annotations_path, xml)
    if '/' in xml:
      images_path = os.path.join(root_path, 'JPEGImages', xml.split('/')[0])
    res, image_file, saj = xml2json(
        xml_file, images_path, bucket, dataset_label)

    skip_item = []
    for item in saj['label'][0]['data']:
      if item['class'] == skip_class:
        skip_item.append(item)
    for item in skip_item:
      saj['label'][0]['data'].remove(item)

    if saj['label'][0]['data']:
      json_array.append(saj)

  if len(json_array) > 0:
    with open(json_file, "w") as fi:
      for saj in json_array:
        json.dump(saj, fi)
        fi.write("\n")
    fi.close()
  else:
    print "No AVA Json Builded."
  return


def make_avajson__uploadimages2bucket(xml_list, root_path, bucket, json_file, dataset_label, ak, sk, skip_class=''):
  annotations_path = os.path.join(root_path, 'Annotations')
  images_path = os.path.join(root_path, 'JPEGImages')
  json_array = []
  qapi = QiniuStorage(ak, sk)
  for xml in xml_list:
    xml_file = os.path.join(annotations_path, xml)
    if '/' in xml:
      images_path = os.path.join(root_path, 'JPEGImages', xml.split('/')[0])
    res, image_file, saj = xml2json(
        xml_file, images_path, bucket, dataset_label)
    image_file = os.path.join(images_path, image_file)
    res = qapi.upload(bucket, image_file, image_file)  # upload file
    if res == "fail":
      print "upload ", res
      break
    skip_item = []
    for item in saj['label'][0]['data']:
      if item['class'] == skip_class:
        skip_item.append(item)
    for item in skip_item:
      saj['label'][0]['data'].remove(item)

    if saj['label'][0]['data']:
      json_array.append(saj)

  if len(json_array) > 0:
    with open(json_file, "w") as fi:
      for saj in json_array:
        json.dump(saj, fi)
        fi.write("\n")
    fi.close()
    res = qapi.upload(bucket, json_file, json_file)  # upload file
    if res == "fail":
      print "upload ", res
  else:
    print "No AVA Json Builded."
  return


def test_make_avajson():
  root_path = "/Users/linyining/Documents/code/atlab/ataraxia/common/dataflow"
  VOC_path = "/Users/linyining/Documents/code/VOCdevkit/VOC2007"
  train_file = os.path.join(VOC_path, "ImageSets/Layout/train.txt")
  dataset_label = "det-test"
  res, xml_list = make_xmllist(VOC_path)
  print xml_list
  if res == "success":
    make_avajson_only(xml_list, VOC_path, "test-bucket",
                      "avadet1.json", dataset_label)
  else:
  	print "make xml list ERROR"

  res, xml_list = make_xmllist(train_file)
  if res == "success":
    make_avajson_only(xml_list, VOC_path, "test-bucket",
                      "avadet2.json", dataset_label)
  else:
  	print "make xml list ERROR"


def test_xml2json(xmlfile, images_path, bucket):
  res = xml2json(xmlfile, images_path, bucket)
  res_json = json.dumps(res, indent=1)
  print res_json


def process_voc_fun():
  res, xml_list = make_xmllist(args.SRC)
  if res == "fail":
  	print "make xml list ERROR"
  if args.OP == 1:
    make_avajson_only(xml_list, args.ROOT_PATH, args.DST,
                      args.JSON_LIST, args.DATASET_LABEL, args.SKIP)
  else:
    make_avajson__uploadimages2bucket(
        xml_list, args.ROOT_PATH, args.DST, args.JSON_LIST, args.DATASET_LABEL, args.AK, args.SK, args.SKIP)
  pass



def process_coco_fun_extract_AnnoJsonFile(annoJsonFile=None):
  # read annotation json file to dict
  anno_dict = json.load(open(annoJsonFile, 'r')) 
  # extract categories info to a dict 
  categories_dict = {}  # id(int) : name (str)
  categories_value = anno_dict.get('categories')
  def processCategories(categories_value):
    for line in categories_value:
        categories_dict[line.get('id')] = line.get('name')
  processCategories(categories_value=categories_value)
  # extract annotations info to a dict
  annotations_dict = {}
  # imageIdInt (int) : annotation_list (list);
  # list element is : {"image_id":****,"category_id":***,'bbox': [200.61, 89.65, 400.22, 251.02]} (dict)  
  def processAnnotations(annotations_value):
    for line in annotations_value:
        imageIdInt = int(line.get('image_id'))
        one_annotation_dict = {}
        one_annotation_dict['image_id'] = line.get('image_id')
        one_annotation_dict['category_id'] = line.get('category_id')
        one_annotation_dict['bbox'] = line.get('bbox')
        if annotations_dict.get(imageIdInt, None):
            imageIdInt_dict = annotations_dict.get(imageIdInt)
            imageIdInt_dict.append(one_annotation_dict)
        else:
            imageIdInt_dict = []
            imageIdInt_dict.append(one_annotation_dict)
            annotations_dict[imageIdInt] = imageIdInt_dict
  annotations_value = anno_dict.get('annotations')
  processAnnotations(annotations_value)
  return categories_dict,annotations_dict

def process_coco_fun_anno2Json(image_absolute_path=None,detections=None,bucket=None,dataset_label=None):
  result = {
      'source_url': '',
      "ops": "download()",
      'type': 'image',
      'label': [
          {
              'name': dataset_label,
	    	'type': "detection",
	    	'version': "1",
	    	'data': []
          }
      ]
  }
  url = "qiniu:///" + bucket + "/" + image_absolute_path
  result['url'] = url
  result['label'][0]['data'] = detections
  return result


def process_coco_fun():
  """
  if --dataformatflag == coco 
  then --root is coco image base path (absolute)
       --src  is coco annotation json file absolute path
  """
  json_array = []
  if args.OP == 0:
    qapi = QiniuStorage(args.AK, args.SK)
  categories_dict,annotations_dict = process_coco_fun_extract_AnnoJsonFile(annoJsonFile=args.SRC)
  for image in os.listdir(args.ROOT_PATH):
      image_absolute_path = os.path.join(args.ROOT_PATH, image)
      image_id_int = int(image.split('.')[0]) # just coco image id  int type
      annotation_list=annotations_dict.get(image_id_int)
      detections=[]
      if annotation_list:
        for annotation in annotation_list:
          #{"image_id":****,"category_id":***,'bbox': [200.61, 89.65, 400.22, 251.02]}
          bndbox = {}
          bndbox['class'] = categories_dict.get(int(annotation.get('category_id')))
          xmin = annotation.get('bbox')[0]
          ymin = annotation.get('bbox')[1]
          xmax = annotation.get('bbox')[2]
          ymax = annotation.get('bbox')[3]
          bndbox['bbox'] = [
    			     [float(xmin), float(ymin)],
    			     [float(xmax), float(ymin)],
        			 [float(xmax), float(ymax)],
        			 [float(xmin), float(ymax)]
          ]
          detections.append(bndbox)
      saj = process_coco_fun_anno2Json(image_absolute_path=image_absolute_path,detections=detections,bucket=args.DST,dataset_label=args.DATASET_LABEL)
      if args.OP == 0:
        res = qapi.upload(args.DST, image_absolute_path,image_absolute_path)  # upload file
        if res == "fail":
          print ("upload %s %s"%(image_absolute_path,res))
          break
      if saj['label'][0]['data']:
        json_array.append(saj)
  if len(json_array) > 0:
    with open(args.JSON_LIST, "w") as fi:
      for saj in json_array:
        json.dump(saj, fi)
        fi.write("\n")
    fi.close()
    if args.OP == 0:
      res = qapi.upload(args.DST, args.JSON_LIST, args.JSON_LIST)  # upload file
      if res == "fail":
        print ("upload %s %s"%(args.JSON_LIST,res))
  else:
    print "No AVA Json Builded."
  return

def parse_args():
  parser = argparse.ArgumentParser(
      description='AVA Image Uploader And AVA Json Builder')
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
      '--src', dest='SRC', help='image list det format, or VOC root path', default=None, type=str
  )
  parser.add_argument(
      '--jsonlist', dest='JSON_LIST', help='JSON_LIST', default=None, type=str
  )
  parser.add_argument(
      '--root', dest='ROOT_PATH', help='VOC format root path, which contain Annotations, JPEGImages, ... ', default=None, type=str
  )
  parser.add_argument(
      '--skip', dest='SKIP', help='skip class', default='', type=str
  )
  parser.add_argument(
      '--datasetlabel', dest='DATASET_LABEL', help='DATASET_LABEL', default=None, type=str
  )
  parser.add_argument(
      '--op', dest='OP', help='operation type: 0 - upload images and make json/1 - make json only ', default=0, type=int
  )
  parser.add_argument(
      '--dataformatflag', dest='dataformatflag', help='data format upload,eg: voc,coco,imagenet', default='voc', type=str
  )
  return parser.parse_args()

args = parse_args()

if __name__ == '__main__':
  '''
  test_make_avajson()
  # test_xml2json("000005.xml", os.path.join(root_path, "JPEGImages"), "test-bucket")
  '''
  print args
  if args.dataformatflag == 'voc':
    process_voc_fun()
  elif args.dataformatflag == 'coco':
    process_coco_fun()
    
  
