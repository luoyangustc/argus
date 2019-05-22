'''
input: region proposals to be merged, in .pkl format, split by ','
output: merged region proposal, in .pkl format

usage:
python -m rcnn.tools.merge_region_proposal --pklfileIn data/cache/imagenet__val_detections1.pkl,data/cache/imagenet__val_detections2.pkl --pklfileOut a.pkl
'''

from __future__ import print_function
import argparse
import cPickle
import numpy as np

def merge_pklfileIn(pklfileIn, pklfileOut, num_reserve):
    pklfileIn = pklfileIn.split(',')
    all_boxes = []
    num_images = -1
    for detfile_idx, detfile in enumerate(pklfileIn):
        with open(detfile.strip(), 'r') as f:
            print("loading file:{}".format(detfile.strip()))
            rec = cPickle.load(f)
            if num_images == -1:
                num_images = len(rec)
                all_boxes = rec
                print("num_images:{}".format(num_images))
                for img_idx in xrange(num_images):
                    print("processing {}-th file:{}".format(detfile_idx,img_idx))
                    all_boxes[img_idx] = rec[img_idx][:num_reserve,:]

            else:
                for img_idx in xrange(num_images):
                    print("processing {}-th file:{}".format(detfile_idx,img_idx))
                    #print(all_boxes[cls_idx][img_idx], rec_per_model[cls_idx][img_idx])
                    if rec[img_idx].size == 0:
                        continue
                    elif all_boxes[img_idx].size == 0:
                        all_boxes[img_idx] = rec[img_idx][:num_reserve,:]
                    else:
                        all_boxes[img_idx]\
                            = np.concatenate((all_boxes[img_idx], rec[img_idx][:num_reserve,:]), axis=0)
        f.close()

    if num_images != -1:
        with open(pklfileOut, 'wb') as f:
            cPickle.dump(all_boxes, f, protocol=cPickle.HIGHEST_PROTOCOL)
        print("save to {}".format(pklfileOut))
        f.close()


def parse_args():
    parser = argparse.ArgumentParser(description='Post processing')

    parser.add_argument('--pklfileIn', help='bbox results from different models, split by ,', type=str)
    parser.add_argument('--pklfileOut', help='merge bbox result', type=str)
    parser.add_argument('--numReserve', help='number of bboxes to reserve in each model result', type=int, default=1200)

    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    print(args)
    merge_pklfileIn(args.pklfileIn, args.pklfileOut, args.numReserve)


if __name__ == '__main__':
    main()



