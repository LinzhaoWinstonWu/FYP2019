#!/usr/bin/env python

# --------------------------------------------------------
# Tensorflow Faster R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Xinlei Chen, based on code from Ross Girshick
# --------------------------------------------------------

"""
Demo script showing detections in sample images.

See README.md for installation instructions before running.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths
from model.config import cfg
from model.test import im_detect
from model.nms_wrapper import nms

from colorthief import ColorThief
import sys
sys.path.append("/home/weijun/colory")
from colors import Color
from PIL import Image
from utils.timer import Timer
import tensorflow as tf
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os, cv2
import argparse
import shutil
import time
import glob
import json
import re

from nets.vgg16 import vgg16
from nets.resnet_v1 import resnetv1
from tqdm import tqdm

#CLASSES = ('__background__', 'button', 'imagebutton', 'seekbar', 'checkbox', 'radiobutton', 'edittext', 'cirprogressbar', 'hrzprogressbar')
CLASSES = ('__background__',  # always index 0
                    'button', 'imagebutton', 'seekbar', 'checkbox', 'radiobutton', 'edittext')
                    #'button', 'imagebutton', 'seekbar', 'checkbox', 'radiobutton', 'edittext', 
                    #'chronometer', 'compoundbutton', 'progressbar', 'ratingbar', 'spinner', 'switch', 'togglebutton')
NETS = {'vgg16': ('vgg16_faster_rcnn_iter_70000.ckpt',),'res101': ('res101_faster_rcnn_iter_140000.ckpt',)}
DATASETS= {'pascal_voc': ('voc_2007_trainval',),'pascal_voc_0712': ('voc_2007_trainval+voc_2012_trainval',),'android_voc': ('android_train',)}
CWD = os.getcwd()
#DIR = os.path.join(CWD, "widget_clippings_google")
DIR = "/data_raid5/weijun/FYP_paper/widget_clippings_google"

meta_dump = {}
for cls in CLASSES[1:]:
    meta_dump[cls] = []
count = 1

def vis_detections(fig, ax, class_name, dets, thresh=0.5):
    """Draw detected bounding boxes."""
    inds = np.where(dets[:, -1] >= thresh)[0]
    if len(inds) == 0:
        return

    #im = im[:, :, (2, 1, 0)]
    #fig, ax = plt.subplots(figsize=(12, 12))
    #ax.imshow(im, aspect='equal')
    for i in inds:
        bbox = dets[i, :4]
        score = dets[i, -1]



        ax.add_patch(
            plt.Rectangle((bbox[0], bbox[1]),
                          bbox[2] - bbox[0],
                          bbox[3] - bbox[1], fill=False,
                          edgecolor='red', linewidth=3.5)
            )
        ax.text(bbox[0], bbox[1] - 2,
                '{:s} {:.3f}'.format(class_name, score),
                bbox=dict(facecolor='blue', alpha=0.5),
                fontsize=14, color='white')

    
    #plt.axis('off')
    #plt.tight_layout()
    #plt.draw()

def create_clippings(image_name, class_name, dets, thresh=0.5):
    global count
    """Draw detected bounding boxes."""
    inds = np.where(dets[:, -1] >= thresh)[0]
    if len(inds) == 0:
        return

    widget_dir = os.path.join(DIR, class_name) 
    im = Image.open(image_name)
    for i in inds:
        duplicate = False
        bbox = dets[i, :4]
        score = dets[i, -1]
        meta_data = {}

        try:
            clip = im.crop((bbox[0], bbox[1], bbox[2], bbox[3]))
        except OSError as err:
            print("[-] OSError - " + str(err))
            continue
        except IndexError as err:
            print("[-] IndexError - " + str(err))
            continue
        except IOError as err: #image file is truncated
            print("[-] IOError - " + str(err))
            continue

        filename = "clipping-" + str(count)
        filepath = os.path.join(widget_dir, filename + ".png")
        if not os.path.exists(widget_dir):
            os.makedirs(widget_dir)
        clip.save(filepath)

        try:
            ct = ColorThief(filepath)
            # get the dominant color
            dominant_color = ct.get_color(quality=1)
            meta_data['color'] = getColor(dominant_color)
        except Exception as e:
            os.remove(filepath)
            continue

        if not meta_dump[class_name]:
            meta_dump[class_name].append(filepath)
        else:
            for f in meta_dump[class_name]:
                diff_score = compareHisto(filepath, f)
                if diff_score < 0.13:
                    os.remove(filepath)
                    duplicate = True
                    break
                
        if not duplicate:
            meta_dump[class_name].insert(0, filepath)
            meta_data['widget_class'] = class_name
            meta_data['dimensions'] = {}
            meta_data['dimensions']['width'] = int(round(bbox[2] - bbox[0]))
            meta_data['dimensions']['height'] = int(round(bbox[3] - bbox[1]))
            meta_data['coordinates'] = {}
            meta_data['coordinates']['from'] = [int(round(bbox[0])), int(round(bbox[1]))]
            meta_data['coordinates']['to'] = [int(round(bbox[2])), int(round(bbox[3]))] 
            meta_data['src'] = image_name
            head, tail = os.path.split(image_name)
            pkg = re.match(r'.+(?=(_\d+.png))', tail).group().replace('_', '.')
            print(pkg)
            meta_data['url'] = 'https://play.google.com/store/apps/details?id='+pkg
            meta_data['package_name'] = pkg

            filepath = filepath.replace(".png", ".txt")
            with open(filepath, "a+") as f:
                json.dump(meta_data, f, indent=3, separators=(',', ': '))

            count += 1

def getColor(rgb):
    hex = "#{:02x}{:02x}{:02x}".format(rgb[0], rgb[1], rgb[2])
    color = Color(hex)
    return color.name

def compareHisto(first, sec):
    imA = Image.open(first)
    imB = Image.open(sec)

    # Normalise the scale of images 
    if imA.size[0] > imB.size[0]:
        imA = imA.resize((imB.size[0], imA.size[1]))
    else:
        imB = imB.resize((imA.size[0], imB.size[1]))

    if imA.size[1] > imB.size[1]:
        imA = imA.resize((imA.size[0], imB.size[1]))
    else:
        imB = imB.resize((imB.size[0], imA.size[1]))

    hA = imA.histogram()
    hB = imB.histogram()
    sum_hA = 0.0
    sum_hB = 0.0
    diff = 0.0

    for i in range(len(hA) if len(hA) <= len(hB) else len(hB)):
        #print(sum_hA)
        sum_hA += hA[i]
        sum_hB += hB[i]
        diff += abs(hA[i] - hB[i])

    return diff/(2*max(sum_hA, sum_hB))

def demo(sess, net, image_name):
    """Detect object classes in an image using pre-computed object proposals."""

    # Load the demo image
    #im_file = os.path.join(folder, image_name)
    #im_file = "/data_raid5/weijun/android_data/PNGImages/" + image_name
    #im_file = "data/VOCdevkit/android_data/PNGImages/" + image_name
    im = cv2.imread(image_name)
 
    im = im[:, :, (2, 1, 0)]
    #fig, ax = plt.subplots(figsize=(12, 12))
    #ax.imshow(im, aspect='equal')

    # Detect all object classes and regress object bounds
    timer = Timer()
    timer.tic()
    scores, boxes = im_detect(sess, net, im)
    timer.toc()
    #print('Detection took {:.3f}s for {:d} object proposals'.format(timer.total_time, boxes.shape[0]))

    # Visualize detections for each class
    CONF_THRESH = 0.8
    NMS_THRESH = 0.3
    for cls_ind, cls in enumerate(CLASSES[1:]):
        cls_ind += 1 # because we skipped background
        cls_boxes = boxes[:, 4*cls_ind:4*(cls_ind + 1)]
        cls_scores = scores[:, cls_ind]
        dets = np.hstack((cls_boxes,
                          cls_scores[:, np.newaxis])).astype(np.float32)
        keep = nms(dets, NMS_THRESH)
        dets = dets[keep, :]
        #vis_detections(im, cls, dets, thresh=CONF_THRESH)
        #vis_detections(fig, ax, cls, dets, thresh=CONF_THRESH)
        create_clippings(image_name, cls, dets, thresh=CONF_THRESH)

def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Tensorflow Faster R-CNN demo')
    parser.add_argument('--net', dest='demo_net', help='Network to use [vgg16 res101]',
                        choices=NETS.keys(), default='res101')
    parser.add_argument('--dataset', dest='dataset', help='Trained dataset [pascal_voc pascal_voc_0712]',
                        choices=DATASETS.keys(), default='pascal_voc_0712')
    parser.add_argument('--input_folder', help='Input folder to test PNG images', required=True)

    args = parser.parse_args()

    return args

if __name__ == '__main__':
    cfg.TEST.HAS_RPN = True  # Use RPN for proposals

    args = parse_args()

    # model path
    demonet = args.demo_net
    dataset = args.dataset
    tfmodel = os.path.join('output', demonet, DATASETS[dataset][0], 'default',
                              NETS[demonet][0])


    if not os.path.isfile(tfmodel + '.meta'):
        raise IOError(('{:s} not found.\nDid you download the proper networks from '
                       'our server and place them properly?').format(tfmodel + '.meta'))

    # set config
    tfconfig = tf.ConfigProto(allow_soft_placement=True)
    tfconfig.gpu_options.allow_growth=True

    # init directory
    if os.path.exists(DIR):
        shutil.rmtree(DIR)
    os.makedirs(DIR)

    # init session
    sess = tf.Session(config=tfconfig)
    # load network
    if demonet == 'vgg16':
        net = vgg16()
    elif demonet == 'res101':
        net = resnetv1(num_layers=101)
    else:
        raise NotImplementedError
   
    net.create_architecture("TEST", 7, tag='default',
                          anchor_scales=cfg.ANCHOR_SCALES,
                          anchor_ratios=cfg.ANCHOR_RATIOS)
    saver = tf.train.Saver()
    saver.restore(sess, tfmodel)

    print('Loaded network {:s}'.format(tfmodel))
    #for im_name in [png for png in os.listdir(args.input_folder)]:
    #for im_name in im_names:
    #for im_name in open("/data_raid5/weijun/android_data/ImageSets/val.txt").read().splitlines():
    #for im_name in open("data/VOCdevkit/android_data/ImageSets/val.txt").read().splitlines():
    #num_processed = 0
    start = time.time()
    files = glob.glob(args.input_folder+"/*.png")
    for im_name in tqdm(files, total=len(files)):
        #print(im_name)
        #print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
        #print('Demo for {}'.format(im_name))
        demo(sess, net, im_name)
        #num_processed += 1
        #if num_processed%5 is 0:
        #    print("%s of %s processed" % (num_processed, len(files)))
    
    print("Total processing time: {}".format(time.time()-start)) 
