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

from nets.vgg16 import vgg16
from nets.resnet_v1 import resnetv1
'''
CLASSES = ('__background__',
           'aeroplane', 'bicycle', 'bird', 'boat',
           'bottle', 'bus', 'car', 'cat', 'chair',
           'cow', 'diningtable', 'dog', 'horse',
           'motorbike', 'person', 'pottedplant',
           'sheep', 'sofa', 'train', 'tvmonitor')
'''
'''
CLASSES = ('__background__', 'button', 'imagebutton', 'compoundbutton', 'progressbar', 'seekbar', 'chronometer', 'checkbox', 'radiobutton', 'switch', 'edittext', 'togglebutton', 'ratingbar', 'spinner')
'''
CLASSES = ('__background__', 'button', 'imagebutton', 'seekbar', 'checkbox', 'radiobutton', 'edittext')
                    #'button', 'imagebutton', 'seekbar', 'checkbox', 'radiobutton', 'edittext', 
                    #'chronometer', 'compoundbutton', 'progressbar', 'ratingbar', 'spinner', 'switch', 'togglebutton')

#CLASSES = ('__background__', 'button', 'imagebutton', 'seekbar', 'checkbox', 'radiobutton', 'edittext', 'cirprogressbar', 'hrzprogressbar')
NETS = {'vgg16': ('vgg16_faster_rcnn_iter_70000.ckpt',),'res101': ('res101_faster_rcnn_iter_140000.ckpt',)}
DATASETS= {'pascal_voc': ('voc_2007_trainval',),'pascal_voc_0712': ('voc_2007_trainval+voc_2012_trainval',),'android_voc': ('android_train',)}

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

    #ax.set_title(('{} detections with '
    #              'p({} | box) >= {:.1f}').format(class_name, class_name,
    #                                              thresh),
    #              fontsize=14)
    #plt.axis('off')
    #plt.tight_layout()
    #plt.draw()
 

def demo(sess, net, image_name, folder):
    """Detect object classes in an image using pre-computed object proposals."""

    # Load the demo image
    #im_file = os.path.join(folder, image_name)
    #im_file = "/data_raid5/weijun/android_data/PNGImages/" + image_name
    im_file = "data/VOCdevkit/android_data/PNGImages/" + image_name
    im = cv2.imread(im_file)
 
    im = im[:, :, (2, 1, 0)]
    fig, ax = plt.subplots(figsize=(12, 12))
    ax.imshow(im, aspect='equal')

    # Detect all object classes and regress object bounds
    timer = Timer()
    timer.tic()
    scores, boxes = im_detect(sess, net, im)
    timer.toc()
    print('Detection took {:.3f}s for {:d} object proposals'.format(timer.total_time, boxes.shape[0]))

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
        vis_detections(fig, ax, cls, dets, thresh=CONF_THRESH)

def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Tensorflow Faster R-CNN demo')
    parser.add_argument('--net', dest='demo_net', help='Network to use [vgg16 res101]',
                        choices=NETS.keys(), default='res101')
    parser.add_argument('--dataset', dest='dataset', help='Trained dataset [android_voc pascal_voc pascal_voc_0712]',
                        choices=DATASETS.keys(), default='android_voc')
    
    parser.add_argument('--input_folder', help='Input folder to test PNG images', default='../data/play_store_screenshots')

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

    # init session
    sess = tf.Session(config=tfconfig)
    # load network
    if demonet == 'vgg16':
        net = vgg16()
    elif demonet == 'res101':
        net = resnetv1(num_layers=101)
    else:
        raise NotImplementedError
    '''
    net.create_architecture("TEST", 9,
                          tag='default', anchor_scales=[8, 16, 32])
    '''
    net.create_architecture("TEST", 7, tag='default',
                          anchor_scales=cfg.ANCHOR_SCALES,
                          anchor_ratios=cfg.ANCHOR_RATIOS)
    saver = tf.train.Saver()
    saver.restore(sess, tfmodel)

    print('Loaded network {:s}'.format(tfmodel))
    '''
    im_names = ['000456.jpg', '000542.jpg', '001150.jpg',
                '001763.jpg', '004545.jpg']
    '''
    outputDir = "demo_output"
    if os.path.exists(outputDir):
        shutil.rmtree(outputDir)
    os.makedirs(outputDir)
    #for im_name in [png for png in os.listdir(args.input_folder)]:
    #for im_name in im_names:
    #for im_name in open("/data_raid5/weijun/android_data/ImageSets/val.txt").read().splitlines():
    num_processed = 0
    start = time.time()
    #for im_name in open("data/VOCdevkit/android_data/ImageSets/val.txt").read().splitlines():
    for im_name in [png for png in os.listdir(args.input_folder)]:
        if num_processed == 3000:        
             break
        #im_name += ".png"
        print(im_name)
        print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
        print('Demo for data/demo/{}'.format(im_name))
        demo(sess, net, im_name, args.input_folder)
        plt.axis('off')
        plt.tight_layout()
        plt.draw()	
        plt.savefig("demo_output/" + im_name)
        num_processed += 1

    print(num_processed)    
    print("Total processing time: {}".format(time.time()-start))    
    plt.show()
