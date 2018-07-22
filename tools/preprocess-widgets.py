import argparse
import os
import glob
import shutil
import json
import re
import hashlib
import numpy as np
import sys
import imgaug as ia
import random

from imgaug import augmenters as iaa
from multiprocessing import Pool, Value, Manager
from lxml import etree
from PIL import Image, ImageFile, ImageDraw, ImageFont
ImageFile.LOAD_TRUNCATED_IMAGES = True

parser = argparse.ArgumentParser()
parser.add_argument("--input_dir", required=False, help="path to folder containing app packages")
a = parser.parse_args()

inputDir = "android_data_ambig_aug2_pb"
dir_img = os.path.join(inputDir, "PNGImages")
dir_set = os.path.join(inputDir, "ImageSets")
dir_ann = os.path.join(inputDir, "Annotations")

target_list = ["Button", "ImageButton", "CompoundButton", "ProgressBar", "SeekBar", "Chronometer", "CheckBox", "RadioButton", "Switch", "EditText", "ToggleButton", "RatingBar", "Spinner",] # "View"]

targets = {}
for t in target_list:
    targets[t] = 0

rare_targets = {}
rare_targets["Chronometer"] = []
rare_targets["CompoundButton"] = []
rare_targets["Spinner"] = []

redundant_targets = {}
redundant_targets["Button"] = []
redundant_targets["ImageButton"] = []


def checkFileValidity(inputFile):
    '''
    Check the validity of the XML file and ignore it if possible
    Due to the unknown reasons, the content in some XML file is repetative or   
    '''
    homeScreen_list = ["Make yourself at home", "You can put your favorite apps here.", "To see all your apps, touch the circle."]
    unlockHomeScreen_list = ["Camera", "[16,600][144,728]", "Phone", "[150,1114][225,1189]", "People", "[256,1114][331,1189]", "Messaging", "[468,1114][543,1189]", "Browser", "[574,1114][649,1189]"]
    browser = ["com.android.browser:id/all_btn", "[735,108][800,172]", "com.android.browser:id/taburlbar", "com.android.browser:id/urlbar_focused"]
    with open(inputFile) as f:
        content = f.read()
        #it is the layout code for the whole window and no rotation
        if 'bounds="[0,0][800,1216]"' in content and '<hierarchy rotation="1">' not in content:
            if not all(keyword in content for keyword in browser) and not all(keyword in content for keyword in homeScreen_list) and not all(keyword in content for keyword in unlockHomeScreen_list):
                #it should not be the homepage of the phone
                bounds_list = re.findall(r'bounds="(.+?)"', content)
                if len(bounds_list) < 2:
                    return False
                #if float(len(bounds_list)) / len(set(bounds_list)) < 1.2:   #so far, we do not check this option
                    #print len(text_list), len(set(text_list)), inputFile.split("\\")[-1]
                return True
            
    return False        


def getDimensions(coor_from, coor_to):
    dim = {}
    dim['width'] = coor_to[0] - coor_from[0]
    dim['height'] = coor_to[1] - coor_from[1]
    return dim


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

    for i in range(len(hA)):
        #print(sum_hA)
        sum_hA += hA[i]
        sum_hB += hB[i]
        diff += abs(hA[i] - hB[i])

    return diff/(2*max(sum_hA, sum_hB))

def rem(ann):
    ann['coordinates']['from'] = list(ann['coordinates']['from'])
    ann['coordinates']['to'] = list(ann['coordinates']['to'])
    ann['coordinates']['from'][1] = ann['coordinates']['from'][1] - 33
    ann['coordinates']['to'][1] = ann['coordinates']['to'][1] - 33
    ann['coordinates']['from'] = tuple(ann['coordinates']['from'])
    ann['coordinates']['to'] = tuple(ann['coordinates']['to'])
    return ann

def augment(img, anns):
    global stats
    width, height = img.size
    img = np.array(img, dtype=np.uint8)
    valid = []

    kps = []
    for a in anns:
        x1 = a['coordinates']['from'][0] 
        y1 = a['coordinates']['from'][1] 
        x2 = a['coordinates']['to'][0] 
        y2 = a['coordinates']['to'][1] 
        kps.extend([ia.Keypoint(x=x1, y=y1), ia.Keypoint(x=x2, y=y2),])
        stats[a['widget_class']] += 1

    keypoints = ia.KeypointsOnImage(kps, shape=img.shape)

    '''
    seq = iaa.SomeOf((1, None), [
        # iaa.Fliplr(1.0),
        iaa.AverageBlur(k=2),
        iaa.Sometimes(
            0.5,
            iaa.CropAndPad(percent=(0, 0.25),
                           pad_mode=["constant"],
                           pad_cval=(0, 128)),
            iaa.CropAndPad(
                px=((0, 100), 100, (0, 100), 100),
                pad_cval=(0, 128)
            )
        )

    ])
    
    seq = iaa.Sometimes(
        0.1,
        iaa.Fliplr(1.0),
        iaa.CropAndPad(percent=(0, 0.25))
    )

    '''

    scale = random.uniform(0.6, 0.9)

    tran_num = (1 - scale) / 2
    trans_per = random.uniform(-tran_num, tran_num)
    color = random.randrange(200,250)

    seq = iaa.Sequential([
        # iaa.Affine(scale=(0.6, 1), cval=(128, 255)),
        iaa.Affine(scale=scale, cval=color),
        # iaa.Affine(translate_percent={"x": 0.2, "y": (0, 0.4)}, cval=(128, 255))
        iaa.Affine(translate_percent={"y": trans_per}, cval=color)
                          ])

    seq_det = seq.to_deterministic()

    #augment keypoints and images
    # im = Image.fromarray(img)
    # im.show()
    img_aug = seq_det.augment_images([img])[0]
    keypoints_aug = seq_det.augment_keypoints([keypoints])[0]

    # im = Image.fromarray(img_aug)
    # im.show()

    aug_anns = []
    #for i, value in range(len(anns)):
    for i, value in enumerate(anns):
        '''
        if keypoints_aug.keypoints[i*2].x == -1:
            keypoints_aug.keypoints[i*2].x = 0
        if keypoints_aug.keypoints[i*2+1].x == -1:
            keypoints_aug.keypoints[i*2+1].x = 0
        '''
        if keypoints_aug.keypoints[i*2].x > keypoints_aug.keypoints[i*2+1].x:
            temp = keypoints_aug.keypoints[i*2].x
            keypoints_aug.keypoints[i*2].x = keypoints_aug.keypoints[i*2+1].x
            keypoints_aug.keypoints[i*2+1].x = temp

        if keypoints_aug.keypoints[i*2].x < 0:
            keypoints_aug.keypoints[i*2].x = 0
        if keypoints_aug.keypoints[i*2].x > width:
            keypoints_aug.keypoints[i*2].x = width
        if keypoints_aug.keypoints[i*2+1].x < 0:
            keypoints_aug.keypoints[i*2+1].x = 0
        if keypoints_aug.keypoints[i*2+1].x > width:
            keypoints_aug.keypoints[i*2+1].x = width
        if keypoints_aug.keypoints[i*2].y < 0:
            keypoints_aug.keypoints[i*2].y = 0
        if keypoints_aug.keypoints[i*2].y > height:
            keypoints_aug.keypoints[i*2].y = height
        if keypoints_aug.keypoints[i*2+1].y < 0:
            keypoints_aug.keypoints[i*2+1].y = 0
        if keypoints_aug.keypoints[i*2+1].y > height:
            keypoints_aug.keypoints[i*2+1].y = height

        anns[i]['dimensions'] = getDimensions((keypoints_aug.keypoints[i*2].x, keypoints_aug.keypoints[i*2].y),
                                              (keypoints_aug.keypoints[i*2+1].x, keypoints_aug.keypoints[i*2+1].y))
        if anns[i]['dimensions']['width'] == 0 or anns[i]['dimensions']['height'] == 0:
            continue
        anns[i]['coordinates']['from'] = (keypoints_aug.keypoints[i*2].x, keypoints_aug.keypoints[i*2].y)
        anns[i]['coordinates']['to'] = (keypoints_aug.keypoints[i*2+1].x, keypoints_aug.keypoints[i*2+1].y)
        valid.append(i)

    aug_anns = []
    for idx in valid:
        aug_anns.append(anns[idx])

    im = Image.fromarray(img_aug)
    # draw = ImageDraw.Draw(im)
    '''
    for a in aug_anns:
        draw.rectangle((a['coordinates']['from'], a['coordinates']['to']), outline="red")
    '''
    '''
    for i in range(0,len(keypoints.keypoints),2):
        before = keypoints.keypoints[i]
        after = keypoints_aug.keypoints[i]
        print "Keypoint %d: (%.8f, %.8f) -> (%.8f, %.8f)" % (i, before.x, before.y, after.x, after.y)
        draw.rectangle((keypoints_aug.keypoints[i].x, keypoints_aug.keypoints[i].y, keypoints_aug.keypoints[i+1].x, keypoints_aug.keypoints[i+1].y), outline="red")
    im.show()
    '''

    if aug_anns:
        train_test_split(im, aug_anns, True)
    

def train_test_split(clip, anns, aug=False):
    global countValidFile, train_val

    with countValidFile.get_lock():
        countValidFile.value += 1
        count = countValidFile.value

        if aug:
            train = train_val['train']
            train.append("{:06d}".format(count))
            train_val['train'] = train
        else:
            if count % 15 == 1:
                val = train_val['val']
                val.append("{:06d}".format(count))
                train_val['val'] = val
            else:
                aug = True
                train = train_val['train']
                train.append("{:06d}".format(count))
                train_val['train'] = train

        trainval = train_val['trainval']
        trainval.append("{:06d}".format(count))
        train_val['trainval'] = trainval

        with open(os.path.join(dir_ann, "{0:0>6}.txt".format(count)), 'a+') as f:
            json.dump(anns, f, sort_keys=True, indent=3, separators=(',', ': '))

        '''
        draw = ImageDraw.Draw(clip)
        for a in anns:
            draw.rectangle((a['coordinates']['from'], a['coordinates']['to']), outline="red")
        '''

        clip.save(os.path.join(dir_img, "{0:0>6}.png".format(count)))

        return aug

def preprocess(input_folder):
    global countValidFile, train_val

    with open(os.path.join(dir_set, "train.txt"), 'r') as f:
        train = f.readlines()

    with open(os.path.join(dir_set, "val.txt"), 'r') as f:
        val = f.readlines()

    with open(os.path.join(dir_set, "trainval.txt"), 'r') as f:
        train_val = f.readlines()

    #for infile in glob.glob(input_folder + "/*.txt"):
    #    with open(infile) as f:
    #        datastore = json.load(f)
    #        for data in datastore:
    #            if data["widget_class"] == "Button":
    #                redundant_targets["Button"].append(infile)
    #            elif data["widget_class"] == "ImageButton":
    #                redundant_targets["ImageButton"].append(infile)
    #            elif data["widget_class"] == "Chronometer":
    #                rare_targets["Chronometer"].append(infile)
    #            elif data["widget_class"] == "CompoundButton":
    #                rare_targets["CompoundButton"].append(infile)
    #            elif data["widget_class"] == "Spinner":
    #                rare_targets["Spinner"].append(infile)

    #print("======= As a whole =============")
    #print("The number of Button", len(redundant_targets["Button"]))
    #print("The number of ImageButton", len(redundant_targets["ImageButton"]))
    #print("The number of Chronometer", len(rare_targets["Chronometer"]))
    #print("The number of CompoundButton", len(rare_targets["CompoundButton"]))
    #print("The number of Spinner", len(rare_targets["Spinner"]))

    for fn in train_val:
        file_name = os.path.join(input_folder, "{0:0>6}.txt".format(int(fn)))
        with open(file_name, 'r') as f:
            datastore = json.load(f)
            for data in datastore:
                if data["widget_class"] in target_list:
                    targets[data["widget_class"]] += 1
                else:
                    raise RuntimeError('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!111')
                    # continue

    print("======= For whole dataset ======")
    print("length of whole dataset :", len(train_val))
    n_total = 0
    print(targets)
    for i in targets.values():
        n_total += i
    for i in targets.keys():
        targets[i] = 100. * targets[i] / n_total
        print("%14s : %5.2f%%"% (i, targets[i]))

    n_total = 0
    for t in target_list:
        targets[t] = 0

    for fn in train:
        file_name = os.path.join(input_folder, "{0:0>6}.txt".format(int(fn)))
        with open(file_name, 'r') as f:
            datastore = json.load(f)
            for data in datastore:
                if data["widget_class"] in target_list:
                    targets[data["widget_class"]] += 1
                else:
                    raise RuntimeError('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!111')
                    # continue

    print("======= For train dataset ======")
    print("length of train dataset :", len(train))
    print(targets)
    for i in targets.values():
        n_total += i
    for i in targets.keys():
        targets[i] = 100. * targets[i] / n_total
        print("%14s : %5.2f%%"% (i, targets[i]))

    n_total = 0
    for t in target_list:
        targets[t] = 0

    for fn in val:
        file_name = os.path.join(input_folder, "{0:0>6}.txt".format(int(fn)))
        with open(file_name, 'r') as f:
            datastore = json.load(f)
            for data in datastore:
                if data["widget_class"] in target_list:
                    targets[data["widget_class"]] += 1
                else:
                    raise RuntimeError('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!111')
                    # continue

    print("======= For val dataset ======")
    print("length of validation dataset :", len(val))
    print(targets)
    for i in targets.values():
        n_total += i
    for i in targets.keys():
        targets[i] = 100. * targets[i] / n_total
        print("%14s : %5.2f%%"% (i, targets[i]))

#def print_statistics(data, string, folder)

def init(c, t, s):
    global countValidFile, train_val, stats
    countValidFile = c
    train_val = t
    train_val['train'] = []
    train_val['val'] = []
    train_val['trainval'] = []
    stats = s
    for t in target_list:
        stats[t] = 0

if __name__ == '__main__':
    countValidFile = Value('i', 0)
    train_val = Manager().dict()
    stats = Manager().dict()

    # folders_list = glob.glob(a.input_dir + "/*20170510_cleaned_outputs*/**")

    fname = os.path.join(inputDir, "Annotations")
    # with open(fname) as f:
    #     datastore = json.load(f)
    #     bb_from = datastore[0]['coordinates']['from']
    #     print(bb_from, bb_from[0])

    # num_processed = 0
    # pool = Pool(processes=1, initializer=init, initargs=(countValidFile, train_val, stats))
    #pool = Pool(processes=6)

    preprocess(fname)


