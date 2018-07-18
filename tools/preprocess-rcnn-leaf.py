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

from imgaug import augmenters as iaa
from multiprocessing import Pool, Value, Manager
from lxml import etree
from PIL import Image, ImageFile, ImageDraw, ImageFont
ImageFile.LOAD_TRUNCATED_IMAGES = True

parser = argparse.ArgumentParser()
parser.add_argument("--input_dir", required=True, help="path to folder containing app packages")
a = parser.parse_args()

outputDir = "android_data_ambig_aug2_pb"
dir_img = os.path.join(outputDir, "PNGImages")
dir_set = os.path.join(outputDir, "ImageSets")
dir_ann = os.path.join(outputDir, "Annotations")

#target_list = ["Button", "ImageButton", "CompoundButton", "ProgressBar", "SeekBar", "Chronometer", "CheckBox", "RadioButton", "Switch", "EditText", "ToggleButton", "RatingBar", "Spinner",] # "View"]
#target_list = ["TextView", "Button", "ImageButton", "SeekBar", "CheckBox", "RadioButton", "EditText",]
target_list = ['Checkbox', 'Button', 'Chronometer', 'RadioButton', 'RatingBar', 'SeekBar', 
                    #'EditText',
                    'Spinner', 'ToggleButton', 'ProgressBar', 'CompoundButton', 'Switch', 'ImageButton']

        
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


def remove_overlap(widgets, status_bar):
    global countValidFile, train_val
    w = []
    widgets.reverse()
    layout = np.zeros((801,1217), dtype=np.int)
    for idx, widget in enumerate(widgets):
        overlap = False
        starti = widget['coordinates']['from'][0] + 1
        endi = widget['coordinates']['to'][0] - 1
        startj = widget['coordinates']['from'][1] + 1
        endj = widget['coordinates']['to'][1] - 1

        for i in range(starti, endi + 1):
            for j in range(startj, endj + 1):
                if layout[i][j] == 1:
                    if widget['leaf']:
                        overlap = True
                        layout[i][j] = -1
                elif layout[i][j] == -1:
                    overlap = True
                elif layout[i][j] == 2:
                    if widget['leaf']:
                        overlap = True
                        layout[i][j] = -1
                else:
                    if widget['leaf']:
                        layout[i][j] = 1
                    else:
                        layout[i][j] = 2

        if overlap == True:
            continue
        else:
            if widget['leaf'] and widget['widget_class'] in target_list:
                if widget['widget_class'] == "View" and not(widget['clickable'] == "true" and widget['focusable'] == "true"):
                    continue

                w.append(widget)

    if w:
        with countValidFile.get_lock():
            countValidFile.value += 1
        
        try:
            im = Image.open(w[0]['src']+'.png')
            if status_bar:
                clip = im.crop((0, 33, 800, 1216))
            else:
                clip = im.crop((0, 0, 800, 1216))
        except OSError as err:
            print(w[0]['src'])
            print("[-] OSError - " + str(err))
            sys.stdout.flush()
            pass
        except IndexError as err:
            print(w[0]['src'])
            print("[-] IndexError - " + str(err))
            sys.stdout.flush()
            #print "[-] " + str(widget['coordinates'])
            #print "[-] " + str(widget['dimensions'])
            pass
        except IOError as err: #image file is truncated
            print(w[0]['src'])
            print("[-] IOError - " + str(err))
            sys.stdout.flush()
            pass

        clip.save(os.path.join(dir_img, "{0:0>6}.png".format(countValidFile.value)))
        if countValidFile.value % 30 == 1:
            val = train_val['val']
            val.append("{:06d}".format(countValidFile.value))
            train_val['val'] = val
        else:
            train = train_val['train']
            train.append("{:06d}".format(countValidFile.value))
            train_val['train'] = train

        trainval = train_val['trainval']
        trainval.append("{:06d}".format(countValidFile.value))
        train_val['trainval'] = trainval

            
        with open(os.path.join(dir_ann, "{0:0>6}.txt".format(countValidFile.value)), 'a+') as f:
            #tmp = {}
            #tmp['app'] = w
            json.dump(w, f, sort_keys=True, indent=3, separators=(',', ': '))

                #with open(os.path.join(dir_ann, "{0:0>6}.txt".format(countValidFile.value)), 'a+') as f:
                #    json.dump(widget, f, sort_keys=True, indent=3, separators=(',', ': '))


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
    #seq = iaa.Sequential([iaa.Fliplr(1.0)])
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

    '''
    seq = iaa.Sometimes(
        0.3,
        iaa.Fliplr(1.0),
        iaa.CropAndPad(percent=(0, 0.25))
    )

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
    pnglist = []
    hash_dict = {}

    for infile in glob.glob(input_folder + "/stoat_fsm_output/ui/*.xml"):
        widgets_xml = []
        status_bar = False
        name, ext = os.path.splitext(infile)
        pngfile = infile.replace('.xml', '.png')
        if os.path.exists(pngfile) and os.stat(pngfile).st_size > 0 and checkFileValidity(infile):
            try:
                Image.open(pngfile)
            except Exception as e:
                print(e)
                sys.stdout.flush()
                continue

            print(pngfile)
            # check for duplicate image
            dup = False
            if not pnglist:
                pnglist.append(pngfile)
            else:
                for png in pnglist:
                    diff_score = compareHisto(pngfile, png)
                    if diff_score < 0.051:
                        dup = True
                        break
                if dup:
                    continue
                else:
                    pnglist.append(pngfile)                

            ctx = etree.iterparse(infile, events=('start',), tag='node')
            for event, elem in ctx:
                # Check for android status bar
                if elem.attrib['bounds'] == "[0,33][800,1216]": 
                    status_bar = True

                widget_name = elem.attrib['class'].split('.')[-1]
                coordinates = re.findall(r"(?<=\[).*?(?=\])", elem.attrib["bounds"])
                if len(coordinates) != 2: 
                    continue

                #if not widget_name in target_list:
                #    continue

                #if not widget_name in layout and len(elem.getchildren()) != 0:
                #    continue

                if not widget_name in target_list or len(elem.getchildren()) != 0:
                    continue

                coor_from = tuple(map(int, coordinates[0].split(",")))
                coor_to = tuple(map(int, coordinates[1].split(",")))
                
                if not (coor_from[0] > 800 or coor_from[0] < 0 or coor_to[0] > 800 or coor_to[0] < 0 or coor_from[1] > 1216 or coor_from[1] < 0 or coor_to[1] > 1216 or coor_to[1] < 0):
                    meta_data = {}
                    meta_data['widget_class'] = widget_name

                    meta_data['coordinates'] = {'from': coor_from, 'to': coor_to}
                    meta_data['dimensions'] = getDimensions(coor_from, coor_to)
                    if meta_data['dimensions']['width'] == 0 or meta_data['dimensions']['height'] == 0:
                        continue

                    #if (meta_data['widget_class'] in layout and not(len(elem.getchildren()) == 2 and elem.getchildren()[0].attrib['class'].split('.')[-1] == "ImageView" and len(elem.getchildren()[0].getchildren()) == 0 and elem.getchildren()[1].attrib['class'].split('.')[-1] == "TextView" and len(elem.getchildren()[1].getchildren()) == 0)):
                    #    continue

                    #if meta_data['widget_class'] == "TextView" and not(not(elem.attrib['checkable'] == "true") and not(elem.attrib['checked'] == "true") and elem.attrib['clickable']  == "true" and elem.attrib['enabled']  == "true" and elem.attrib['focusable']  == "true" and not(elem.attrib['focused'] == "true") and not(elem.attrib['scrollable'] == "true") and elem.attrib['long-clickable'] == "true"):
                    #    continue

                    if (meta_data['widget_class'] == "TextView" and not(len(elem.attrib['text'].split(" ")) <= 3 and not(elem.attrib['checkable'] == "true") and not(elem.attrib['checked'] == "true") and elem.attrib['clickable']  == "true" and elem.attrib['enabled']  == "true" and elem.attrib['focusable']  == "true" and not(elem.attrib['focused'] == "true") and not(elem.attrib['scrollable'] == "true") and elem.attrib['long-clickable'] == "true")):
                        continue

                    # Classify ProgressBar(PB) as circular/horizontal based on aspect ratio, remaining regarded as invalid 
                    # Circular PB:         aspect_ratio == 1
                    # Horizontal PB:    aspect_ratio < 0.3
                    '''
                    if meta_data['widget_class'] == "ProgressBar":
                        ar = float(meta_data['dimensions']['height']) / float(meta_data['dimensions']['width'])
                        if ar == 1:
                            meta_data['widget_class'] = "CirProgressBar"
                        elif ar < 0.3:
                            meta_data['widget_class'] = "HrzProgressBar"
                        else:
                            widgets_xml = []
                            break
                    '''

                    meta_data['text'] = elem.attrib['text'] 
                    meta_data['clickable'] = elem.attrib['clickable']
                    meta_data['focusable'] = elem.attrib['focusable']
                    meta_data["content-desc"] = elem.attrib['content-desc']
                    meta_data['src'] = name

                    #if meta_data['widget_class'] == "TextView" or meta_data['widget_class'] in layout:
                    if meta_data['widget_class'] == "TextView":
                        meta_data['widget_class'] = "ImageButton"

                    widgets_xml.append(meta_data)

        else:
            continue

        if widgets_xml:        
            try:
                im = Image.open(pngfile)
                if status_bar:
                    clip = im.crop((0, 32, 800, 1216))
                    w = [rem(ann) for ann in widgets_xml if not (ann['coordinates']['from'][1] < 33 or ann['coordinates']['to'][1] < 33)]

                else:
                    clip = im.crop((0, 0, 800, 1216))
                    w = widgets_xml
            except OSError as err:
                print(pngfile)
                print("[-] OSError - " + str(err))
                sys.stdout.flush()
                continue
            except IndexError as err:
                print(pngfile)
                print("[-] IndexError - " + str(err))
                sys.stdout.flush()
                continue
            except IOError as err: #image file is truncated
                print(pngfile)
                print("[-] IOError - " + str(err))
                sys.stdout.flush()
                continue

            '''
            fnt = ImageFont.truetype("arial.ttf", 17)
            draw = ImageDraw.Draw(clip)
            for label in w:
                draw.rectangle((label['coordinates']['from'], label['coordinates']['to']), outline="black")
                draw.text(label['coordinates']['from'], label['widget_class'], font=fnt, fill="black")
            clip.save(os.path.join(dir_img, "{0:0>6}.png".format(count)))
            '''

            # Train/Test split for augmented dataset
            #aug = train_test_split(clip, w)
            #if aug:
            #    augment(clip, w)
            # Train/Test split for unaugmented dataset
            with countValidFile.get_lock():
                countValidFile.value += 1
                count = countValidFile.value
 
                if count % 10 == 1:    
                    val = train_val['val']
                    val.append("{:06d}".format(count))
                    train_val['val'] = val
                else:
                    train = train_val['train']
                    train.append("{:06d}".format(count))
                    train_val['train'] = train
                
                trainval = train_val['trainval']
                trainval.append("{:06d}".format(count))
                train_val['trainval'] = trainval
                
                with open(os.path.join(dir_ann, "{0:0>6}.txt".format(count)), 'a+') as f:
                    json.dump(w, f, sort_keys=True, indent=3, separators=(',', ': '))
 
                clip.save(os.path.join(dir_img, "{0:0>6}.png".format(count)))


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

    folders_list = glob.glob(a.input_dir + "/*20170510_cleaned_outputs*/**")

    '''
    fname = os.path.join("android_data", "Annotations", "000001.txt")
    with open(fname) as f:
        datastore = json.load(f)
        bb_from = datastore[0]['coordinates']['from']
        print bb_from, bb_from[0]
    '''
    #create dataset directory
    dir_list = [outputDir, dir_img, dir_set, dir_ann]
    if os.path.exists(outputDir):
        shutil.rmtree(outputDir)
    for d in dir_list:
        os.makedirs(d)

    num_processed = 0
    pool = Pool(processes=6, initializer=init, initargs=(countValidFile, train_val, stats))
    #pool = Pool(processes=6)
    for r in pool.imap(preprocess, folders_list):
        num_processed += 1
        if num_processed%5 is 0:
            print("%s of %s processed" % (num_processed, len(folders_list)))
            sys.stdout.flush()
    pool.close()
    pool.join()

    with open(os.path.join(dir_set, "train.txt"), 'a+') as f:
        for idx in train_val["train"]:
            f.write("%s\n" % idx)

    with open(os.path.join(dir_set, "val.txt"), 'a+') as f:
        for idx in train_val["val"]:
            f.write("%s\n" % idx)

    with open(os.path.join(dir_set, "trainval.txt"), 'a+') as f:
        for idx in train_val["trainval"]:
            f.write("%s\n" % idx)
    
    with open(os.path.join(dir_set, "stats.txt"), 'a+') as f:
        for t in target_list:
            f.write("{} - {}\n".format(t, stats[t]))
    
