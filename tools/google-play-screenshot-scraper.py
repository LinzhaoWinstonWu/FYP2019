import argparse
import glob
import io
import json
import os
import os.path as osp
import re
import shutil
import time
import sys
import requests

#from urllib.request import urlopen
from PIL import Image
from multiprocessing import Pool, Value
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("--input_dir", required=True, help="path to folder containing Play Store json files", default='data/play_store_json')
parser.add_argument("--output_dir", help="directory to store output screenshots", default='data')
a = parser.parse_args()

OUTPUT_DIR = "play_store_screenshots"

def scrape():
    start = time.time()
    num_processed = 0
    idx = 0
    files = glob.glob(a.input_dir + "/*.json")
    for infile in files:
        with open(infile) as f:
            datastore = json.load(f)
            urls = datastore['screenshot_url']
            for url in urls:
                try:
                    url = re.sub(r'^//', 'http://', url)
                    #f = urlopen(url)
                    #im = io.BytesIO(f.read())
                    r = requests.get(url)
                    im = io.BytesIO(r.content)
                    img = Image.open(im)
                    img.save(osp.join(OUTPUT_DIR, "{0:0>6}.png".format(idx)))
                except IOError as e:
                    raise e
                idx+=1
        num_processed += 1
        if num_processed%5 is 0:
            print("%s of %s processed" % (num_processed, len(files)))

    print("[+] Completed in {} seconds".format(time.time()-start))

def scrape_parallel(file):
    global count

    screenshots = {}
    meta_data = {}
    #pnglist = []
    duplicate = False

    with open(file) as f:
        datastore = json.load(f)
        urls = datastore['screenshot_url']
        basename = osp.basename(file)
        filename = osp.splitext(basename)[0]
        for url in urls:
            try:
                url = re.sub(r'^//', 'http://', url)
                #f = urlopen(url)
                #im = io.BytesIO(f.read())
                r = requests.get(url)
                im = io.BytesIO(r.content)
                img = Image.open(im)

                for k, v in enumerate(screenshots):
                    diff_score = compareHisto(img, v)
                    if diff_score < 0.17:
                        duplicate = True

                if not duplicate:
                    with count.get_lock():
                        count.value += 1
                        path = osp.join(OUTPUT_DIR, filename+"_{0:0>6}.png".format(count.value))
                        img.save(path)
                        screenshots[path] = 1
                        #pnglist.append(count.value)
                    

            except IOError as e:
                print("[-] IOError - {}".format(file))
                continue

            duplicate = False
    return
    #basename = osp.basename(file)
    #filename = osp.splitext(basename)[0]
    #pkg = filename.replace('_', '.')
    #meta_data[pkg] = pnglist
    #return meta_data

def compareHisto(first, sec):
    #imA = Image.open(first)
    imA = first
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

def init(c):
    global count
    count = c


if __name__ == "__main__":

    if a.output_dir is not None:
        OUTPUT_DIR = osp.join(a.output_dir, OUTPUT_DIR)

    if osp.exists(OUTPUT_DIR):
        shutil.rmtree(OUTPUT_DIR)
    os.makedirs(OUTPUT_DIR)
    
    
    files = glob.glob(a.input_dir + "/*.json")
    meta_dump = {}
    
    count = Value('i', 0)

    start = time.time()
    p = Pool(initializer=init, initargs=(count,))
    for mt in tqdm(p.imap(scrape_parallel, files), total=len(files)):
        #meta_dump = {**meta_dump, **mt}
        continue
    p.close()
    p.join()

    #with open(osp.join(OUTPUT_DIR, "meta_dump.txt"), 'a+') as f:
    #    json.dump(meta_dump, f, sort_keys=True)

    print("[+] Completed in {} seconds".format(time.time()-start))

