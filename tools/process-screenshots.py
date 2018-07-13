import argparse
import os
import glob
import json
import shutil

from tqdm import tqdm

#targets = ['Button', 'Chronometer', 'EditText', 'ProgressBar', \
#'RatingBar', 'Spinner', 'ToggleButton', 'CheckBox', 'CompoundButton', \
#'ImageButton', 'RadioButton', 'SeekBar', 'Switch', 'View']

DIR = '/data_raid5/weijun/UIscreenshots'
SUBPATHS = ['top_10000_google_play_20170510_cleaned', 'top_10000_google_play_20170510_cleaned_outputs_verbose_xml']

parser = argparse.ArgumentParser()
parser.add_argument("--input_dir", required=True, help="path to folder containing widget clippings")
a = parser.parse_args()

def process():
    png_files = {}

    with open(a.input_dir + '/meta_dump.txt') as fh:
        meta_dump = json.load(fh)

    for k in meta_dump:
        src = meta_dump[k]['src']
        if src not in png_files:
            png_files[src] = ""

    for f in tqdm(png_files, total=len(png_files)):
        pkg = f.split('/')
        dst_path = DIR+'/'+pkg[3]+'/'+pkg[4]
        if not os.path.exists(dst_path):
            os.makedirs(dst_path)
        try:
            shutil.copy2(f+'.png', dst_path)
        except IOError:
            continue

    
if __name__ == '__main__':

    if os.path.exists(DIR):
        shutil.rmtree(DIR)
    os.makedirs(DIR)

    for sub in SUBPATHS:
        fpath = os.path.join(DIR, sub)
        os.makedirs(fpath)

    process()
    