# Wirifying App Screenshots
An approach to automatically collect different types of UI components from screenshots of existing mobile apps.

This work leverages a Tensorflow implementation of Faster RCNN, mainly based on the work of [endernewton](https://github.com/endernewton/tf-faster-rcnn).

## Prerequisites
* [Tensorflow r1.2](https://www.tensorflow.org/versions/r1.2/install/)
* [Python 3.6](https://www.python.org/downloads/)
```Shell
pip install Cython opencv-python easydict numpy scipy scikit-image six lxml Pillow imgaug tqdm
```

## Getting Started
1. Clone the repository
  ```Shell
  git clone https://github.com/bernardcwj/FYP_2017.git
  ```

2. Update your -arch in setup script to match your GPU
  ```Shell
  cd FYP_2017/tf-faster-rcnn/lib
  # Change the GPU architecture (-arch) if necessary
  vim setup.py
  ```

  | GPU model  | Architecture |
  | ------------- | ------------- |
  | TitanX (Maxwell/Pascal) | sm_52 |
  | GTX 960M | sm_50 |
  | GTX 1080 (Ti) | sm_61 |
  | Grid K520 (AWS g2.2xlarge) | sm_30 |
  | Tesla K80 (AWS p2.xlarge) | sm_37 |

  **Note**: You are welcome to contribute the settings on your end if you have made the code work properly on other GPUs. Also even if you are only using CPU tensorflow, GPU based code (for NMS) will be used by default, so please set **USE_GPU_NMS False** to get the correct output.


3. Build the Cython modules
  ```Shell
  make clean
  make
  cd ../..
  ```

## Test with pre-trained model
1. Download training and test data
  ```Shell
  cd data
  wget ...
  unzip androidAUG.zip
  cd ..
  ```

2. Create symlinks for the Android dataset
  ```Shell
  cd tf-faster-rcnn/data/VOCdevkit
  ln -s  ../../../data/android_data_aug android_data
  cd ../..
  ```

3. Download and extract pre-trained model
  Android dataset with image augmentation [here](https://drive.google.com/open?id=1D324PezWrsS1tpYLIreKQhSOVdHFbwr-)
  ```Shell
  unzip android_aug_res101_140k.zip
  sudo cp -r android_aug_res101_140k/VOCdevkit data/
  sudo cp -r android_aug_res101_140k/cache data/
  sudo cp -r android_aug_res101_140k/output .
  ```

4. Test with pre-trained model
  ```Shell
  ./experiments/scripts/train_faster_rcnn.sh [GPU_ID] [DATASET] [NET]
  # GPU_ID is the GPU you want to test on
  # NET is the network arch to use
  # DATASET is defined in train_faster_rcnn.sh
  ./experiments/scripts/test_faster_rcnn.sh 0 android_voc res101
  ```

## Demo on app introductory screenshots crawled from Google Play App Store
A collection of app meta-data files can be found in `data/play_store_json`. 

**Note**: For demonstration purposes, the data of only ten apps are made available 

1. Preprocessing
  Crawl images from Google Play App Store
  ```Shell
  # By default, crawled images are saved under data/play_store_screenshots
  cd ..
  python tools/google-play-screenshot-scraper.py 
  ```

2. Run demo on crawled images
  ```Shell
  # By default, demo outputs are saved under demo_output
  cd tf-faster-rcnn
  ./tools/demo_modified.py
  ```