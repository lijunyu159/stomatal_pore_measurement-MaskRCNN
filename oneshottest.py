import os
import sys
import math
import numpy as np
import cv2
import samples.balloon.mrcnn.model as modellib
import skimage.io
from samples.balloon import balloon
from samples.balloon.mrcnn import utils
from mrcnn import visualize
from mrcnn.visualize import display_images
import random

# Root directory of the project
ROOT_DIR = os.path.abspath("./")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library


# Directory to save logs and trained model
# MODEL_DIR = os.path.join(ROOT_DIR, "logs")

# Local path to trained weights file
MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco_0160.h5")
if not os.path.exists(MODEL_PATH):
    print('no weights!')
# IMAGE_DIR = os.path.join(ROOT_DIR, "/samples/balloon/balloon/test1")

config = balloon.BalloonConfig()
config.display()

# Create model object in inference mode.
model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_PATH, config=config)
model.load_weights(MODEL_PATH, by_name=True)

class_names = ['BG', 'stomatal']


# 原demo在使用图像之前已经对图像进行了裁剪 具体代码在model的1224行
# file_names = next(os.walk(IMAGE_DIR))[2]
image = skimage.io.imread(r"D:\poredataset\0024.jpg")
    # image 是 3D图像 [image]是4-D列表 len([image]) = 1 [image]代表把整个图像作为一个batch处理
results = model.detect([image], verbose=1)
r = results[0]
    # 生成实例可视化 r["rois"]是到图像上的位置坐标
# visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'],
#                                 class_names, r['scores'])




