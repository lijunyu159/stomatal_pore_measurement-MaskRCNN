"""
将图像按顺序重命名
生成名字从0001.png~
"""
import os
import shutil
from datetime import datetime

base_path="D:/Mask_RCNN/mask_rcnn/cropped_stomata_mask/"
def is_imag(filename):
    return any(map(filename.endswith, [".png"]))


def get_time(filename):
    timestamp = os.path.getmtime(base_path+filename)
    return datetime.fromtimestamp(timestamp)


filenames = os.listdir(base_path)
images = filter(is_imag, filenames)
filenames.sort(key=get_time)
last_modified = None
for filename in filenames:
    modified = get_time(filename)

    # Sorting the files with date, and set serial number if the date is same
    if last_modified and last_modified.date() == modified.date():
        num += 1
    else:
        num = 1

    # Name:{serial number}.png   >   if the date is all the same
    targetname = str(num).zfill(4)+".png"
    shutil.move(base_path+filename, base_path+targetname)

    last_modified = modified