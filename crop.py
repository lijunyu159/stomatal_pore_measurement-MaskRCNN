"""
将coco.py训练得到的权重应用于test 然后将mrcnn生成的掩模裁剪成更小的二值图
"""
import os
import sys
import math
import numpy as np
import cv2
import samples.balloon.mrcnn.model as modellib
import MyTimer
import tuoyuannihe.ellipses as el
import time
# from samples.balloon import balloon
from samples.coco import coco
from samples.balloon.mrcnn import utils
from mrcnn import visualize
from mrcnn.visualize import display_images
from sklearn.model_selection import cross_val_score

# Root directory of the project
ROOT_DIR = os.path.abspath("../mask_rcnn-master")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library


# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

# Local path to trained weights file
# 选择使用的权重
# MODEL_PATH = os.path.join(ROOT_DIR, "resnet50.h5")
MODEL_PATH=(r"D:\Mask_RCNN\mask_rcnn\测试银杏和杨树的参数误差权重\mask_rcnn_coco_0144.h5")
if not os.path.exists(MODEL_PATH):
    print('no weights!')

config = coco.CocoConfig()
config.display()

# Create model object in inference mode.
model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)
model.load_weights(MODEL_PATH, by_name=True)

class_names = ['BG', 'black_poplar_pore']

dataset_test = coco.CocoDataset()
# 选择测试的数据集
# |-test
# |-|-test
# |-|-|-***.png
# |-|=annotations
# |-|-|-***.json
dataset_test.load_coco(r"D:\Mask_RCNN\mask_rcnn\测试银杏的参数数据集\test2017", "test2017")
dataset_test.prepare()
# image_ids 是记录test_datasets中共有多少张图片
image_ids=dataset_test.image_ids
IoU_all_photos = np.zeros((1, len(image_ids)))
all_statistic = np.zeros((2, len(image_ids)))
count_statistic = np.zeros((2, len(image_ids)))

# 计时
t1 = MyTimer.MyTimer()
t1.start()
# 测试集
for f in range(len(image_ids)):
    image_id = f

    info = dataset_test.image_info[image_id]
    print("image ID: {}.{} ({}) {}".format(info["source"], info["id"], image_id,
                                           dataset_test.image_reference(image_id)))
    # 原demo在使用图像之前已经对图像进行了裁剪 具体代码在model的1224行
    image, image_meta, gt_class_id, gt_bbox, gt_mask = modellib.load_image_gt(
        dataset_test, config, image_id, use_mini_mask=False)
    # image 是 3D图像 [image]是4-D列表 len([image]) = 1 [image]代表把整个图像作为一个batch处理
    results = model.detect([image], verbose=1)
    # 这里的model.detect
    # results是一个列表，里面有个字典元素（rois,scores,mask,class_ids）
    # 这里的rois不是偏移量，是框。
    # r 是把这个字典拿出来
    r = results[0]
    '''
    # 生成实例可视化
    visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'],
                                class_names, r['scores'])
    # gt和pred对比可视化
    visualize.display_differences(image, gt_bbox,  gt_class_id, gt_mask, r['rois'], r['class_ids'],r['scores'] ,r['masks'],
                                        class_names, iou_threshold=0.5 , score_threshold=0.5)
    '''
    det_class_ids = r['class_ids']
    det_count = len(det_class_ids)  # 去掉背景的class_ids
    det_masks = r['masks']
    det_boxes = r['rois']
    det_scores = r['scores']
    captions = ["{} {:.3f}".format(class_names[int(c)], s) if c > 0 else ""
                for c, s in zip(det_class_ids, det_scores)]
    count_statistic[0, f] = len(gt_bbox)

    '''
    # 生成的边界框可视化
    visualize.draw_boxes(
        image,
        refined_boxes=utils.denorm_boxes(detections[:, :4], image.shape[:2]),
        visibilities=[2] * len(detections),
        captions=captions, title="Detections",
        ax=get_ax())
    '''
    # 所有用到的值（或矩阵）初始化
    # 分析气孔参数--真值和预测值进行比较
    pore_match_count = 0
    pred_pore_count = 0
    pred_match = -1 * np.ones(det_count)
    gt_match = -1 * np.ones([gt_bbox.shape[0]])
    IoU_per_photo = np.zeros((1, len(gt_bbox)))
    overlaps = (utils.compute_overlaps_masks(det_masks, gt_mask)).T

    def pred_stomata_resize(stomatal_index):
        # 裁剪整个气孔，寻找整个气孔的四点坐标(y1,x1,y2,x2)
        coordinate = det_boxes[stomatal_index] + [-30, -30, 30, 30]
        # 如果气孔在图像的左边或右边
        if coordinate[0]<0:
            coordinate[0]=det_boxes[stomatal_index][0]
        if coordinate[1] < 0:
            coordinate[1] = det_boxes[stomatal_index][1]
        # if coordinate[3]>det_boxes[stomatal_index][-1]:
        #     coordinate[3]=det_boxes[stomatal_index][-1]
        # 被裁剪的图像
        # cropped_src = image[coordinate[0]:coordinate[2], coordinate[1]:coordinate[3]]
        cropped_mask = det_masks[:, :, stomatal_index][coordinate[0]:coordinate[2], coordinate[1]:coordinate[3]]
        cropped_mask = cropped_mask.astype(int)
        # while f==0 and s==27:
            # print(coordinate)
        for i in range(len(cropped_mask)):
            for j in range(len(cropped_mask[0])):

                if cropped_mask[i, j] == 1:
                    cropped_mask[i, j] = 255

        single_pore_file_folder = './ginkgo_new/'+str(f)
        if not os.path.exists(single_pore_file_folder):
            os.makedirs(single_pore_file_folder)
        cv2.imwrite(single_pore_file_folder +"/"+ str(s) + 'det_masks.png',
                    cropped_mask)

    def gt_stomata_resize(stomatal_index,flag):
        # TODO：将预测图像mask裁剪代码和标记图像mask合并
        # 裁剪整个气孔，寻找整个气孔的四点坐标(y1,x1,y2,x2)
        coordinate = gt_bbox[stomatal_index] + [-30, -30, 30, 30]
        # 如果气孔在图像的左边或右边
        if coordinate[0]<0:
            coordinate[0]=gt_bbox[stomatal_index][0]
            # TODO：如何判断边界点并删除 （用于方法比较）
        if coordinate[1] < 0:
            coordinate[1] = gt_bbox[stomatal_index][1]
        # if coordinate[3]>gt_bbox[stomatal_index][-1]:
        #     coordinate[3]=gt_bbox[stomatal_index][-1]
        # 如果flag=="check", 则裁剪原气孔的图像，用于查找哪些气孔没有被识别
        if flag=="check":
            cropped_src = image[coordinate[0]:coordinate[2], coordinate[1]:coordinate[3]]
            single_pore_file_folder = './unsegm_pore/'
            if not os.path.exists(single_pore_file_folder):
                os.makedirs(single_pore_file_folder)
            cv2.imwrite(single_pore_file_folder + str(stomatal_index) + 'img.png',
                        cropped_src)
        # 如果flag=="measure", 则裁剪气孔的gt_masks（与pred_mask匹配），用于做测量
        else:
            # cropped_src = image[coordinate[0]:coordinate[2], coordinate[1]:coordinate[3]]
            cropped_mask = gt_mask[:, :, stomatal_index][coordinate[0]:coordinate[2], coordinate[1]:coordinate[3]]
            cropped_mask = cropped_mask.astype(int)
            for i in range(len(cropped_mask)):
                for j in range(len(cropped_mask[0])):
                    if cropped_mask[i, j] == 1:
                        cropped_mask[i, j] = 255

            # single_pore_file_folder = './cropped_stomata_img/'
            single_pore_file_folder2="./cropped_stomata_mask/"+str(f)
            # if not os.path.exists(single_pore_file_folder):
            #     os.makedirs(single_pore_file_folder)
            if not os.path.exists(single_pore_file_folder2):
                os.makedirs(single_pore_file_folder2)
            cv2.imwrite(single_pore_file_folder2 +"/"+ str(stomatal_index) + 'gt_masks.png',
                        cropped_mask)
            # cv2.imwrite(single_pore_file_folder + str(f).zfill(3)+str(stomatal_index) + 'img.png',
            #             cropped_src)
    def find_match(gt_index):
        sorted_ixs = np.argsort(overlaps[gt_index])[::-1]
        low_score_idx = np.where(overlaps[gt_index, sorted_ixs] < 0.5)[0]
        if low_score_idx.size > 0:
            sorted_ixs = sorted_ixs[:low_score_idx[0]]
        # 3 寻找匹配
        for j in sorted_ixs:
            # 根据IoU最大 匹配真值框和预测框
            if pred_match[j] > -1:
                continue
            iou = overlaps[gt_index, j]
            IoU_per_photo[:, s] = iou
            if iou < 0.5:
                break
            # 预测框和真值框对应
            pred_match[j] = gt_index
            gt_match[gt_index] = j
            return j
    def el_fitting(mask):
        padded_mask = np.zeros(
            (mask.shape[0] + 2, mask.shape[1] + 2), dtype=np.uint8)
        padded_mask[1:-1, 1:-1] = mask
        contours = visualize.find_contours(padded_mask, 0.5)
        if len(contours)!=1:
            point_num = [0] * len(contours)
            for i in range(len(contours)): # i=0,1,2...
                point_num[i]=len(contours[i])
            # print(point_num.index(max(point_num[:])))
            contours=[contours[point_num.index(max(point_num[:]))]]
        # 将坐标格式改写成椭圆拟合程序需要的格式
        # 也就是[[横坐标]，[纵坐标]]
        for verts in contours:
            # Subtract the padding and flip (y, x) to (x, y)
            verts = np.fliplr(verts) - 1
            h = verts[:, 0]
            z = verts[:, 1]
            #  width height分别是半长轴和半短轴
            data = np.array([h, z])
            lsqe = el.LSqEllipse()
            lsqe.fit(data)
            center, width, height, phi = lsqe.parameters()
            a = max(width, height)
            b = min(width, height)
            area = math.pi * a * b
            eccentricity = pow((1 - (b / a) ** 2), 1 / 2)
            pore_opening_degree = b / a
            return a, b, area, eccentricity, pore_opening_degree

    # (一张图片的 )main function
    for s in range(len(gt_bbox)):
        # 寻找预测值
        pred_index = find_match(s)
        if pred_index is None:
            # 检查哪些图片中有气孔没有被检测到 写入.txt
            # print(info["path"])
            # unsegm = open('./unsegm.txt', 'a')
            # unsegm.write('\n' + info["path"])
            # 将原气孔图像裁剪出来
            # gt_stomata_resize(s,"check")
            continue

        # 将单个气孔裁剪出来 用于CV模型 做方法对比
        # TODO:需要修改裁剪气孔程序来适应标记气孔和预测气孔坐标
        # 剔除在边界的点

        gt_stomata_resize(s,"measure")
        pred_stomata_resize(pred_index)

        pore_match_count += 1
        # 把 iou 放入矩阵中
        IoU_all_photos[0, f] = IoU_per_photo.sum(axis=1) / pore_match_count
    # 精确率
    if det_count:
        pore_precision = pore_match_count / det_count
    else:
        pore_precision=0
    # 召回率
    pore_recall = pore_match_count / len(gt_bbox)
    all_statistic[0, f] = pore_precision
    all_statistic[1, f] = pore_recall


# 求所有图片IoU的均值
IoU_average = float(IoU_all_photos.sum(axis=1)) / len(image_ids)

print('准确率 = {:.3f}, 召回率 = {:.3f}'
      .format((float(all_statistic.sum(axis=1)[0])) / len(image_ids),
              (float(all_statistic.sum(axis=1)[1])) / len(image_ids)))

print('100张图片的平均IoU为{:.2f}'.format(IoU_average))
t1.stop()



