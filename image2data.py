from PIL import Image
import numpy as np
import tuoyuannihe.ellipses as el
import math
import MyTimer
from skimage.measure import find_contours


# 依次读取掩模图像
class getEllipsesParameters:
    """
    根据已经得到的单个气孔的掩模二值图批量得到拟合后的椭圆参数误差
    输入是掩模图像
    输出是error
    """
    def img2data(self,mask):
        # mask是掩膜图像
        mask=Image.open(mask)
        mask=Image.fromarray(np.uint8(mask))
        mask=np.array(mask)
        padded_mask = np.zeros((mask.shape[0] + 2, mask.shape[1] + 2), dtype=np.uint8)
        padded_mask[1:-1, 1:-1] = mask
        # 提取边界
        contours = find_contours(padded_mask, 0.5)
        # 如果一个气孔孔隙不止提出一个边界，取最类似椭圆的边界
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
            return a,b,area,eccentricity,pore_opening_degree

if __name__=="__main__":
    import os
    base_path=r"C:\Users\LJY\Desktop\第一篇小论文-第二阶段\cropped_stomata\cropped_det"
    # 计时开始
    mytime=MyTimer.MyTimer()
    mytime.start()

    # 先统计一下
    path = r"D:\Mask_RCNN\mask_rcnn\cropped_stomata\first_time"
    # image_num 是 f
    image_num=len([lists for lists in os.listdir(path) if os.path.isdir(os.path.join(path, lists))])
    # pore_num 是 s，表示总共有多少个匹配气孔
    pore_num=0
    for img in range(image_num):
        dirname = r"D:\Mask_RCNN\mask_rcnn\cropped_stomata\first_time\{}".format(img)
        filenum = len([lists for lists in os.listdir(dirname) if os.path.isfile(os.path.join(dirname, lists))])  # 文件个数
        filenum //= 2  # gt和pred成对出现
        pore_num+=filenum
    # -----------------------------------------------------
    #               初始化一些统计用的全局表格和变量
    # -----------------------------------------------------

    # 所有图片下误差统计(以图片数为单位)
    all_statistic=np.zeros((5,image_num))
    # 记录所有气孔开度（以图片数为单位）
    opening_degree_range_nums_stastic=np.zeros((4,image_num))
    # 统计误差与开度关系
    all_opening_degree_and_error_analysis=np.zeros((8,image_num))
    # 统计开度在某个范围内的图片数量
    above_40_present_photo=0
    in_30_40_present_photo=0
    in_20_30_present_photo=0
    in_10_20_present_photo = 0

    # 循环，依次输入掩模图像
    path = r"D:\Mask_RCNN\mask_rcnn\cropped_stomata\first_time"
    dirnums=len([lists for lists in os.listdir(path) if os.path.isdir(os.path.join(path, lists))])
    for dir in range(dirnums):
        # dir 是第几个文件夹
        dirname = r"D:\Mask_RCNN\mask_rcnn\cropped_stomata\first_time\{}".format(dir)
        # 打开文件夹
        filenum=len([lists for lists in os.listdir(dirname) if os.path.isfile(os.path.join(dirname, lists))]) # 文件个数
        porenums=filenum//2 # gt和pred成对出现 porenums表示每个图片里有几个气孔
        # 在**一张图片**中每一个孔隙的五项参数相对误差记录表格
        absolute_error_statistics = np.zeros((5, pore_num))
        # 某个开度范围内气孔范围统计
        opening_degree_above_40 = 0
        opening_degree_30_40 = 0
        opening_degree_20_30 = 0
        opening_degree_10_20 = 0
        # 在某个开度范围内误差的统计
        # 说明:opening_degree_above_40_error_statistic
        # 在一张图片中 如果这个气孔开度大于40%
        # ———————————————————————————————————————————————————————————————————————————————————————
        # |  第一个开度>40%的气孔的**长**轴误差         |   第二个开度>40%的气孔的长轴误差       |  ……
        # ____________________________________________________________________________________
        # |  第一个开度>40%的气孔的**短**轴误差         |   第二个开度>40%的气孔的短轴误差       |  ……
        # ———————————————————————————————————————————————————————————————————————————————————————
        opening_degree_above_40_error_statistic = np.zeros((2, pore_num))
        opening_degree_30_40_error_statistic = np.zeros((2, pore_num))
        opening_degree_20_30_error_statistic = np.zeros((2, pore_num))
        opening_degree_10_20_error_statistic = np.zeros((2, pore_num))
        for pore in range(porenums):
            # pore是单个气孔(第几个气孔)
            det_porename=os.path.join(dirname,"{}det_masks.png".format(pore))
            gt_porename=os.path.join(dirname,"{}gt_masks.png".format(pore))
            if os.path.exists(det_porename):
                # 拒绝不连续
                patameters = getEllipsesParameters()
                # 监测点，跳过并打印出出错的气孔
                try:
                    pred_a,pred_b,pred_area,pred_eccentricity,pred_pore_opening_degree=patameters.img2data(det_porename)
                    gt_a, gt_b, gt_area, gt_eccentricity, gt_pore_opening_degree = patameters.img2data(gt_porename)
                except:
                    print(det_porename)
                # 检测是哪些点出了问题
                # if abs(gt_b-pred_b)>1:
                #     print(det_porename)
                # # 创建.csv图像，绘制散点图
                # a_parameter=open("./a_parameter.csv",mode="a")
                # b_parameter=open("./b_parameter.csv",mode="a")
                # area_parameter = open("./area_parameter.csv", mode="a")
                # eccentricity_parameter = open("./eccentricity_parameter.csv", mode="a")
                # od_parameter = open("./od_parameter.csv", mode="a")
                # a_parameter.write("{},{}\n".format(gt_a,pred_a))
                # b_parameter.write("{},{}\n".format(gt_b, pred_b))
                # area_parameter.write("{},{}\n".format(gt_area, pred_area))
                # eccentricity_parameter.write("{},{}\n".format(gt_eccentricity, pred_eccentricity))
                # od_parameter.write("{},{}\n".format(gt_pore_opening_degree, pred_pore_opening_degree))
                # 测试
                # print(pred_major_axes,pred_minor_axes,pred_area,pred_eccentricity,pred_open_degree)
                # print(gt_major_axes, gt_minor_axes, gt_area,gt_eccentricity, gt_open_degree)
                # 定量测量 (来自:coco_test.py)
                # 一张图片中气孔开度统计
                if pred_pore_opening_degree * 100 > 40:
                    # opening_degree_above_40 是开度超过40%的气孔数 （+1）
                    opening_degree_above_40 += 1
                    major_axis_relative_error = 1 - min(gt_a, pred_a) / max(gt_a, pred_a)
                    minor_axis_relative_error = 1 - min(gt_b, pred_b) / max(gt_b, pred_b)
                    opening_degree_above_40_error_statistic[0, pore] = major_axis_relative_error
                    opening_degree_above_40_error_statistic[1, pore] = minor_axis_relative_error
                elif pred_pore_opening_degree * 100 > 30:
                    opening_degree_30_40 += 1
                    major_axis_relative_error = 1 - min(gt_a, pred_a) / max(gt_a, pred_a)
                    minor_axis_relative_error = 1 - min(gt_b, pred_b) / max(gt_b, pred_b)
                    opening_degree_30_40_error_statistic[0, pore] = major_axis_relative_error
                    opening_degree_30_40_error_statistic[1, pore] = minor_axis_relative_error
                elif pred_pore_opening_degree * 100 > 20:
                    opening_degree_20_30 += 1
                    major_axis_relative_error = 1 - min(gt_a, pred_a) / max(gt_a, pred_a)
                    minor_axis_relative_error = 1 - min(gt_b, pred_b) / max(gt_b, pred_b)
                    opening_degree_20_30_error_statistic[0, pore] = major_axis_relative_error
                    opening_degree_20_30_error_statistic[1, pore] = minor_axis_relative_error
                elif pred_pore_opening_degree * 100 > 10:
                    opening_degree_10_20 += 1
                    major_axis_relative_error = 1 - min(gt_a, pred_a) / max(gt_a, pred_a)
                    minor_axis_relative_error = 1 - min(gt_b, pred_b) / max(gt_b, pred_b)
                    opening_degree_10_20_error_statistic[0, pore] = major_axis_relative_error
                    opening_degree_10_20_error_statistic[1, pore] = minor_axis_relative_error
                # 每一个孔隙的误差计算
                major_axis_relative_error = 1 - min(gt_a, pred_a) / max(gt_a, pred_a)
                minor_axis_relative_error = 1 - min(gt_b, pred_b) / max(gt_b, pred_b)
                area_relative_error = 1 - min(gt_area, pred_area) / max(gt_area, pred_area)
                eccentricity_relative_error = 1 - min(gt_eccentricity, pred_eccentricity) / max(gt_eccentricity,
                                                                                                pred_eccentricity)
                pore_opening_degree_relative_error = 1 - min(gt_pore_opening_degree, pred_pore_opening_degree) \
                                                     / max(gt_pore_opening_degree, pred_pore_opening_degree)
                # 将误差放入一个矩阵中
                absolute_error_statistics[0, pore] = major_axis_relative_error
                absolute_error_statistics[1, pore] = minor_axis_relative_error
                absolute_error_statistics[2, pore] = area_relative_error
                absolute_error_statistics[3, pore] = eccentricity_relative_error
                absolute_error_statistics[4, pore] = pore_opening_degree_relative_error
        # ------------------------------------------------------------------------------------
        #                               一张气孔图片统计完成
        # ------------------------------------------------------------------------------------
        # 统计一张图片中的误差
        # absolute_error是一张图片中五项参数误差各自相加
        # 然后将各自的和除以这张图片中匹配成功的气孔的个数
        absolute_error = absolute_error_statistics.sum(axis=1)
        major_axis_relative_error_per_photo = float(absolute_error[0]) / porenums
        minor_axis_relative_error_per_photo = float(absolute_error[1]) / porenums
        area_relative_error_per_photo = float(absolute_error[2]) / porenums
        eccentricity_relative_error_per_photo = float(absolute_error[3]) / porenums
        pore_opening_degree_relative_error_per_photo = float(absolute_error[4]) / porenums
        # 气孔相对误差统计
        all_statistic[0, dir] = major_axis_relative_error_per_photo
        all_statistic[1, dir] = minor_axis_relative_error_per_photo
        all_statistic[2, dir] = area_relative_error_per_photo
        all_statistic[3, dir] = eccentricity_relative_error_per_photo
        all_statistic[4, dir] = pore_opening_degree_relative_error_per_photo
        # 每张图片中气孔开度数量统计
        # ———————————————————————————————————————————————————————————————————————————————————————
        # |  第一张图片开度>40%的气孔数量          |   第二张图片开度>40%的气孔数量        |  ……
        # ____________________________________________________________________________________
        # |  第一张图片开度属于[30,40]的气孔数量    |   第二张图片开度属于[30,40]的气孔数量  |  ……
        # ———————————————————————————————————————————————————————————————————————————————————————
        #       ……
        # ———————————————————————————————————————————————————————————————————————————————————————
        opening_degree_range_nums_stastic[0, dir] = int(opening_degree_above_40)
        opening_degree_range_nums_stastic[1, dir] = int(opening_degree_30_40)
        opening_degree_range_nums_stastic[2, dir] = int(opening_degree_20_30)
        opening_degree_range_nums_stastic[3, dir] = int(opening_degree_10_20)
        # 每张图片的开度与误差关系
        # 每张图片中，开度大于40%的气孔长轴误差之和
        if opening_degree_above_40:
        # 存在气孔开度大于40%的图片
            above_40_present_photo += 1
        # 说明:all_opening_degree_and_error_analysis[0,0]
        # (第一张图片中气孔开度>40%的长轴误差)=第一张图片气孔开度>40%的长轴误差和/第一张图片气孔开度大于40%的气孔数目
        # ———————————————————————————————————————————————————————————————————————————————————————
        # |  第一张图片气孔开度>40%的长轴误差           |   第二张图片气孔开度>40%的长轴误差         |  ……
        # ____________________________________________________________________________________
        # |  第一张图片气孔开度>40%的短轴误差           |   第二张图片气孔开度>40%的短轴误差         |  ……
        # ———————————————————————————————————————————————————————————————————————————————————————
        # |  第一张图片气孔开度属于[30,40]的长轴误差     |   第二张图片气孔开度属于[30,40]的长轴误差   |  ……
        # ———————————————————————————————————————————————————————————————————————————————————————
        # |             ……                        |              ……                      |  ……
        # ———————————————————————————————————————————————————————————————————————————————————————
            all_opening_degree_and_error_analysis[0, dir] = float(
                opening_degree_above_40_error_statistic.sum(axis=1)[0]) / opening_degree_above_40
            # 每张图片中，开度大于40%的气孔短轴误差之和
            all_opening_degree_and_error_analysis[1, dir] = float(
                opening_degree_above_40_error_statistic.sum(axis=1)[1]) / opening_degree_above_40
        # 每张图片中，开度大于30%小于等于40%的气孔长轴误差之和
        if opening_degree_30_40:
        #     continue
        # else:
            in_30_40_present_photo += 1
            all_opening_degree_and_error_analysis[2, dir] = float(
                opening_degree_30_40_error_statistic.sum(axis=1)[0]) / opening_degree_30_40
            # 每张图片中，开度大于30%小于等于40%的气孔短轴误差之和
            all_opening_degree_and_error_analysis[3, dir] = float(
                opening_degree_30_40_error_statistic.sum(axis=1)[1]) / opening_degree_30_40
        # 每张图片中，开度大于20%小于等于30%的气孔长轴误差之和
        if opening_degree_20_30:
        #     continue
        # else:
            in_20_30_present_photo += 1
            all_opening_degree_and_error_analysis[4, dir] = float(
                opening_degree_20_30_error_statistic.sum(axis=1)[0]) / opening_degree_20_30
            # 每张图片中，开度大于20%小于等于30%的气孔短轴误差之和
            all_opening_degree_and_error_analysis[5, dir] = float(
                opening_degree_20_30_error_statistic.sum(axis=1)[1]) / opening_degree_20_30
        # 每张图片中，开度大于10%小于等于20%的气孔长轴误差之和
        if opening_degree_10_20:
        #     continue
        # else:
            in_10_20_present_photo += 1
            all_opening_degree_and_error_analysis[6, dir] = float(
                opening_degree_10_20_error_statistic.sum(axis=1)[0]) / opening_degree_10_20
            # 每张图片中，开度大于10%小于等于20%的气孔短轴误差之和
            all_opening_degree_and_error_analysis[7, dir] = float(
                opening_degree_10_20_error_statistic.sum(axis=1)[1]) / opening_degree_10_20
    # --------------------------------------------------------
    #                       所有图片的统计
    # --------------------------------------------------------

    opening_degree_above_40_total_pore_count = opening_degree_range_nums_stastic.sum(axis=1)[0]  # 开度大于40%的气孔数量总数
    opening_degree_30_40_total_pore_count = opening_degree_range_nums_stastic.sum(axis=1)[1]
    opening_degree_20_30_total_pore_count = opening_degree_range_nums_stastic.sum(axis=1)[2]
    opening_degree_10_20_total_pore_count = opening_degree_range_nums_stastic.sum(axis=1)[3]
    # opening_degree_above_40_major_axis_error = opening_degree_30_40_major_axis_error = opening_degree_20_30_major_axis_error = opening_degree_10_20_major_axis_error = 0
    # opening_degree_above_40_minor_axis_error = opening_degree_30_40_minor_axis_error = opening_degree_20_30_minor_axis_error = opening_degree_10_20_minor_axis_error = 0
    if above_40_present_photo:
        opening_degree_above_40_major_axis_error = float(
            all_opening_degree_and_error_analysis.sum(axis=1)[0]) / above_40_present_photo
        opening_degree_above_40_minor_axis_error = float(
            all_opening_degree_and_error_analysis.sum(axis=1)[1]) / above_40_present_photo
    if in_30_40_present_photo:
        opening_degree_30_40_major_axis_error = float(
            all_opening_degree_and_error_analysis.sum(axis=1)[2]) / in_30_40_present_photo
        opening_degree_30_40_minor_axis_error = float(
            all_opening_degree_and_error_analysis.sum(axis=1)[3]) / in_30_40_present_photo
    if in_20_30_present_photo:
        opening_degree_20_30_major_axis_error = float(
            all_opening_degree_and_error_analysis.sum(axis=1)[4]) / in_20_30_present_photo
        opening_degree_20_30_minor_axis_error = float(
            all_opening_degree_and_error_analysis.sum(axis=1)[5]) / in_20_30_present_photo
    if in_10_20_present_photo:
        opening_degree_10_20_major_axis_error = float(
            all_opening_degree_and_error_analysis.sum(axis=1)[6]) / in_10_20_present_photo
        opening_degree_10_20_minor_axis_error = float(
            all_opening_degree_and_error_analysis.sum(axis=1)[7]) / in_10_20_present_photo
    print('总的来说，长轴相对误差 = {:.3f},短轴相对误差 = {:.3f},'
          '面积相对误差 = {:.3f},离心率相对误差 = {:.3f},孔隙开度相对误差 = {:.3f} '
          .format((float(all_statistic.sum(axis=1)[0])) / dirnums,
                  (float(all_statistic.sum(axis=1)[1])) / dirnums,
                  (float(all_statistic.sum(axis=1)[2])) / dirnums,
                  (float(all_statistic.sum(axis=1)[3])) / dirnums,
                  (float(all_statistic.sum(axis=1)[4])) / dirnums))
    print('开度大于40%的孔隙数量为={}, 开度大于30%且小于40%的孔隙数量为={}，开度大于20%且小于30%的孔隙数量为={}，'
          '开度大于10%且小于20%的孔隙数量为={}'.format(opening_degree_above_40_total_pore_count,
                                          opening_degree_30_40_total_pore_count, opening_degree_20_30_total_pore_count
                                          , opening_degree_10_20_total_pore_count))
    print('开度大于40%的孔隙的长轴误差是={},短轴误差是={}'
          '开度大于30%且小于40%的孔隙的长轴误差是={},短轴误差是={}'
          '开度大于20%且小于30%的孔隙的长轴误差是={},短轴误差是={}'
          '开度大于10%且小于20%的孔隙的长轴误差是={},短轴误差是={}'
          .format(opening_degree_above_40_major_axis_error, opening_degree_above_40_minor_axis_error,
                  opening_degree_30_40_major_axis_error, opening_degree_30_40_minor_axis_error,
                  opening_degree_20_30_major_axis_error, opening_degree_20_30_minor_axis_error,
                  opening_degree_10_20_major_axis_error, opening_degree_10_20_minor_axis_error))
    # 计时结束
    mytime.stop()