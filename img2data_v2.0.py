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
    输出是.csv表格
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
    # TODO:尝试直接用一个文件夹里的连续mask生成误差
    import os
    det_base_path=r"D:\Mask_RCNN\mask_rcnn\ginkgo_new"
    gt_base_path=r"D:\Mask_RCNN\mask_rcnn\cropped_stomata_mask"
    # 计时开始
    mytime=MyTimer.MyTimer()
    mytime.start()

    # # image_num 是 f
    # image_num=len([lists for lists in os.listdir(path) if os.path.isdir(os.path.join(path, lists))])
    # pore_num 是 s，表示总共有多少个匹配气孔
    porenums = len([lists for lists in os.listdir(det_base_path) if os.path.isfile(os.path.join(det_base_path, lists))])
    for pore in range(1,porenums):
        patameters = getEllipsesParameters()
        # pore是单个气孔(第几个气孔)
        det_porename=str(pore).zfill(4)+".png"
        gt_porename=str(pore).zfill(4)+".png"
        # 监测点，跳过并打印出出错的气孔
        try:
            pred_a,pred_b,pred_area,pred_eccentricity,pred_pore_opening_degree=patameters.img2data(os.path.join(det_base_path,det_porename))
            gt_a, gt_b, gt_area, gt_eccentricity, gt_pore_opening_degree = patameters.img2data(os.path.join(gt_base_path,gt_porename))
        except:
            print(det_porename)
        # 检测是哪些点出了问题
        # if abs(gt_b-pred_b)>1:
        #     print(det_porename)
        # # 创建.csv图像，绘制散点图
        gt_parameter=open("./gt_parameter.csv",mode="a")
        pred_parameter=open("./pred_parameter.csv",mode="a")
        gt_parameter.write("{},{},{},{},{}\n".format(gt_area,gt_a,gt_b,gt_eccentricity,gt_pore_opening_degree))
        pred_parameter.write("{},{},{},{},{}\n".format(pred_area,pred_a, pred_b,pred_eccentricity,pred_pore_opening_degree))



    # 计时结束
    mytime.stop()