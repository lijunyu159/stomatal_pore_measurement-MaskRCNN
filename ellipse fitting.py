import numpy as np
import matplotlib.pyplot as plt

from tuoyuannihe import ellipses as el
from matplotlib.patches import Ellipse
from PIL import Image
from mrcnn import visualize
# 读取掩模图像
class getEllipsesParameters:
    """
    根据已经得到的单个气孔的掩模二值图画出拟合的椭圆轮廓
    对单个气孔掩膜图像进行拟合椭圆后的可视化（排查错误原因）
    """
    def img2data(self,mask):
        # mask是掩膜图像
        mask=Image.open(mask)
        mask=Image.fromarray(np.uint8(mask))
        mask=np.array(mask)
        padded_mask = np.zeros((mask.shape[0] + 2, mask.shape[1] + 2), dtype=np.uint8)
        padded_mask[1:-1, 1:-1] = mask
        # 提取边界
        contours = visualize.find_contours(padded_mask, 0.5)
        # 可能提取到好几个边界,取点最多的边界
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

            plt.close('all')
            fig = plt.figure(figsize=(6,6))
            ax = fig.add_subplot(111)
            ax.axis('equal')
            ax.plot(data[0], data[1], 'ro', label='test data', zorder=1)

            ellipse = Ellipse(xy=center, width=2*width, height=2*height, angle=np.rad2deg(phi),
                           edgecolor='b', fc='None', lw=2, label='Fit', zorder = 2)
            ax.add_patch(ellipse)

            plt.legend()
            plt.show()
if __name__=="__main__":
    parameters=getEllipsesParameters()
    para=parameters.img2data(r"C:\Users\LJY\Desktop\第一篇小论文-第二阶段\cropped_stomata\cropped_det\0209.png")