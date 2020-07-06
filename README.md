# An automatic model to measure the parameters of living pores based on MRCNN
本项目是在 [https://github.com/matterport/Mask_RCNN](mrcnn官方实现版本(keras+tensorflow)) 的基础上，按照实际需求实现的。
 实现配置:Tensorflow 1.10.0;keras 2.0.8;python 3.5。
 实现的功能是通过椭圆拟合操作自动测量**活体气孔**的孔隙参数(长轴，短轴，面积，离心率，开度)，并与标记真值做比较，进行定量分析。
 标记工具是:[https://github.com/wkentaro/labelme](labelme) 。
 椭圆拟合程序是:[https://github.com/bdhammel/least-squares-ellipse-fitting/tree/v1.0](最小二乘法拟合椭圆) 。
 在coco.py中训练代码。在crop.py中将检测出的掩模裁剪为单个的二进制掩模。在image2data.py中做定量分析。ellipse_fitting.py可以将单个掩模图像拟合成椭圆后可视化。
