### 基于卷积神经网络的人脸关键点检测，使用 TensorFlow 搭建
- 使用 3 个卷积层， 每个卷积层后面都接一个池化层，最后再接 3 个全连接层
- 由于是回归，损失函数为 rmse，此外还计算了 $R^2$ 用来直观地评价模型的好坏
- 使用 TensorBoard 可视化训练过程
- 数据来自 [Kaggle](https://www.kaggle.com/c/facial-keypoints-detection/data)，后续会使用 [MUCT](http://www.milbo.org/muct/)

