# sematic-seg-gluon
hack gluoncv-fcn的代码，研究分析fcn使用


# 数据   
VOCAug  
[下载](https://gluon-cv.mxnet.io/build/examples_datasets/pascal_voc.html#sphx-glr-build-examples-datasets-pascal-voc-py)

数据准备和gluoncv一致，如果遇到问题可以[参考](https://blog.csdn.net/z0n1l2/article/details/83053429)

# 训练

python train.py


# 测试
python test.py


# 实验结果
[模型](https://pan.baidu.com/s/1JdS2WEi5RX4xSG_Lx2YsTA)这个在val.txt上meanIoU = 53%

左侧是原图，中间是groundtruth， 右侧是预测结果   
![pic1](https://github.com/z01nl1o02/sematic-seg-gluon/blob/master/images/72.jpg)    
![pic1](https://github.com/z01nl1o02/sematic-seg-gluon/blob/master/images/111.jpg)     
![pic1](https://github.com/z01nl1o02/sematic-seg-gluon/blob/master/images/116.jpg)    
![pic1](https://github.com/z01nl1o02/sematic-seg-gluon/blob/master/images/105.jpg)    



