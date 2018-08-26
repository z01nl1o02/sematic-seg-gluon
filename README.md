# sematic-seg-gluon
sematic segmentation with gluon


# 数据   
以VOC2007为例，当前目录设置为本工程目录
1. 转换标签    
   把VOC2007的SegmentationClass目录里的png的绝对路径写到voc2007.segclass.txt里，一行一个路径，运行cvt_voc_label.py，生成voc2007/labels/目录，其中的png是转换后的标签图

2. 生成train/test列表   
   把get_data_list.py里的voc_root修改指向VOC2007根目录，运行get_data_list.py,生成train.txt和test.list
   其中一行以|分割，第一项是jpg图像，第二项是png标签图


# FCN

## 训练
1. 训练fcnx32    
   直接 ```python train.py```

2. 训练fcnx16   
修改train.py   
   ```net_type = "fcn16"```   
   ```net_pretrained = 'models/fcn32_00001.params'```   
其中net_pretrained指向训练好的fcn32模型

3. 训练fcn8    
      
## 实验   
