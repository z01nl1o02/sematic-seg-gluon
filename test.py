import mxnet as mx
from mxnet import gluon,nd
from mxnet.gluon import nn
import numpy as np
from symbol import fcn
import logging
from mxnet.gluon import utils
import cv2
import os,sys
import logging
import datetime

nowTime=datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')

formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
handleFile = logging.FileHandler("log/test_%s.txt"%nowTime,mode="wb")
handleFile.setFormatter(formatter)

handleConsole = logging.StreamHandler()
handleConsole.setFormatter(formatter)

logger = logging.getLogger(__name__)
logger.setLevel(level=logging.DEBUG)
logger.addHandler(handleFile)
logger.addHandler(handleConsole)

path_model = 'models/fcn16_00009.params'
class_num = 21
crop_size = 512
ctx_list = [mx.gpu(0)]
path_test = 'test.txt'
path_color_label = 'C:/dataset/voc/VOCdevkit/VOC2007/SegmentationClass'

#net = fcn.FCNx32(class_num=class_num,ctx=ctx_list[0])
net = fcn.FCNx16(class_num=class_num,ctx=ctx_list[0],fcnx32_path=None)
#net.initialize(ctx = ctx_list[0] )


def pascal_label2rgb(): #RGB mode
  palette = { 0: (  0,   0,   0),
             1: (128,   0,   0),
             2: (  0, 128,   0),
             3: (128, 128,   0),
             4: (  0,   0, 128) ,
             5: (128,   0, 128) ,
             6: (  0, 128, 128) ,
             7: (128, 128, 128) ,
             8: ( 64,   0,   0)  ,
             9: (192,   0,   0)  ,
             10: ( 64, 128,   0),
             11: (192, 128,   0) ,
             12: ( 64,   0, 128),
             13: (192,   0, 128),
             14: ( 64, 128, 128),
             15: (192, 128, 128),
             16: (  0,  64,   0),
             17: (128,  64,   0),
             18: (  0, 192,   0),
             19: (128, 192,   0),
             20: (  0,  64, 128) }

  return palette


net.load_params(path_model)
logger.info("loading models {}".format(path_model))


def calc_iou(pred,label):
    if isinstance(label,mx.nd.NDArray):
        label = label.asnumpy()
    if isinstance(pred, mx.nd.NDArray):
        pred = pred.asnumpy()
    pred = pred.argmax(axis=1)
    ab_and = np.sum((pred == label) * (label > 0) )#ignore background
    ab_or = np.sum( (label > 0)  + (pred > 0) ) #- ab_and
    return np.float32(ab_and) / ab_or if ab_or > 0 else 0

iou_list = []
with open(path_test,'rb') as f:
    for line in f:
        path_img, path_mark = line.strip().split('|')
        img = cv2.imread(path_img, 1)
        label = cv2.imread(path_mark, 0)


        img_padding = np.zeros((crop_size,crop_size,3),dtype=np.uint8)
        label_padding  = np.zeros((crop_size,crop_size,1),dtype=np.uint8)
        
        width, height = np.minimum(img.shape[1], crop_size), np.minimum(img.shape[0], crop_size)

        dx, dy = (img_padding.shape[1] - width)//2, (img_padding.shape[0] - height)//2
        sx, sy = (img.shape[1] - width)//2, (img.shape[0] - height)//2
        img_padding[dy:dy+height,dx:dx+width,:] = img[sy:sy+height, sx:sx+width,:]
        label_padding[dy:dy+height,dx:dx+width,0] = label[sy:sy+height, sx:sx+width]
        
        img_padding = np.float32(img_padding)

        img_padding = np.transpose(img_padding,(2,0,1))
        input_data = mx.nd.expand_dims( mx.nd.array(img_padding), axis=0).as_in_context(ctx_list[0])
        pred_prob = net(input_data).asnumpy()
        iou_list.append( calc_iou(pred_prob, label_padding) )

        pred_label = np.argmax(pred_prob, axis=1)
        pred_label = np.squeeze(pred_label)

        pred_color = np.zeros((crop_size,crop_size,3),dtype=np.uint8)
        for y in range(crop_size):
            for x in range(crop_size):
                label_val = pred_label[y,x]
                pred_color[y,x,:] = pascal_label2rgb()[label_val]
        pred_color = cv2.cvtColor(pred_color,cv2.COLOR_RGB2BGR)
        path = os.path.join(path_color_label,os.path.split(path_img)[-1])
        path = os.path.splitext(path)[0] + '.png'
        ground_color = cv2.imread(path,1)[sy:sy+height, sx:sx+width,:]

        cv2.imshow("img",img)
        cv2.imshow("ground",ground_color)
        cv2.imshow("pred",pred_color)
        cv2.waitKey(-1)

print("total {} mean-IoU {}".format(len(iou_list), np.asarray(iou_list).mean()))







