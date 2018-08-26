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
handleFile = logging.FileHandler("log_%s.txt"%nowTime,mode="wb")
handleFile.setFormatter(formatter)

handleConsole = logging.StreamHandler()
handleConsole.setFormatter(formatter)

logger = logging.getLogger(__name__)
logger.setLevel(level=logging.DEBUG)
logger.addHandler(handleFile)
logger.addHandler(handleConsole)

path_model = 'models/00099.params'
class_num = 21
ctx_list = [mx.gpu(0)]
path_test = 'test.txt'
path_color_label = 'C:/dataset/voc/VOCdevkit/VOC2007/SegmentationClass'

net = fcn.RESNET(class_num=class_num,ctx=ctx_list[0])
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
        H,W,C = img.shape

        img_padding = np.zeros((512,512,3),dtype=np.float32)
        img_padding[0:H,0:W,:] = np.float32(img)
        img_padding = np.transpose(img_padding, (2,0,1))

        label_padding  = np.zeros((512,512,1),dtype=np.uint8)
        label_padding[0:H,0:W,0] = label
        label_padding = np.transpose(label_padding,(2,0,1))

        input_data = mx.nd.expand_dims( mx.nd.array(img_padding), axis=0).as_in_context(ctx_list[0])
        pred_prob = net(input_data).asnumpy()
        iou_list.append( calc_iou(pred_prob, label_padding) )

        pred_label = np.argmax(pred_prob, axis=1)
        pred_label = np.squeeze(pred_label)

        #print pred_label.min(), pred_label.max(), pred_label.shape
        pred_color = np.zeros((H,W,C),dtype=np.uint8)
        for y in range(H):
            for x in range(W):
                label_val = pred_label[y,x]
                pred_color[y,x,:] = pascal_label2rgb()[label_val]
        pred_color = cv2.cvtColor(pred_color,cv2.COLOR_RGB2BGR)
        path = os.path.join(path_color_label,os.path.split(path_img)[-1])
        path = os.path.splitext(path)[0] + '.png'
        ground_color = cv2.imread(path,1)

        
        cv2.imshow("img",img)
        cv2.imshow("ground",ground_color)
        cv2.imshow("pred",pred_color)
        cv2.waitKey(-1)

print("total {} mean-IoU {}".format(len(iou_list), np.asarray(iou_list).mean()))







