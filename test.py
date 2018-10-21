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
from vocaug import VocAugDataset

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

path_model = 'models/fcnx8_04542.params'
ctx_list = [mx.gpu(0)]



dataset_dir = os.path.join( os.environ['HOME'], 'data/voc/VOCaug/dataset/')
dataset = VocAugDataset(dataset_dir,'val','val')
data_iter = gluon.data.DataLoader(dataset,1,last_batch="rollover")

net = fcn.FCNx8(class_num=dataset.class_num,ctx=ctx_list[0])

net.load_parameters(path_model)
logger.info("loading models {}".format(path_model))


img_size = dataset.crop_size

for batch in data_iter:
    data, label = batch

    img = data[0].asnumpy()
    img = np.transpose(img,(1,2,0))
    img = np.uint8(img * 255)
    img = np.reshape( img, (img_size, img_size, 3) )

    label = label[0].asnumpy()
    label = np.reshape( np.uint8(label * 10), (img_size, img_size) )

    data = data.as_in_context(ctx_list[0])

    pred = net(data).as_in_context(mx.cpu())

    pred = nd.softmax(pred,axis=1).asnumpy()
    pred_label = np.argmax(pred,axis=1)[0]
    pred_label = np.reshape( np.uint8(pred_label * 10), (img_size, img_size) )

    cv2.imshow("img", img)
    cv2.imshow("label", label)
    cv2.imshow("predict",pred_label)
    cv2.waitKey(-1)

