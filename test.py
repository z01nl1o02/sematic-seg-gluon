import mxnet as mx
from mxnet import gluon,nd
import numpy as np
from symbol import fcn
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

path_model = 'models/fcnx8_epoch55.params'
ctx_list = [mx.gpu(0)]

show_result_image = False

if show_result_image:
    batch_size = 1
else:
    batch_size = 5


dataset_dir = os.path.join( os.environ['HOME'], 'data/voc/VOCaug/dataset/')
dataset = VocAugDataset(dataset_dir,'val','val')
data_iter = gluon.data.DataLoader(dataset,batch_size,last_batch="rollover")

net = fcn.FCNx8(class_num=dataset.class_num,ctx=ctx_list[0])

net.load_parameters(path_model)
logger.info("loading models {}".format(path_model))


img_size = dataset.crop_size


class MeanIou:
    def __init__(self, class_num):
        self.data = {}
        self.class_num = class_num
        for k in range(class_num):
            self.data[k] = [0, 0]

    def update(self, pred, label):
        if isinstance(pred, list):
            for p, l in zip(pred, label):
                self._update(p, l)
        else:
            self._update(pred, label)
        return

    def _update(self, pred, label):
        if isinstance(label, mx.nd.NDArray):
            label = label.asnumpy()
        if isinstance(pred, mx.nd.NDArray):
            pred = pred.asnumpy()
        pred = pred.argmax(axis=1)
        pred = np.squeeze(pred)
        label = np.squeeze(label)
        for key in range(self.class_num):
            p = (pred == key)
            l = (label == key)
            self.data[key][0] += (l + p).sum()
            self.data[key][1] += (l * p).sum()

        return

    def reset(self):
        self.data = {}
        for k in range(self.class_num):
            self.data[k] = [0, 0]
        return

    def get(self):
        iou = []
        for key in range(self.class_num):
            if self.data[key][0] > 0:
                iou.append(self.data[key][1] / float(self.data[key][0]))
            else:
                iou.append(0.0)
        iou = np.asarray(iou)
        return 'meanIoU {}'.format(iou.mean())


class PixelAcc:
    def __init__(self):
        self.total = 0
        self.hit = 0

    def update(self, pred, label):
        if isinstance(pred, list):
            for p, l in zip(pred, label):
                self._update(p, l)
        else:
            self._update(pred, label)
        return

    def _update(self, pred, label):
        if isinstance(label, mx.nd.NDArray):
            label = label.asnumpy()

        if isinstance(pred, mx.nd.NDArray):
            pred = pred.asnumpy()

        pred = pred.argmax(axis=1)

        hit = (pred == label).sum()
        total = (label >= 0).sum()

        self.total += total
        self.hit += hit
        return

    def reset(self):
        self.total, self.hit = 0, 0
        return

    def get(self):
        return 'pixel acc {}'.format(self.hit / float(self.total))


pixel_acc = PixelAcc()
mean_iou = MeanIou(dataset.class_num)


pixel_acc.reset()
mean_iou.reset()

seen_num = 0
for batch in data_iter:
    data, label = batch

    seen_num += data.shape[0]
    if seen_num > len(dataset):
        break

    if show_result_image:
        img = data[0].asnumpy()
        img = np.transpose(img,(1,2,0))
        img = np.uint8(img * 255)
        img = np.reshape( img, (img_size, img_size, 3) )

        label_img = label[0].asnumpy()
        label_img = np.reshape( np.uint8(label_img * 10), (img_size, img_size) )

    data = data.as_in_context(ctx_list[0])

    pred = net(data).as_in_context(mx.cpu())
    mx.nd.waitall()
    pixel_acc.update(pred,label)
    mean_iou.update(pred,label)

    logger.info("seen_num {} {} {}".format(seen_num,pixel_acc.get(), mean_iou.get() ))

    if show_result_image:
        pred = nd.softmax(pred,axis=1).asnumpy()
        pred_label = np.argmax(pred,axis=1)[0]
        pred_label = np.reshape( np.uint8(pred_label * 10), (img_size, img_size) )
        cv2.imshow("img", img)
        cv2.imshow("label", label_img)
        cv2.imshow("predict",pred_label)
        cv2.waitKey(-1)

