import mxnet as mx
from mxnet import gluon,nd
import numpy as np
import logging
from symbol import fcn
from mxnet.gluon import utils
import cv2
import os,sys
import logging
import datetime
import cv2
from vocaug import VocAugDataset
import random
from collections import defaultdict

nowTime=datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')

formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
if not os.path.exists('log'):
    os.makedirs('log')
handleFile = logging.FileHandler("log/log.%s.txt"%nowTime,mode="wb")
handleFile.setFormatter(formatter)

handleConsole = logging.StreamHandler()
handleConsole.setFormatter(formatter)

logger = logging.getLogger()
logger.setLevel(level=logging.INFO)
logger.handles = []
logger.addHandler(handleFile)
logger.addHandler(handleConsole)
#logger.propagate = False

ctx_list = [mx.gpu(0)]
batch_size = 5

model_file = 'models/fcnx8_epoch55.params' #'fcn/fcn32_00022.params'

#epoch-0 1e-4
#epoch-2 1e-5

base_lr = 1e-5
max_epoch = 100

outdir = os.path.join(os.getcwd(),'models')
if not os.path.exists(outdir):
    os.makedirs(outdir)

dataset_dir = os.path.join(os.environ['HOME'],'data/voc/VOCaug/dataset/')
train_dataset = VocAugDataset(dataset_dir,'train', 'train' )
test_dataset = VocAugDataset(dataset_dir, 'val', 'val' )

trainIter = gluon.data.DataLoader(train_dataset,batch_size=batch_size,shuffle=True,last_batch='rollover')
testIter = gluon.data.DataLoader(test_dataset,batch_size=batch_size,shuffle=False,last_batch='rollover')

train_size = len(train_dataset)
display_freq_train = np.maximum(int(train_size * 0.01 / batch_size),1)
max_iter = max_epoch * train_size // batch_size

net = fcn.FCNx8(class_num=train_dataset.class_num,ctx = ctx_list[0])

if model_file != "":
    net.load_parameters(model_file)
    logger.info("loading models {}".format(model_file))

trainer = gluon.Trainer(net.collect_params(),"adam",{"wd":0.0005})
loss_ce = gluon.loss.SoftmaxCrossEntropyLoss(axis=1,sparse_label=True,from_logits=False)
lr_scheduler = mx.lr_scheduler.PolyScheduler(max_update=max_iter, base_lr=base_lr,pwr=2)

class ModelLoss:
    def __init__(self):
        self.data = []
    def reset(self):
        self.data = []
    def update(self,losses):
        if isinstance(losses, mx.nd.NDArray):
            losses = losses.as_in_context(mx.cpu()).asnumpy()
        losses = losses.tolist()
        self.data.extend(losses)
        return
    def get(self):
        str = 'loss {}'.format(np.asarray(self.data).mean())
        return str

class MeanIou:
    def __init__(self,class_num):
        self.data = {}
        self.class_num = class_num
        for k in range(class_num):
            self.data[k] = [0,0]
    def update(self,pred,label):
        if isinstance(pred,list):
            for p,l in zip(pred,label):
                self._update(p,l)
        else:
            self._update(pred,label)
        return
    def _update(self,pred,label):
        if isinstance(label,mx.nd.NDArray):
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
            self.data[k] = [0,0]
        return
    def get(self):
        iou = []
        for key in range(self.class_num):
            if self.data[key][0] > 0:
                iou.append( self.data[key][1] / float(self.data[key][0]) )
            else:
                iou.append(0.0)
        iou = np.asarray(iou)
        return 'meanIoU {}'.format(iou.mean())
    
class PixelAcc:
    def __init__(self):
        self.total = 0
        self.hit = 0
    def update(self,pred,label):
        if isinstance(pred, list):
            for p, l in zip(pred, label):
                self._update(p,l)
        else:
            self._update(pred,label)
        return
    def _update(self,pred,label):
        if isinstance(label,mx.nd.NDArray):
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
mean_iou = MeanIou(train_dataset.class_num)
model_loss = ModelLoss()
iter_num = 0
logger.info("train set {} test set {} max epoch {}".format(len(train_dataset), len(test_dataset), max_epoch))
for epoch in range(max_epoch):
    pixel_acc.reset()
    mean_iou.reset()
    model_loss.reset()
    for batch in trainIter:
        iter_num += 1
        data,label = batch

        data_list = utils.split_and_load(data,ctx_list)
        label_list = utils.split_and_load(label,ctx_list)
        losses = []
        with mx.autograd.record():
            pred_list = [net(x) for x in data_list]
            losses = [loss_ce(pred,label) for label,pred in zip(label_list,pred_list)]
        for loss in losses:
            loss.backward()
        trainer.step(batch_size)
        nd.waitall()
        mean_iou.update(pred_list, label_list)
        pixel_acc.update(pred_list, label_list)
        model_loss.update(loss)
        if iter_num % display_freq_train == 0:
            logger.info("train iter {} {} {} {}".format(iter_num,  model_loss.get(),  pixel_acc.get(), mean_iou.get() ))
        trainer.set_learning_rate( lr_scheduler(iter_num) )

    if 1:
        pixel_acc.reset()
        mean_iou.reset()
        model_loss.reset()
        net.save_parameters(os.path.join(outdir,'%s_%.5d.params'%(net.desc,iter_num)))
        for batch in testIter:
            data,label = batch
            data_list = utils.split_and_load(data,ctx_list)
            label_list = utils.split_and_load(label,ctx_list)
            pred_list = [net(x) for x in data_list]
            losses = [loss_ce(pred,label) for label,pred in zip(label_list,pred_list)]
            nd.waitall()
            pixel_acc.update(pred,label)
            mean_iou.update(pred,label)
            model_loss.update(loss)
        logger.info("test iter {} lr {} loss {} {} {}".format(
            iter_num, trainer.learning_rate,model_loss.get(), pixel_acc.get(), mean_iou.get() ))








