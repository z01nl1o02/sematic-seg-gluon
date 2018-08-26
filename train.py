import mxnet as mx
from mxnet import gluon,nd
from mxnet.gluon import nn
import numpy as np
import logging
from symbol import fcn
from mxnet.gluon import utils
import cv2
import os,sys
import logging
import datetime
import cv2

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

path_train_data = os.path.join( os.getcwd(), 'train.txt'  )

path_test_data = os.path.join( os.getcwd(), 'test.txt')
class_num = 21
ctx_list = [mx.gpu(0)]
batch_size = 1 # 1: unsteady converge 5:low converge


net_type = "fcn32"
net_pretrained = None #'models/fcn32_00001.params'


start_weights = -1
base_lr = 1e-2  #0.01
train_size = 502
max_epoch = 100
max_iter = max_epoch * train_size // batch_size

display_freq_train = np.maximum(int(train_size * 0.2 / batch_size),10)
display_freq_test = 1 #epoch


outdir = os.path.join(os.getcwd(),'models')
if not os.path.exists(outdir):
    os.makedirs(outdir)

class SegDataset(gluon.data.Dataset):
    def __init__(self,path_data, out_size = (512,512),train = True, padding_label=0):
        self.data_list = []
        with open(path_data,'rb') as f:
            for line in f:
                img_path, label_path = line.strip().split('|')
                self.data_list.append((img_path,label_path))
        self.img_shape = (out_size[0],out_size[1],3) #height,width
        self.label_shape = (out_size[0],out_size[1],1) #height,width
        self.train = train
        self.label_padding = padding_label
        return

    def __len__(self):
        return len(self.data_list)

    def _load_with_size(self,idx):
        img_path,label_path = self.data_list[idx]
       # img = np.float32(cv2.imread(img_path,1)) - np.asarray([0,0,0])
       # label = cv2.imread(label_path,0)
        img = cv2.imread(img_path,1)
        label = cv2.imread(label_path,0)
        H,W,C= img.shape
        assert(C == 3 and H <= self.img_shape[0] and W <= self.img_shape[1])
       # img_padding = np.zeros(self.img_shape,np.float32)
       # label_padding = np.zeros(self.label_shape,np.int64) + self.label_padding
        img_padding = np.zeros(self.img_shape,np.uint8)
        label_padding = np.zeros(self.label_shape,np.uint8) + self.label_padding
        img_padding[0:H,0:W,:] = img
        label_padding[0:H,0:W,0] = label
        #print label_padding.shape
        #label_padding = cv2.resize(label_padding,(64,64),interpolation=cv2.INTER_NEAREST)
        #label_padding = np.expand_dims(label_padding,2)
        #print label_padding.shape

        img_padding = np.float32(np.transpose(img_padding,[2,0,1]))
        img_padding[0,:,:] -= 123.68
        img_padding[1,:,:] -= 116.779
        img_padding[2,:,:] -= 103.939
        label_padding = np.transpose(label_padding,[2,0,1])

        return img_padding,label_padding

    def __getitem__(self, idx):
        img,label = self._load_with_size(idx)
        return img,label


ds_train = SegDataset(path_train_data)
ds_test = SegDataset(path_test_data)
trainIter = gluon.data.DataLoader(ds_train,batch_size=batch_size,shuffle=True,last_batch='discard')
testIter = gluon.data.DataLoader(ds_test,batch_size=batch_size,shuffle=False,last_batch='discard')

if net_type == 'fcn32':
    net = fcn.FCNx32(class_num=class_num,ctx = ctx_list[0])
elif net_type == 'fcn16' and net_pretrained is not None:
    net = fcn.FCNx16(class_num=class_num,ctx = ctx_list[0],fcnx32_path=net_pretrained)

if start_weights >= 0:
    path_weight = os.path.join(outdir,'{}_{:0>5d}.params'.format(net_type,start_weights))
    net.load_params(path_weight)
    logger.info("loading models {}".format(path_weight))
else:
    start_weights = 0
trainer = gluon.Trainer(net.collect_params(),"sgd",{"wd":0.0005})

loss_ce = gluon.loss.SoftmaxCrossEntropyLoss(axis=1,sparse_label=True,from_logits=False)


#lr_scheduler = mx.lr_scheduler.PolyScheduler(max_update=max_iter,base_lr=base_lr,pwr=1)
t = train_size //  batch_size
lr_scheduler = mx.lr_scheduler.MultiFactorScheduler(step=[5*t, 10*t, 60*t, 80*t], factor=0.01)
lr_scheduler.base_lr = base_lr


def calc_iou(pred,label):
    if isinstance(label,mx.nd.NDArray):
        label = label.asnumpy()
    if isinstance(pred, mx.nd.NDArray):
        pred = pred.asnumpy()
    pred = pred.argmax(axis=1)
    if 0 :
        idx = 0
        for p, l in zip(pred, label):
            cv2.imwrite('%d_pred.bmp'%idx,np.uint8(p) * 10)
            cv2.imwrite('%d_label.bmp'%idx,np.uint8(l[0,:,:]) * 10)
            idx += 1
    ab_and = np.sum((pred == label) * (label > 0) )#ignore background
    ab_or = np.sum( (label > 0)  + (pred > 0) ) #- ab_and
    #print ab_and, ab_or
    #if ab_or == 0:
    #    print 'label.sum() = ', label.sum()
    return np.float32(ab_and) / ab_or if ab_or > 0 else 0


iter_num = 0
for epoch in range(start_weights,max_epoch):
    for batch in trainIter:
        iter_num += 1
        data,label = batch
        data_list = utils.split_and_load(data,ctx_list)
        label_list = utils.split_and_load(label,ctx_list)
        loss_list, iou_list = [], []
        losses = []
        with mx.autograd.record():
            pred_list = [net(x) for x in data_list]
            losses = [loss_ce(pred,label) for label,pred in zip(label_list,pred_list)]
        for loss in losses:
            loss.backward()
        trainer.step(batch_size)
        loss_list.extend( [loss.asnumpy() for loss in losses])
        iou_list.extend( [calc_iou(pred,label) for pred,label in zip(pred_list,label_list) ])
        if iter_num % display_freq_train == 0:
            logger.info("iter {} train-loss {} train-iou {}".format(
                iter_num, np.asarray(loss_list).mean(), np.asarray(iou_list).mean() ) )
        trainer.set_learning_rate( lr_scheduler(iter_num) )
    if epoch % display_freq_test == 0:
        net.save_params(os.path.join(outdir,'%s_%.5d.params'%(net_type,epoch)))
        iou_list, loss_list = [], []
        for batch in testIter:
            data,label = batch
            data_list = utils.split_and_load(data,ctx_list)
            label_list = utils.split_and_load(label,ctx_list)
            pred_list = [net(x) for x in data_list]
            iou_list.extend( [calc_iou(pred,label) for pred,label in zip(pred_list,label_list)] )
            loss_list.extend( [loss_ce(pred,label).asnumpy() for pred,label in zip(pred_list, label_list)] )
        logger.info("epoch {} lr {} test-loss {} test-iou {}".format(
            epoch, trainer.learning_rate, np.asarray(loss_list).mean(), np.asarray(iou_list).mean()))







