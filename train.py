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

path_train_data = os.path.join( os.getcwd(), 'train.txt'  )

path_test_data = os.path.join( os.getcwd(), 'test.txt')
class_num = 21
ctx_list = [mx.gpu(0)]
batch_size = 1


net_type = "fcn32"
net_pretrained = ""#'models/fcn32_00017.params' #'fcn/fcn32_00022.params'


flag_test_init = False
start_weights = 52
base_lr = 1e-3
max_epoch = 1000

display_freq_test = 1 #epoch

outdir = os.path.join(os.getcwd(),'models')
if not os.path.exists(outdir):
    os.makedirs(outdir)

class SegDataset(gluon.data.Dataset):
    def __init__(self,path_data, out_size = (160*2,160*2),train = True, padding_label=0):
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

    def _augment_resize(self, img, label):
        rnd = np.random.randint(100,400)
        scale = rnd / 200.0
        H = int( img.shape[0] * scale )
        W = int( img.shape[1] * scale )
        img = cv2.resize(img, dsize=(W,H), interpolation=cv2.INTER_LINEAR)
        label = cv2.resize(label, dsize=(W,H), interpolation=cv2.INTER_NEAREST)
        return img,label
    def _augment_noise(self,img,label):
        noise = np.random.randn(*(img.shape)) * np.random.randint(10,20)
        #cv2.imshow("src",img)
        img = np.int32(img) + np.int32(noise)
        img = np.maximum(img,0)
        img = np.minimum(img,255)
        img = np.uint8(img)
       # print sigma
       # cv2.imshow("noise",img)
        #cv2.waitKey(-1)
        return img,label
    def _augment_blur(self,img,label):
        rnd = np.random.randint(100,600)
        sigma = rnd/200.0
        img = cv2.GaussianBlur(img,(7,7),sigmaX=sigma,sigmaY=sigma)
        #print sigma
        #cv2.imshow("blur",img)
       # cv2.waitKey(-1)
        return img,label
    def _augment_flip(self,img,label):
        rnd = np.random.randint(0, 100)
        code = 2
        if rnd > 75:
            code = 1
        elif rnd > 50:
            code = 0
        elif rnd > 25:
            code = -1
        if code != 2:
            img = cv2.flip(img, code)
            label = cv2.flip(label, code)
      #  cv2.imshow("img", img)
      #  cv2.imshow("label", label * 11)
      #  cv2.waitKey(-500)
        return img,label
    def _load_with_size(self,idx):
        img_path,label_path = self.data_list[idx]
        img = cv2.imread(img_path,1)
        label = cv2.imread(label_path,0)


        if self.train and np.random.randint(0, 100) > 50:
            img,label = self._augment_flip(img,label)
        if self.train and np.random.randint(0,100) > 50:
            img,label = self._augment_resize(img,label)
        if self.train and np.random.randint(0,100) > 50:
            img,label = self._augment_blur(img,label)
        if self.train and np.random.randint(0,100) > 50:
            img,label = self._augment_noise(img,label)

        img_padding = np.zeros(self.img_shape,np.uint8)
        label_padding = np.zeros(self.label_shape,np.uint8) + self.label_padding
        width, height = np.minimum(img.shape[1], self.img_shape[1]), np.minimum(img.shape[0], self.img_shape[0])

        dx, dy = (img_padding.shape[1] - width)//2, (img_padding.shape[0] - height)//2
        sx, sy = (img.shape[1] - width)//2, (img.shape[0] - height)//2
        if self.train and np.random.randint(0,100) > 50:
            if sx > 0:
                sx = np.random.randint(0,sx)
            if sy > 0:
                sy = np.random.randint(0,sy)

        img_padding[dy:dy+height,dx:dx+width,:] = img[sy:sy+height, sx:sx+width,:]
        label_padding[dy:dy+height,dx:dx+width,0] = label[sy:sy+height, sx:sx+width]




        img_padding = np.float32(np.transpose(img_padding,[2,0,1])) / 255.0
       # img_padding[0,:,:] -= 123.68
       # img_padding[1,:,:] -= 116.779
       # img_padding[2,:,:] -= 103.939
        label_padding = np.transpose(label_padding,[2,0,1])

        #print img_padding.shape, label_padding.shape
        return img_padding,label_padding

    def __getitem__(self, idx):
        img,label = self._load_with_size(idx)
        return img,label


ds_train = SegDataset(path_train_data,train=True)
ds_test = SegDataset(path_test_data, train=False)
trainIter = gluon.data.DataLoader(ds_train,batch_size=batch_size,shuffle=True,last_batch='discard')
testIter = gluon.data.DataLoader(ds_test,batch_size=batch_size,shuffle=False,last_batch='discard')

train_size = len(ds_train)
display_freq_train = np.maximum(int(train_size * 0.2 / batch_size),1)
max_iter = max_epoch * train_size // batch_size




if net_type == 'fcn32':
    net = fcn.FCNx32(class_num=class_num,ctx = ctx_list[0])
elif net_type == 'fcn16' and net_pretrained is not None:
    net = fcn.FCNx16(class_num=class_num,ctx = ctx_list[0],fcnx32_path=net_pretrained)

if start_weights >= 0:
    path_weight = os.path.join(outdir,'{}_{:0>5d}_a.params'.format(net_type,start_weights))
    net.load_params(path_weight)
    logger.info("loading models {}".format(path_weight))
    start_weights += 1
else:
    start_weights = 0

trainer = gluon.Trainer(net.collect_params(),"sgd",{"wd":0.00005})

loss_ce = gluon.loss.SoftmaxCrossEntropyLoss(axis=1,sparse_label=True,from_logits=False)
acc = mx.metric.Accuracy()

#lr_scheduler = mx.lr_scheduler.PolyScheduler(max_update=max_iter,base_lr=base_lr,pwr=1)
t = train_size //  batch_size
lr_scheduler = mx.lr_scheduler.MultiFactorScheduler(step=[30*t, 50*t,80*t], factor=0.1)
lr_scheduler.base_lr = base_lr


def calc_iou(pred,label,name = ""):
    if isinstance(label,mx.nd.NDArray):
        label = label.asnumpy()
    if isinstance(pred, mx.nd.NDArray):
        pred = pred.asnumpy()
    pred = pred.argmax(axis=1)
    if name != "":
        img = np.uint8( np.squeeze(pred) * 10 )
        cv2.imwrite("{}_pred.bmp".format(name),img)
        img = np.uint8(np.squeeze(label) * 10)
        cv2.imwrite("{}_label.bmp".format(name),img)
    pred = np.squeeze(pred)
    label = np.squeeze(label)
    ab_and = np.sum((pred == label) * (label > 0) )#ignore background
    ab_or = np.sum( (label > 0)  + (pred > 0) ) #- ab_and
    #print ab_or, ab_and, (label > 0).sum(), (pred > 0).sum(), label.shape, pred.shape
    return np.float32(ab_and) / ab_or if ab_or > 0 else 0
    
def calc_pixel_acc(pred,label):
    if isinstance(label,mx.nd.NDArray):
        label = label.asnumpy()
    if isinstance(pred, mx.nd.NDArray):
        pred = pred.asnumpy()
    pred = pred.argmax(axis=1)
    
    hit = (pred == label).sum()
    total = (label >= 0).sum()
    
    return (hit,total)
    

class SegMetrics_mean_iou:
    def __init__(self,class_num):
        self.data = {}
        self.class_num = class_num
        for k in range(class_num):
            self.data[k] = [0,0] 
    def update(self,pred,label):
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
    

class SegMetrics_PA:
    def __init__(self):
        self.total = 0
        self.hit = 0
    def update(self,pred,label):
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

metrics_pa = SegMetrics_PA()
metrics_iou = SegMetrics_mean_iou(class_num)
iter_num = start_weights * train_size // batch_size
for epoch in range(start_weights,max_epoch):
    acc.reset()
    metrics_pa.reset()
    metrics_iou.reset()
    loss_list, iou_list = [], []
    debug_train, debug_test = 0, 0
    if epoch % 2 == 0:
        debug_train, debug_test = 1,1
    if flag_test_init and epoch == start_weights:
        print 'init test...'
    else:
        for batch in trainIter:
            iter_num += 1
            data,label = batch
            if 0:
                img =  np.transpose(np.squeeze(data.asnumpy()),(1,2,0))
                cv2.imwrite("debug/{}_train_image.bmp".format(epoch),img.astype(np.uint8))

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
            loss_list.extend( [loss.asnumpy() for loss in losses])
            debug_name = ""
            if debug_train == 1:
                debug_name = "debug/{}_train".format(epoch)
                debug_train = 0
            iou_list.extend( [calc_iou(pred,label,debug_name) for pred,label in zip(pred_list,label_list) ])
            map(lambda (pred,label):metrics_iou.update(pred,label), zip(pred_list, label_list))
            for pred, label in zip(pred_list, label_list):
                acc.update(preds = pred, labels=label)
            if iter_num % display_freq_train == 0:
                logger.info("iter {} train-loss {} train-iou {} {} {}".format(
                    iter_num, np.asarray(loss_list).mean(), np.asarray(iou_list).mean(), acc.get(), metrics_iou.get() ))
            trainer.set_learning_rate( lr_scheduler(iter_num) )
            
    if (epoch % display_freq_test == 0) or (epoch == start_weights and flag_test_init):
        acc.reset()
        metrics_pa.reset()
        metrics_iou.reset()
        #if epoch % 100 == 0:
        net.save_params(os.path.join(outdir,'%s_%.5d.params'%(net_type,epoch)))
        iou_list, loss_list = [], []
        for batch in testIter:
            data,label = batch
            if 0:
                img = np.squeeze(data.asnumpy())
                img = np.transpose(img, (1,2,0))
                img = img.astype(np.uint8)
                cv2.imwrite("debug/{}_test_image.bmp".format(epoch),img)

            data_list = utils.split_and_load(data,ctx_list)
            label_list = utils.split_and_load(label,ctx_list)
            #with mx.autograd.record():
            pred_list = [net(x) for x in data_list]
            nd.waitall()
            debug_name = ""
            if debug_test == 1:
                debug_name = "debug/{}_test".format(epoch)
                debug_test = 0
            iou_list.extend( [calc_iou(pred,label,debug_name) for pred,label in zip(pred_list,label_list)] )
            map(lambda (pred,label):metrics_iou.update(pred,label), zip(pred_list, label_list))
            loss_list.extend( [loss_ce(pred,label).asnumpy() for pred,label in zip(pred_list, label_list)] )
            for pred, label in zip(pred_list, label_list):
                acc.update(preds = pred, labels = label)

        logger.info("epoch {} lr {} test-loss {} test-iou {} {} {}".format(
            epoch, trainer.learning_rate, np.asarray(loss_list).mean(), np.asarray(iou_list).mean(), acc.get(), metrics_iou.get() ))








