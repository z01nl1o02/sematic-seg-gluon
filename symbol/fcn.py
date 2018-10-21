import numpy as np
import mxnet as mx
from mxnet import gluon
from mxnet import nd
from mxnet.gluon import nn
import pdb
from gluoncv.model_zoo.segbase import *


class EncodeNet_gluoncv(nn.Block):
    def __init__(self, ctx, verbose = False):
        super(EncodeNet_gluoncv,self).__init__()
        self.verbose = verbose
        base_net = get_segmentation_model(model='fcn', dataset="pascal_voc",
                                       backbone="resnet50", pretrained=False,aux=False)
        base_net.collect_params().reset_ctx(ctx=ctx)
        with self.name_scope():
            self.block = nn.Sequential()
            self.block.add(
            base_net.conv1,
            base_net.bn1,
            base_net.relu,
            base_net.maxpool,     
            base_net.layer1,
            base_net.layer2,
            base_net.layer3,
            base_net.layer4,
            )
        return

    def forward(self, *args):
        out = args[0]
        return self.block(out)

class EncodeNet(nn.Block):
    def __init__(self, ctx, verbose = False):
        super(EncodeNet,self).__init__()
        self.verbose = verbose
        base_net = gluon.model_zoo.vision.resnet50_v1(ctx=ctx,pretrained=True)

        with self.name_scope():
            self.pool8 = nn.Sequential()
            self.pool16 = nn.Sequential()
            self.pool32 = nn.Sequential()
            for layer in base_net.features[:-3]:
                self.pool8.add(layer)
            self.pool16.add(base_net.features[-3])
            self.pool32.add(base_net.features[-2])
        #for layer in self.pool8:
        #    if isinstance(layer, nn.Conv2D):
        #        layer.weight.lr_mult = 0.01
        #        if not (layer.bias is None):
        #           layer.bias.lr_mult = 0.01
        #for layer in self.pool16:
        #    if isinstance(layer, nn.Conv2D):
        #        layer.weight.lr_mult = 0.01
        #        if not (layer.bias is None):
        #            layer.bias.lr_mult = 0.01
        #for layer in self.pool32:
        #    if isinstance(layer,nn.Conv2D):
        #        layer.weight.lr_mult = 0.01
        #        if not (layer.bias is None):
        #            layer.bias.lr_mult = 0.01
        return

    def forward(self, *args):
        out = args[0]
        for layer in self.pool8:
            out = layer(out)
        out8 = out
        for layer in self.pool16:
            out = layer(out)
        out16 = out
        for layer in self.pool32:
            out = layer(out)
        out32 = out
        return (out8,out16,out32)


class FCNx8(nn.Block):
    def __init__(self, class_num,ctx):
        super(FCNx8, self).__init__()
        self.desc = "fcnx8"
        with self.name_scope():
            self.encode = EncodeNet_gluoncv(ctx)
            self.decode = nn.Sequential(prefix="decode")
            self.decode.add(
                nn.Conv2D(channels=512,kernel_size=3,padding=1,strides=1),
                nn.BatchNorm(),
                nn.Activation("relu"),
                nn.Dropout(0.1),
                nn.Conv2D(channels=class_num,kernel_size=1,padding=0,strides=1)
            )
            for layer in self.decode:
                if isinstance(layer,nn.Conv2D):
                    layer.initialize(init=mx.init.Xavier(), ctx=ctx)
                else:
                    layer.initialize(ctx=ctx)
            self.decode.collect_params().setattr('lr_mult',10)
        self.upscale = 8

    def upsampling(self, X, scale):
        #avoid nn.ConvTranspose2D() due to https://github.com/apache/incubator-mxnet/issues/11203
        _,_,H,W = X.shape
        return mx.nd.contrib.BilinearResize2D(X,height=H*scale,width=W*scale)

    def forward(self, *args):
        out = args[0].astype(np.float32)
        out = self.encode(out)
        out = self.decode(out)
        out = self.upsampling(out,self.upscale)
        return out


        

import cv2

if 0:
    ctx = mx.gpu()
    #net = FCNx32(class_num=21,ctx=ctx)
    net = FCNx(class_num=21,ctx=ctx)
    X = cv2.imread('outmax.bmp',3)
    X = np.transpose(X,(2,0,1))
    X = np.expand_dims(X,0)
    X = nd.array(X).as_in_context(ctx)
   # X = mx.nd.zeros((1,3, 512,512),ctx=ctx)
    Y = net(X)
    Y = Y.asnumpy()
    Y = np.squeeze(Y)
    Y = np.transpose(Y, (1,2,0))
    Y = np.uint8(Y)
    cv2.imwrite('output.bmp',Y)
   # print net
    print "X - >Y: {} -> {}".format(X.shape,Y.shape)
    
if 0:
    ctx = mx.gpu()
    net = FCNx8(class_num=21,ctx=ctx)
    X = mx.nd.zeros((1,3, 400,400),ctx=ctx)
    Y = net(X)
    
    print "X - >Y: {} -> {}".format(X.shape,Y.shape)


