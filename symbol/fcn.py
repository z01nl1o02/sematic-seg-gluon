import numpy as np
import mxnet as mx
from mxnet import gluon
from mxnet import nd
from mxnet.gluon import nn
import pdb
from gluoncv.model_zoo.segbase import *


class EncodeNet_2(nn.Block):
    def __init__(self, ctx, verbose = False):
        super(EncodeNet_2,self).__init__()
        self.verbose = verbose
        base_net = get_segmentation_model(model='fcn', dataset="pascal_voc",
                                       backbone="resnet50", pretrained=True,aux=True)
        base_net.collect_params().reset_ctx(ctx=ctx)
        #print base_net
        #print base_net
        with self.name_scope():
            self.pool8 = nn.Sequential()
            self.pool8.add(
            base_net.conv1,
            base_net.bn1,
            base_net.relu,
            base_net.maxpool,     
            base_net.layer1,
            base_net.layer2,
            )
            self.pool16 = nn.Sequential()
            self.pool16.add(
                base_net.layer3,
            )
            self.pool32 = nn.Sequential()
            self.pool32.add(
                base_net.layer4,
                )
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


class FCNx32(nn.Block):
    def __init__(self, class_num,ctx):
        super(FCNx32, self).__init__()
        with self.name_scope():
            self.encode = EncodeNet_2(ctx)
            # self.encode = nn.Sequential(prefix="fcn_")
            # self.encode.add(
            #     nn.Conv2D(channels=512,kernel_size=3,padding=1,strides=32),
            #     nn.BatchNorm(),
            #     nn.Activation("relu"),
            #     #nn.Dropout(0.5),
            #
            #     # nn.Conv2D(channels=128,kernel_size=3,padding=1,strides=4),
            #     # nn.BatchNorm(),
            #     # nn.Activation("relu"),
            #
            #     # nn.Conv2D(channels=128,kernel_size=3,padding=1,strides=4),
            #     # nn.BatchNorm(),
            #     # nn.Activation("relu"),
            #
            # )
            # for layer in self.encode:
            #     layer.initialize(ctx=ctx)
            self.preproces_up = nn.Sequential(prefix="fcn2_")
            self.preproces_up.add(
                nn.Conv2D(channels=512,kernel_size=3,padding=1,strides=1),
                nn.BatchNorm(),
                nn.Activation("relu"),
                nn.Dropout(0.1),
                nn.Conv2D(channels=class_num,kernel_size=1,padding=0,strides=1)
            )
            for layer in self.preproces_up:
                if isinstance(layer,nn.Conv2D):
                    layer.initialize(init=mx.init.Xavier(), ctx=ctx)
                else:
                    layer.initialize(ctx=ctx)

    def decode(self, X, scale):
        #avoid nn.ConvTranspose2D() due to https://github.com/apache/incubator-mxnet/issues/11203
        _,_,H,W = X.shape
        return mx.nd.contrib.BilinearResize2D(X,height=H*scale,width=W*scale)
        #return mx.nd.UpSampling(X,scale=scale,sample_type="nearest")

    def forward(self, *args):
        out = args[0]
        for layer in self.encode.pool8:
            out = layer(out)
        #print 'out of pool8 ',out.shape
        for layer in self.encode.pool16:
            out = layer(out)
        #print 'out of pool16 ',out.shape
        for layer in self.encode.pool32:
            out = layer(out)
        #print 'out of pool32 ',out.shape
        for layer in self.preproces_up:
            out = layer(out)
        out = self.decode(out,8)

        # for layer in self.encode:
        #     out = layer(out)
        # for layer in self.preproces_up:
        #     out = layer(out)
        # #print self.encode[0].weight.data().sum().asnumpy()
        # out = self.decode(out,32)
        return out

class FCNx16(nn.Block):
    def __init__(self, fcnx32_path, class_num,ctx):
        super(FCNx16, self).__init__()
        with self.name_scope():
            self.parentNet = FCNx32(class_num,ctx)
            if fcnx32_path is not None:
                self.parentNet.load_params(fcnx32_path,ctx=ctx)

            # self.preprocess_up = nn.Sequential()
            # self.preprocess_up.add(
            #     nn.Dropout(0.5),
            #     nn.Conv2D(channels=class_num,kernel_size=1,padding=0,strides=1,use_bias=False)
            # )
            # for layer in self.preprocess_up:
            #     if isinstance(layer, nn.Conv2D):
            #         layer.initialize(init=mx.init.Xavier(), ctx=ctx)
            #     else:
            #         layer.initialize(ctx=ctx)

            self.preprocess = nn.Sequential()
            self.preprocess.add(
                nn.Dropout(0.5),
                nn.Conv2D(channels=class_num,kernel_size=1,padding=0,strides=1,use_bias=False)
            )
            for layer in self.preprocess:
                if isinstance(layer, nn.Conv2D):
                    layer.initialize(init=mx.init.Xavier(), ctx=ctx)
                else:
                    layer.initialize(ctx=ctx)

    def decode(self, X, scale):
        #avoid nn.ConvTranspose2D() due to https://github.com/apache/incubator-mxnet/issues/11203
        _,_,H,W = X.shape
        return mx.nd.contrib.BilinearResize2D(X,height=H*scale,width=W*scale)

    def forward(self, *args):
        pool8 = args[0]
        for layer in self.parentNet.encode.pool8:
            pool8 = layer(pool8)
        pool16 = pool8
        for layer in self.parentNet.encode.pool16:
            pool16 = layer(pool16)
        pool32 = pool16
        for layer in self.parentNet.encode.pool32:
            pool32 = layer(pool32)

        #print pool32.shape, pool32.shape
        for layer in self.parentNet.preproces_up:
            pool32 = layer(pool32)

        # for layer in self.preprocess_up:
        #     pool32 = layer(pool32)
        pool32up = self.decode(pool32,2)

        for layer in self.preprocess:
            pool16 = layer(pool16)
        out = pool16 + pool32up
        out = self.decode(out,16)
        return out


        


class FCNx(nn.Block):
    def __init__(self, class_num,ctx):
        super(FCNx, self).__init__()
        return

    def decode(self, X, scale):
        #avoid nn.ConvTranspose2D() due to https://github.com/apache/incubator-mxnet/issues/11203
        _,_,H,W = X.shape
        return mx.nd.contrib.BilinearResize2D(X,height=H*scale,width=W*scale)
        #return mx.nd.UpSampling(X,scale=scale,sample_type="nearest")

    def forward(self, *args):
        out = args[0]
        out = self.decode(out,32)
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
    net = FCNx32(class_num=21,ctx=ctx)
    #print net
    #net = FCNx16(class_num=21,fcnx32_path="../fcn/fcn32_00099.params",ctx=ctx)
    X = mx.nd.zeros((1,3, 160,160),ctx=ctx)
    Y = net(X)
    
    print "X - >Y: {} -> {}".format(X.shape,Y.shape)


