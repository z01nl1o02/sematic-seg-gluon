import numpy as np
import mxnet as mx
from mxnet import gluon
from mxnet import nd
from mxnet.gluon import nn


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
        for layer in self.pool8:
            if isinstance(layer, nn.Conv2D):
                layer.weight.lr_mult = 0.01
        return

    def forward(self, *args):
        out = args[0]
        for layer in self.pool8:
            out = layer(out)
        for layer in self.pool16:
            out = layer(out)
        for layer in self.pool32:
            out = layer(out)
        return out


class FCNx32(nn.Block):
    def __init__(self, class_num,ctx):
        super(FCNx32, self).__init__()
        with self.name_scope():
            self.encode = EncodeNet(ctx)

            self.dropout = nn.Dropout(0.5)
            self.conv = nn.Conv2D(channels=class_num,kernel_size=1,padding=0,strides=1,use_bias=False)
            self.conv.initialize(init=mx.init.Xavier(), ctx=ctx)

    def decode(self, X, scale):
        #avoid nn.ConvTranspose2D() due to https://github.com/apache/incubator-mxnet/issues/11203
        _,_,H,W = X.shape
        return mx.nd.contrib.BilinearResize2D(X,height=H*scale,width=W*scale)

    def forward(self, *args):
        out = args[0]
        for layer in self.encode.pool8:
            out = layer(out)
        for layer in self.encode.pool16:
            out = layer(out)
        for layer in self.encode.pool32:
            out = layer(out)
        out = self.dropout(out)
        out = self.conv(out)
        out = self.decode(out,32)
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
        pool32 = self.parentNet.dropout(pool32)
        pool32 = self.parentNet.conv(pool32)
        # for layer in self.preprocess_up:
        #     pool32 = layer(pool32)
        pool32up = self.decode(pool32,2)

        for layer in self.preprocess:
            pool16 = layer(pool16)
        out = pool16 + pool32up
        out = self.decode(out,16)
        return out


if 0:
    ctx = mx.gpu()
    #net = FCNx32(class_num=21,ctx=ctx)
    net = FCNx16(class_num=21,fcnx32_path="../fcn/fcn32_00099.params",ctx=ctx)
    X = mx.nd.zeros((1,3, 512,512),ctx=ctx)
    Y = net(X)
   # print net
    print "X - >Y: {} -> {}".format(X.shape,Y.shape)


