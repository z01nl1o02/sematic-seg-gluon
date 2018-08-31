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
        out = self.conv(out)
        out = self.decode(out,32)
        return out

class FCNx16(nn.Block):
    def __init__(self, fcnx32_path, class_num,ctx):
        super(FCNx16, self).__init__()
        fcnx32 = FCNx32(class_num,ctx)
        if fcnx32_path is not None:
            fcnx32.load_params(fcnx32_path)
        with self.name_scope():
            self.stage_1 = nn.Sequential()
            for layer in fcnx32.stage_1:
                self.stage_1.add(layer)
            self.stage_2 = nn.Sequential()
            for layer in fcnx32.stage_2:
                self.stage_2.add(layer)
            self.decode_1 = nn.Conv2DTranspose(channels=class_num,kernel_size=32, padding=8,strides=16)
            self.decode_1.initialize(init=mx.init.Bilinear(),ctx=ctx)

            self.conv_1 = nn.Conv2D(channels=class_num,kernel_size=1)
            self.conv_1.initialize(init=mx.init.Xavier(), ctx=ctx)

            self.decode_2 = nn.Conv2DTranspose(channels=class_num,kernel_size=4, padding=1,strides=2)
            self.decode_2.initialize(init=mx.init.Bilinear(),ctx=ctx)
    def forward(self, *args):
        out_1 = args[0]
        for layer in self.stage_1:
            out_1 = layer(out_1)
        out_2 = out_1
        for layer in self.stage_2:
            out_2 = layer(out_2)
        decode_2 = self.decode_2(out_2)
        out_1 = self.conv_1(out_1)
        out = self.decode_1(decode_2 + out_1)
        return out


if 0:
    ctx = mx.gpu()
    net = FCNx32(class_num=21,ctx=ctx)
    X = mx.nd.zeros((1,3, 512,512),ctx=ctx)
    Y = net(X)
    print net
    print "X - >Y: {} -> {}".format(X.shape,Y.shape)


