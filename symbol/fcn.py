import numpy as np
import mxnet as mx
from mxnet import gluon
from mxnet import nd
from mxnet.gluon import nn


class EncodeNet(nn.Block):
    def __init__(self, ctx, verbose = False):
        super(EncodeNet,self).__init__()
        channels = 256
        self.verbose = verbose
        pretrained = gluon.model_zoo.vision.resnet50_v1(ctx=ctx,pretrained=True)
        with self.name_scope():
            self.stage_1 = nn.Sequential()
            for layer in pretrained.features[:-2]:
                #if isinstance(layer, nn.Conv2D):
                #    layer.weight.lr_mult = 1
                self.stage_1.add(layer)

            self.stage_2 = nn.Sequential()
            self.stage_2.add(
                nn.Conv2D(channels=channels,kernel_size=3,padding=1),
                nn.Activation(activation="relu"),

                nn.Conv2D(channels=channels,kernel_size=3,padding=1),
                nn.Activation(activation="relu"),

                nn.Conv2D(channels=channels,kernel_size=3,padding=1),
                nn.Activation(activation="relu"),

                nn.MaxPool2D(pool_size=2, strides=2),

                nn.Conv2D(channels=channels,kernel_size=3,padding=1),
                nn.Activation(activation="relu"),

                nn.Conv2D(channels=channels,kernel_size=3,padding=1),
                nn.Activation(activation="relu"),

                nn.Conv2D(channels=channels,kernel_size=3,padding=1),
                nn.Activation(activation="relu"),
            )
        for layer in self.stage_2:
            if isinstance(layer,nn.Conv2D):
                layer.initialize(init=mx.init.Xavier(),ctx=ctx)
            else:
                layer.initialize(ctx=ctx)

       # self.nets[-1].initialize(init = mx.init.Bilinear(),ctx=ctx)
       # self.nets[-1].weight.lr_mult = 0
       # self.nets[-1].bias.lr_mult = 0
       #
       #      for k in range(decode_start, len(self.nets)):
       #          if isinstance(self.nets[k], nn.Conv2D):
       #              self.nets[k].initialize(init = mx.init.Xavier(),ctx=ctx)
       #          else:
       #              self.nets[k].initialize(ctx=ctx)

    def forward(self, *args):
        raise Exception("no forward")
        return


class FCNx32(nn.Block):
    def __init__(self, class_num,ctx):
        super(FCNx32, self).__init__()
        encode = EncodeNet(ctx)
        with self.name_scope():
            self.stage_1 = nn.Sequential()
            for layer in encode.stage_1:
                self.stage_1.add(layer)
            self.stage_2 = nn.Sequential()
            for layer in encode.stage_2:
                self.stage_2.add(layer)
            self.decode = nn.Conv2DTranspose(channels=class_num,kernel_size=64, padding=16,strides=32)
            self.decode.initialize(mx.init.Bilinear(), ctx=ctx)
    def forward(self, *args):
        out = args[0]
        for layer_net in self.stage_1:
            out = layer_net(out)
        for layer_net in self.stage_2:
            out = layer_net(out)
        return self.decode(out)

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
    #net = FCNx16(class_num=21,ctx=ctx,fcnx32_path=None)
    net = FCNx32(class_num=21,ctx=ctx)
    X = mx.nd.zeros((1,3, 512,512),ctx=ctx)
    Y = net(X)
    #print net
    print "X - >Y: {} -> {}".format(X.shape,Y.shape)


