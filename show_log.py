from matplotlib import pyplot as plt
import numpy as np
from collections import defaultdict
import re
from scipy.interpolate import spline,interp1d

logpath = 'log/log.2018-09-13-07-05-57.txt'

def show_one(data,name,color = None,smooth=False):
    X = np.array( [d[0] for d in data[name]] )
    Y = np.array( [d[1] for d in data[name]] )
    X = np.reshape(X,len(X))
    Y = np.reshape(Y,len(Y))
    if smooth:
        Xnew = np.linspace(X.min(), X.max(), 300)
        #Ynew = spline(X,Y,Xnew)
        #print X.shape, Y.shape
        func = interp1d(X,Y,kind="slinear")
        Ynew = func(Xnew)
        X,Y = Xnew, Ynew
    if color is None:
        color = np.random.random((1,3)).tolist()[0]
    plt.plot(X,Y,color=color,label=name)

data = defaultdict(list)
with open(logpath,'rb') as f:
    for line in f:
        if re.search('loading models',line):
            continue
        print line
        iter = re.findall(r'iter (\d+)',line)
        train_loss = re.findall(r'train-loss (\d+\.\d+)',line)
        train_iou = re.findall(r'train-iou (\S+)',line)
        test_iou = re.findall(r'test-iou (\S+)',line)
        test_loss = re.findall(r'test-loss (\S+)',line)
        epoch = re.findall(r'epoch (\d+)',line)
        lr = re.findall(r'lr (\S+) ',line)

        if len(iter) > 0:
            iter,train_loss,train_iou = np.int64(iter[0] ), np.float32(train_loss[0]), np.float32(train_iou)
            data['train_loss'].append(  (iter, train_loss) )
            data['train_iou'].append( (iter, train_iou))
        elif len(epoch) > 0:
            iter,lr,test_loss,test_iou = data['train_loss'][-1][0], np.float32(lr[0]), np.float32(test_loss[0]), np.float32(test_iou[0])
            data['lr'].append( (iter, lr) )
            data['test_loss'].append( (iter,test_loss))
            data['test_iou'].append((iter,test_iou))
            
key_for_plot1 = 'train_loss,test_loss,lr'.split(',')
key_for_plot2 = 'train_iou,test_iou,lr'.split(',')


fig_loss = plt.figure()
for key in key_for_plot1:
    show_one(data,key,smooth=False)
plt.legend()
fig_loss.savefig("loss.png")


fig_iou = plt.figure()
for key in key_for_plot2:
    if key == 'train_iou':
        smooth = False #True
    else:
        smooth = False
    show_one(data,key,smooth=smooth)
plt.legend()
fig_iou.savefig("iou.png")
