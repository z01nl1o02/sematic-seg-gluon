import os,sys
import numpy as np
from collections import defaultdict
import cv2
from matplotlib import pyplot as plt

def run_one(input_path,class_num = 21):
    clsdict = defaultdict(int)
    with open(input_path,'rb') as f:
        for line in f:
            img_path,mark_path = line.strip().split('|')
            mark = cv2.imread(mark_path,0)
            for cls in range(class_num):
                clsdict[cls] += (mark == cls).sum()
    total = np.sum( [clsdict[key] for key in clsdict.keys()] )
    for key in clsdict.keys():
        clsdict[key] /= float(total)
    return clsdict

train_dict = run_one('train.txt')
test_dict = run_one('test.txt')

test_list = []
train_list = []
for cls in range(21):
    if cls in train_dict:
        train_list.append(train_dict[cls])
    else:
        train_list.append(0)
    if cls in test_dict:
        test_list.append( test_dict[cls] )
    else:
        test_list.append(0)
fig = plt.figure()
plt.plot(range(1,21),train_list[1:],color='red',label='train')
plt.plot(range(1,21),test_list[1:],color='blue',label='test')
plt.legend()
fig.savefig("class-stat.png")
for key in range(21):
    print train_list[key],test_list[key]

