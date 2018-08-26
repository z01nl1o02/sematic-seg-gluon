from __future__ import print_function
import os,sys
voc_root = 'C:/dataset/voc/VOCdevkit/VOC2007'
voc_label_root = os.path.join(os.getcwd(),"voc2007")

lines = []

for name in os.listdir(os.path.join(voc_label_root,'labels')):
    label_path = os.path.join( os.path.join(voc_label_root,'labels'), name)
    jpeg = os.path.splitext(name)[0] + '.jpg'
    image_path = os.path.join(os.path.join(voc_root,'JPEGImages'),jpeg)
    if os.path.exists(label_path) and os.path.exists(image_path):
        lines.append('|'.join([image_path,label_path]))
    else:
        print("error: miss file\r\n\t{}\r\n\t{}\r\n".format(image_path,label_path)
              ,file=sys.stderr)

total_num = len(lines)
train_num = int(total_num * 0.8)

train_line = lines[0:train_num]
test_line = lines[train_num:]

with open('train.txt','wb') as f:
    f.write('\n'.join(train_line))
with open('test.txt','wb') as f:
    f.write('\n'.join(test_line))


