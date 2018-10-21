import numpy as np
import cv2,os,sys
import scipy.io
import random
from mxnet import gluon


class VocAugDataset(gluon.data.Dataset):
    def __init__(self,root_dir,split, mode, base_size = 520, crop_size = 480, **kwargs):
        super(VocAugDataset,self).__init__()
        mask_dir = os.path.join(root_dir, 'cls')
        image_dir = os.path.join(root_dir, 'img')

        if split == 'train':
            split_file = os.path.join(root_dir, 'trainval.txt')
        elif split == 'val':
            split_file = os.path.join(root_dir, 'val.txt')
        else:
            raise RuntimeError('Unknown dataset split: {}'.format(split))
        self.mode = mode
        self.images = []
        self.masks = []
        self.base_size = base_size
        self.crop_size = crop_size
        self.class_num = 21
        with open(os.path.join(split_file), "r") as lines:
            for line in lines:
                image = os.path.join(image_dir, line.rstrip('\n')+".jpg")
                assert os.path.isfile(image)
                self.images.append(image)
                mask = os.path.join(mask_dir, line.rstrip('\n')+".mat")
                assert os.path.isfile(mask)
                self.masks.append(mask)

        assert (len(self.images) == len(self.masks))

    def _val_transform(self,image,target):
        h,w,c = image.shape
        if h < w:
            out_h = self.crop_size
            out_w = int(w * out_h/h)
        else:
            out_w = self.crop_size
            out_h = int(h * out_w/w)
        image = cv2.resize(image,(out_w,out_h),None,0,0,cv2.INTER_LINEAR)
        target = cv2.resize(target,(out_w,out_h),None,0,0,cv2.INTER_NEAREST)

        x = (out_w - self.crop_size)//2 if out_w > self.crop_size else 0
        y = (out_h - self.crop_size)//2 if out_h > self.crop_size else 0
        image = image[y:y+self.crop_size, x: x+self.crop_size]
        target = target[y:y+self.crop_size, x: x+self.crop_size]


        image = np.float32(image) / 255.0

        return image,target

    def _transform(self,image,target):
        #mirror
        if random.random() > 0.5:
            image = cv2.flip(image,1)
            target = cv2.flip(target,1)
        #resize accordign base_size
        rescale_size = random.randint(int(self.base_size*0.5), int(self.base_size*1.5))
        h,w,c = image.shape
        if h < w:
            out_h = rescale_size
            out_w = int(w * out_h/h)
        else:
            out_w = rescale_size
            out_h = int(h * out_w/w)
        image = cv2.resize(image,(out_w,out_h),None,0,0,cv2.INTER_LINEAR)
        target = cv2.resize(target,(out_w,out_h),None,0,0,cv2.INTER_NEAREST)

        #rotation
        deg = random.uniform(-10,10)
        rot_mat = cv2.getRotationMatrix2D((out_w/2.0, out_h/2.0),deg, 1.0)
        image = cv2.warpAffine(image,rot_mat, (out_w,out_h), None, cv2.INTER_LINEAR,cv2.BORDER_CONSTANT, 0)
        target = cv2.warpAffine(target,rot_mat, (out_w,out_h), None, cv2.INTER_NEAREST,cv2.BORDER_CONSTANT, 0)

        #crop
        if rescale_size < self.crop_size:
            padw = self.crop_size - out_w if out_w < self.crop_size else 0
            padh = self.crop_size - out_h if out_h < self.crop_size else 0
            image = np.pad(image, ((0,padh),(0,padw),(0,0)),'constant')
            target = np.pad(target,((0,padh),(0,padw)),'constant')
        h,w,c = image.shape
        x = random.randint(0, w - self.crop_size)
        y = random.randint(0, h - self.crop_size)
        image = image[y:y+self.crop_size,x:x+self.crop_size]
        target = target[y:y+self.crop_size,x:x+self.crop_size]

        #blur
        if random.random() > 0.5:
            image = cv2.GaussianBlur(image,(3,3),random.random())


        image = np.float32(image) / 255.0

        return image, target


    def _to_tensor(self,image,target):
        image = np.transpose(image,(2,0,1))
        #image = np.expand_dims(image,0)
        #target = np.expand_dims(target,0)
        return image, target

    def __getitem__(self, index):
        img = cv2.imread(self.images[index],1)
        target = self._load_mat(self.masks[index])

        if self.mode == 'train':
            img, target = self._transform(img,target)
        else:
            img, target = self._val_transform(img,target)

        img, target = self._to_tensor(img,target)
        return img, target

    def _load_mat(self, filename):
        mat = scipy.io.loadmat(filename, mat_dtype=True, squeeze_me=True,
                               struct_as_record=False)
        mask = mat['GTcls'].Segmentation
        return mask

    def __len__(self):
        return len(self.images)

    @property
    def classes(self):
        """Category names."""
        return ('background', 'airplane', 'bicycle', 'bird', 'boat', 'bottle',
                'bus', 'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse',
                'motorcycle', 'person', 'potted-plant', 'sheep', 'sofa', 'train',
                'tv')


if 0:
    ds = VocAugDataset("/home/c001/data/voc/VOCaug/dataset/",'train','train')
    for img, mask in ds:
        img = np.uint8(img * 255)
        cv2.imshow("img.jpg",img)
        cv2.imshow("mask.jpg",mask*10)
        cv2.waitKey(-1)

