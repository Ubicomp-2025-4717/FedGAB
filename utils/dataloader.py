import cv2
import numpy as np
import torch
import torch.utils.data as data
from PIL import Image
import torch.nn.functional as F
from .utils import cvtColor, preprocess_input
from .utils_aug import CenterCrop, ImageNetPolicy, RandomResizedCrop, Resize


class DataGenerator(data.Dataset):
    def __init__(self, annotation_lines_r,annotation_lines_i,annotation_lines_a,input_shape, random=True, autoaugment_flag=True):
        self.annotation_lines_r = annotation_lines_r
        self.annotation_lines_i = annotation_lines_i
        self.annotation_lines_a = annotation_lines_a
        self.input_shape        = input_shape
        self.random             = random
        self.autoaugment_flag   = autoaugment_flag

        if self.autoaugment_flag:
            self.resize_crop = RandomResizedCrop(input_shape)
            self.policy      = ImageNetPolicy()
            
            self.resize      = Resize(input_shape[0] if input_shape[0] == input_shape[1] else input_shape)
            self.center_crop = CenterCrop(input_shape)

    def __len__(self):
        return len(self.annotation_lines_r)

    def __getitem__(self, index):
        annotation_path_r = self.annotation_lines_r[index].split(';')[1].split()[0]
        image = Image.open(annotation_path_r)
        #------------------------------#
        #   读取图像并转换成RGB图像
        #------------------------------#
        image   = cvtColor(image)
        if self.autoaugment_flag:
            image = self.AutoAugment(image, random=self.random)
        else:
            image = self.get_random_data(image, self.input_shape, random=self.random)
        image_r = np.transpose(preprocess_input(np.array(image).astype(np.float32)), [2, 0, 1])
#----------------------------------------------------------------------------------------------------------------
        annotation_path_i = self.annotation_lines_i[index].split(';')[1].split()[0]
        image_ = Image.open(annotation_path_i)
        #------------------------------#
        #   读取图像并转换成RGB图像
        #------------------------------#
        image_   = cvtColor(image_)
        if self.autoaugment_flag:
            image_ = self.AutoAugment(image_, random=self.random)
        else:
            image_ = self.get_random_data(image_, self.input_shape, random=self.random)
        image_i = np.transpose(preprocess_input(np.array(image_).astype(np.float32)), [2, 0, 1])
#----------------------------------------------------------------------------------------------------------------------------
        annotation_path_a = self.annotation_lines_a[index].split(';')[1].split()[0]
        file = torch.load(annotation_path_a)
        image_a = np.transpose(np.array(file).astype(np.float32), [2, 0, 1])
        image_a = F.interpolate(torch.tensor(image_a).unsqueeze(0), size=(320, 224), mode='bilinear',
                                             align_corners=False).squeeze(0).numpy()

        # 扩展通道数，从 (1, 320, 224) 扩展为 (3, 320, 224)
        image_a = np.repeat(image_a, 3, axis=0)

        img_pair=[]
        img_pair.append(image_r)
        img_pair.append(image_i)
        img_pair.append(image_a)
        img_pair=np.array(img_pair)

        y = int(self.annotation_lines_r[index].split(';')[0])
        # print(img_pair.shape)
        # return image, y
        return img_pair, y

    def rand(self, a=0, b=1):
        return np.random.rand()*(b-a) + a

    def get_random_data(self, image, input_shape, jitter=.3, hue=.1, sat=1.5, val=1.5, random=True):
        #------------------------------#
        #   获得图像的高宽与目标高宽
        #------------------------------#
        iw, ih  = image.size
        h, w    = input_shape

        if not random:
            scale = min(w/iw, h/ih)
            nw = int(iw*scale)
            nh = int(ih*scale)
            dx = (w-nw)//2
            dy = (h-nh)//2

            #---------------------------------#
            #   将图像多余的部分加上灰条
            #---------------------------------#
            image       = image.resize((nw,nh), Image.BICUBIC)
            new_image   = Image.new('RGB', (w,h), (128,128,128))
            new_image.paste(image, (dx, dy))
            image_data  = np.array(new_image, np.float32)

            return image_data

        #------------------------------------------#
        #   对图像进行缩放并且进行长和宽的扭曲
        #------------------------------------------#
        new_ar = iw/ih * self.rand(1-jitter,1+jitter) / self.rand(1-jitter,1+jitter)
        scale = self.rand(.75, 1.5)
        if new_ar < 1:
            nh = int(scale*h)
            nw = int(nh*new_ar)
        else:
            nw = int(scale*w)
            nh = int(nw/new_ar)
        image = image.resize((nw,nh), Image.BICUBIC)

        #------------------------------------------#
        #   将图像多余的部分加上灰条
        #------------------------------------------#
        dx = int(self.rand(0, w-nw))
        dy = int(self.rand(0, h-nh))
        new_image = Image.new('RGB', (w,h), (128,128,128))
        new_image.paste(image, (dx, dy))
        image = new_image

        #------------------------------------------#
        #   翻转图像
        #------------------------------------------#
        flip = self.rand()<.5
        if flip: image = image.transpose(Image.FLIP_LEFT_RIGHT)
        
        rotate = self.rand()<.5
        if rotate: 
            angle = np.random.randint(-15,15)
            a,b = w/2,h/2
            M = cv2.getRotationMatrix2D((a,b),angle,1)
            image = cv2.warpAffine(np.array(image), M, (w,h), borderValue=[128, 128, 128]) 

        image_data      = np.array(image, np.uint8)
        #---------------------------------#
        #   对图像进行色域变换
        #   计算色域变换的参数
        #---------------------------------#
        r               = np.random.uniform(-1, 1, 3) * [hue, sat, val] + 1
        #---------------------------------#
        #   将图像转到HSV上
        #---------------------------------#
        hue, sat, val   = cv2.split(cv2.cvtColor(image_data, cv2.COLOR_RGB2HSV))
        dtype           = image_data.dtype
        #---------------------------------#
        #   应用变换
        #---------------------------------#
        x       = np.arange(0, 256, dtype=r.dtype)
        lut_hue = ((x * r[0]) % 180).astype(dtype)
        lut_sat = np.clip(x * r[1], 0, 255).astype(dtype)
        lut_val = np.clip(x * r[2], 0, 255).astype(dtype)

        image_data = cv2.merge((cv2.LUT(hue, lut_hue), cv2.LUT(sat, lut_sat), cv2.LUT(val, lut_val)))
        image_data = cv2.cvtColor(image_data, cv2.COLOR_HSV2RGB)
        return image_data
    
    def AutoAugment(self, image, random=True):
        if not random:
            image = self.resize(image)
            image = self.center_crop(image)
            return image

        #------------------------------------------#
        #   resize并且随即裁剪
        #------------------------------------------#
        image = self.resize_crop(image)
        
        #------------------------------------------#
        #   翻转图像
        #------------------------------------------#
        flip = self.rand()<.5
        if flip: image = image.transpose(Image.FLIP_LEFT_RIGHT)
        
        #------------------------------------------#
        #   随机增强
        #------------------------------------------#
        image = self.policy(image)
        return image
            
# def detection_collate(batch):
#     images = []
#     targets = []
#     for image, y in batch:
#         images.append(image)
#         targets.append(y)
#     images  = torch.from_numpy(np.array(images)).type(torch.FloatTensor)
#     targets = torch.from_numpy(np.array(targets)).type(torch.FloatTensor).long()
#     return images, targets

def detection_collate(batch):
    left_images = []
    centre_images= []
    right_images = []
    labels = []
    for pair_imgs, pair_labels in batch:
        left_images.append(pair_imgs[0])
        centre_images.append(pair_imgs[1])
        right_images.append(pair_imgs[2])
        labels.append(pair_labels)

    images = torch.from_numpy(np.array([left_images,centre_images,right_images])).type(torch.FloatTensor)
    labels = torch.from_numpy(np.array(labels)).type(torch.FloatTensor).long()
    # print("images",images.size())
    return images, labels
