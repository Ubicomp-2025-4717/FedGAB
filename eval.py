import os

import numpy as np
import torch

from classification import (Classification, cvtColor, letterbox_image,
                            preprocess_input)
from utils.utils import letterbox_image
from utils.utils_metrics import evaluteTop1_5

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn

from nets import get_model_from_name
from utils.utils import (cvtColor, get_classes, letterbox_image,
                         preprocess_input, show_config)
import argparse
parser = argparse.ArgumentParser(description="Evaluation")
parser.add_argument("--model_path", type=str, default="", help="initial_model_weight")
parser.add_argument("--save_dir", type=str, default="", help="results_save_path")
parser.add_argument("--test_annotation_path_t", type=str, default="", help="test_dataset_t")
parser.add_argument("--test_annotation_path_v", type=str, default="", help="test_dataset_v")
parames = parser.parse_args()

# --------------------------------------------#
#   使用自己训练好的模型预测需要修改3个参数
#   model_path和classes_path和backbone都需要修改！
# --------------------------------------------#
class Classification(object):
    _defaults = {
        # --------------------------------------------------------------------------#
        #   使用自己训练好的模型进行预测一定要修改model_path和classes_path！
        #   model_path指向logs文件夹下的权值文件，classes_path指向model_data下的txt
        #   如果出现shape不匹配，同时要注意训练时的model_path和classes_path参数的修改
        # --------------------------------------------------------------------------#
        "model_path": parames.model_path,
        "classes_path": '/home/zhiqiang/PycharmProjects/multi-model-vehicle/classification-pytorch-main/model_data/car_classes.txt',
        # --------------------------------------------------------------------#
        #   输入的图片大小
        # --------------------------------------------------------------------#
        "input_shape": [320, 240],
        # --------------------------------------------------------------------#
        #   所用模型种类：
        #   mobilenetv2、
        #   resnet18、resnet34、resnet50、resnet101、resnet152
        #   vgg11、vgg13、vgg16、vgg11_bn、vgg13_bn、vgg16_bn、
        #   vit_b_16、
        #   swin_transformer_tiny、swin_transformer_small、swin_transformer_base
        # --------------------------------------------------------------------#
        "backbone": 'md_a',
        # --------------------------------------------------------------------#
        #   该变量用于控制是否使用letterbox_image对输入图像进行不失真的resize
        #   否则对图像进行CenterCrop
        # --------------------------------------------------------------------#
        "letterbox_image": False,
        # -------------------------------#
        #   是否使用Cuda
        #   没有GPU可以设置成False
        # -------------------------------#
        "cuda": True
    }

    @classmethod
    def get_defaults(cls, n):
        if n in cls._defaults:
            return cls._defaults[n]
        else:
            return "Unrecognized attribute name '" + n + "'"

    # ---------------------------------------------------#
    #   初始化classification
    # ---------------------------------------------------#
    def __init__(self, **kwargs):
        self.__dict__.update(self._defaults)
        for name, value in kwargs.items():
            setattr(self, name, value)

        # ---------------------------------------------------#
        #   获得种类
        # ---------------------------------------------------#
        self.class_names, self.num_classes = get_classes(self.classes_path)
        self.generate()

        show_config(**self._defaults)

    # ---------------------------------------------------#
    #   获得所有的分类
    # ---------------------------------------------------#
    def generate(self):
        # ---------------------------------------------------#
        #   载入模型与权值
        # ---------------------------------------------------#
        if self.backbone not in ['vit_b_16', 'swin_transformer_tiny', 'swin_transformer_small',
                                 'swin_transformer_base']:
            self.model = get_model_from_name[self.backbone](num_classes=self.num_classes, pretrained=False)
        else:
            self.model = get_model_from_name[self.backbone](input_shape=self.input_shape, num_classes=self.num_classes,
                                                            pretrained=False)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.load_state_dict(torch.load(self.model_path, map_location=device))
        self.model = self.model.eval()
        print('{} model, and classes loaded.'.format(self.model_path))

        if self.cuda:
            self.model = nn.DataParallel(self.model)
            self.model = self.model.cuda()

    # ---------------------------------------------------#
    #   检测图片
    # ---------------------------------------------------#
    def detect_image(self, image):
        # ---------------------------------------------------------#
        #   在这里将图像转换成RGB图像，防止灰度图在预测时报错。
        #   代码仅仅支持RGB图像的预测，所有其它类型的图像都会转化成RGB
        # ---------------------------------------------------------#
        image = cvtColor(image)
        # ---------------------------------------------------#
        #   对图片进行不失真的resize
        # ---------------------------------------------------#
        image_data = letterbox_image(image, [self.input_shape[1], self.input_shape[0]], self.letterbox_image)
        # ---------------------------------------------------------#
        #   归一化+添加上batch_size维度+转置
        # ---------------------------------------------------------#
        image_data = np.transpose(np.expand_dims(preprocess_input(np.array(image_data, np.float32)), 0), (0, 3, 1, 2))

        with torch.no_grad():
            photo = torch.from_numpy(image_data)
            if self.cuda:
                photo = photo.cuda()
            # ---------------------------------------------------#
            #   图片传入网络进行预测
            # ---------------------------------------------------#
            preds = torch.softmax(self.model(photo)[0], dim=-1).cpu().numpy()
        # ---------------------------------------------------#
        #   获得所属种类
        # ---------------------------------------------------#
        class_name = self.class_names[np.argmax(preds)]
        probability = np.max(preds)

        # ---------------------------------------------------#
        #   绘图并写字
        # ---------------------------------------------------#
        plt.subplot(1, 1, 1)
        plt.imshow(np.array(image))
        plt.title('Class:%s Probability:%.3f' % (class_name, probability))
        plt.show()
        return class_name


#------------------------------------------------------#
#   test_annotation_path    测试图片路径和标签
#------------------------------------------------------#
test_annotation_path    = 'test.txt'
#------------------------------------------------------#
#   metrics_out_path        指标保存的文件夹
#------------------------------------------------------#
metrics_out_path        = parames.save_dir

class Eval_Classification(Classification):
    def detect_image(self, image):        
        #---------------------------------------------------------#
        #   在这里将图像转换成RGB图像，防止灰度图在预测时报错。
        #   代码仅仅支持RGB图像的预测，所有其它类型的图像都会转化成RGB
        #---------------------------------------------------------#
        image,image_=image
        image       = cvtColor(image)
        #---------------------------------------------------#
        #   对图片进行不失真的resize
        #---------------------------------------------------#
        image_data  = letterbox_image(image, [self.input_shape[1], self.input_shape[0]], self.letterbox_image)
        #---------------------------------------------------------#
        #   归一化+添加上batch_size维度+转置
        #---------------------------------------------------------#
        image_data  = np.transpose(np.expand_dims(preprocess_input(np.array(image_data, np.float32)), 0), (0, 3, 1, 2))

        image_       = cvtColor(image_)
        #---------------------------------------------------#
        #   对图片进行不失真的resize
        #---------------------------------------------------#
        image_data_  = letterbox_image(image_, [self.input_shape[1], self.input_shape[0]], self.letterbox_image)
        #---------------------------------------------------------#
        #   归一化+添加上batch_size维度+转置
        #---------------------------------------------------------#
        image_data_  = np.transpose(np.expand_dims(preprocess_input(np.array(image_data_, np.float32)), 0), (0, 3, 1, 2))
        img_pair=[]
        img_pair.append(image_data)
        img_pair.append(image_data_)
        img_pair=np.array(img_pair)


        with torch.no_grad():
            photo   = torch.from_numpy(img_pair).type(torch.FloatTensor)
            if self.cuda:
                photo = photo.cuda()
            #---------------------------------------------------#
            #   图片传入网络进行预测
            #---------------------------------------------------#
            preds   = torch.softmax(self.model(photo)[0], dim=-1).cpu().numpy()

        return preds

if __name__ == "__main__":
    if not os.path.exists(metrics_out_path):
        os.makedirs(metrics_out_path)
            
    classfication = Eval_Classification()
    
    with open(parames.test_annotation_path_t,"r") as f:
        lines = f.readlines()

    with open(parames.test_annotation_path_v,"r") as f:
        lines_ = f.readlines()


    top1, top5, Recall, Precision = evaluteTop1_5(classfication, lines,lines_, metrics_out_path)
    print("top-1 accuracy = %.2f%%" % (top1*100))
    print("top-5 accuracy = %.2f%%" % (top5*100))
    print("mean Recall = %.2f%%" % (np.mean(Recall)*100))
    print("mean Precision = %.2f%%" % (np.mean(Precision)*100))
