import random
import os
import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms as T  # 可以随机变换
from torchvision.transforms import functional as F  # 变换的参数需要手动给出

class MedicalDataSet(Dataset):
    def __init__(self, annotation, input_shape, num_classes, train, dataset_path):
        """
        初始化数据集的基本信息
        :param names: 除去后缀的图像名称列表，如图片名为1.jpg，列表中的值就是1，也可以看成是图像的编号
        :param input_shape: 输入到网络中的图像尺寸
        :param num_classes: 类别数目（包含背景）
        :param train: 一个bool值，训练集还是测试集，True表示训练集
        :param dataset_path: 数据集的根路径
        """
        super(MedicalDataSet, self).__init__()

        ## 删除了name 增加了annotation
        self.annotations=annotation
        self.length = len(self.annotations)  # 数据集的图像总数
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.train = train
        self.dataset_path = dataset_path

    def __len__(self):
        """
        返回数据集大小
        :return: 数据集大小
        """
        return self.length

    def __getitem__(self, index):
        """
        根据索引得到对应的原始图像和分割图像，并进行处理和增强
        :param index: 数据索引
        :return:
        """
        name = self.annotations[index].strip()
        # 读取图像
        input_image = Image.open(os.path.join(os.path.join(self.dataset_path, "Image"), name + ".bmp"))  # 原始图像
        pre_seg_image = Image.open(os.path.join(os.path.join(self.dataset_path, "Mask"), name + ".bmp"))  # 原始标注图像，只有纯黑色和纯白色，0和255

        input_image, pre_seg_image = augment(input_image, pre_seg_image, (512,512),work=bool(False))  # 数据增强

        input_image = np.array(input_image, dtype=np.float64) / 255  # 归一化到[0, 1]
        input_image = np.transpose(input_image, [2, 0, 1])  # 通道数挪到最前面
        input_image = torch.FloatTensor(input_image)  # 转为tensor

        # seg_image = np.zeros_like(pre_seg_image)  # 初始化一个全是背景类别的标注图像，背景类别编号为0
        #seg_image[pre_seg_image <= 127.5] = 1  # 把像素值小于等于127.5（255的一半）的点设置为需要分割的类别，编号为1

        seg_image=np.array(pre_seg_image)
        seg_image[seg_image == 255] = 1
        seg_image = np.eye(self.num_classes + 1)[seg_image.reshape([-1])]  # 得到对应的one-hot编码，shape=(w*h, num_classes+1)
        seg_image = seg_image.reshape((int(self.input_shape[0]), int(self.input_shape[1]), self.num_classes + 1))  # shape=(h, w, num_classes+1)
        seg_image = torch.FloatTensor(seg_image)  # 转为tensor

        return input_image, seg_image

def augment(image, label, input_shape, hue=.3, sat=.5, val=.5, work=True):
    """
    简单的数据增强
    :param image: 样本
    :param label: 标签
    :param input_shape: 最终大小
    :param hue: 色调随机值因子
    :param sat: 饱和度随机值因子
    :param val: 亮度随机值因子
    :param work: 是否图像增强
    :return:
    """
    image = image.convert('RGB')  # 换成RGB三通道
    h, w = input_shape

    # 不进行数据增强，验证集和测试集都不进行数据增强，进行不失真的resize，然后直接返回
    if not work:
        iw, ih = image.size
        # 长宽扩大相同规模就会避免失真（个人理解）
        scale = min(w / iw, h / ih)
        nw = int(iw * scale)
        nh = int(ih * scale)

        image = image.resize((nw, nh), Image.BICUBIC)
        new_image = Image.new('RGB', [w, h], (128, 128, 128))
        new_image.paste(image, ((w - nw) // 2, (h - nh) // 2))

        label = label.resize((nw, nh), Image.NEAREST)
        new_label = Image.new('L', [w, h], (0))
        new_label.paste(label, ((w - nw) // 2, (h - nh) // 2))
        return new_image, new_label

    # 随机剪裁
    crop_params = T.RandomCrop.get_params(image, input_shape)
    image = F.crop(image, *crop_params)
    label = F.crop(label, *crop_params)
    # 依概率水平翻转
    image = T.RandomHorizontalFlip()(image)
    label = T.RandomHorizontalFlip()(label)
    # 弹性形变
    image = T.ElasticTransform()(image)
    label = T.ElasticTransform()(label)
    # 颜色抖动
    h = random.randint(-hue, hue)
    s = random.randint(max(0, 1-sat), 1+sat)
    v = random.randint(max(0, 1-val), 1+val)
    image = F.adjust_hue(image, h)
    image = F.adjust_saturation(image, s)
    image = F.adjust_brightness(image, v)

    return image, label

