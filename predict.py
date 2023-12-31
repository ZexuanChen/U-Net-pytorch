import torch
import copy
import numpy as np
from PIL import Image
import torch.nn.functional as F
import dataloader
from miou import compute_iou
import unet
from unet import UNet
import cv2

input_shape = [512, 512]
num_classes = 2
colors = [(255, 255, 255), (0, 0, 0), (0, 128, 0), (128, 128, 0), (0, 0, 128), (128, 0, 128), (0, 128, 128),
          (128, 128, 128), (64, 0, 0), (192, 0, 0), (64, 128, 0), (192, 128, 0), (64, 0, 128), (192, 0, 128),
          (64, 128, 128), (192, 128, 128), (0, 64, 0), (128, 64, 0), (0, 192, 0), (128, 192, 0), (0, 64, 128),
          (128, 64, 12)]

def detect_image(image, count=False, name_classes=None,miou=False):
    # ---------------------------------------------------#
    #   对输入图像进行一个备份，后面用于绘图
    # ---------------------------------------------------#
    old_img = copy.deepcopy(image)
    orininal_h = np.array(image).shape[0]
    orininal_w = np.array(image).shape[1]
    # ---------------------------------------------------------#
    #   给图像增加灰条，实现不失真的resize
    #   也可以直接resize进行识别
    # ---------------------------------------------------------#
    image_data, nw, nh = resize_image(image, (input_shape[1], input_shape[0]))
    # ---------------------------------------------------------#
    #   添加上batch_size维度
    # ---------------------------------------------------------#
    image_data = np.expand_dims(np.transpose(preprocess_input(np.array(image_data, np.float32)), (2, 0, 1)), 0)

    with torch.no_grad():       #禁用梯度计算，避免不必要的计算和内存消耗
        images = torch.from_numpy(image_data)
        #是否使用cuda
        #images = images.cuda()

        # ---------------------------------------------------#
        #   图片传入网络进行预测
        # ---------------------------------------------------#

        # 创建一个与原始模型结构相同的实例
        net = unet.UNet(in_channels=3, out_channels=2)
        # 加载模型的状态字典
        net.load_state_dict(torch.load("model/model_dice_99.pth", map_location=torch.device('cpu')))
        # 将模型设置为评估模式（推理模式）
        net.eval()
        pr = net(images)[0]

        # ---------------------------------------------------#
        #   取出每一个像素点的种类
        # ---------------------------------------------------#

        #pr = F.softmax(pr.permute(1, 2, 0), dim=-1).cpu().numpy()
        # 因为更改了输出格式，通道数放在最后，Pr不需要维度转换
        pr = F.softmax(pr, dim=-1).cpu().numpy()

        # --------------------------------------#
        #   将灰条部分截取掉
        # --------------------------------------#
        pr = pr[int((input_shape[0] - nh) // 2): int((input_shape[0] - nh) // 2 + nh), \
             int((input_shape[1] - nw) // 2): int((input_shape[1] - nw) // 2 + nw)]
        # ---------------------------------------------------#
        #   进行图片的resize
        # ---------------------------------------------------#
        pr = cv2.resize(pr, (orininal_w, orininal_h), interpolation=cv2.INTER_LINEAR)
        # ---------------------------------------------------#
        #   取出每一个像素点的种类
        # ---------------------------------------------------#
        pr = pr.argmax(axis=-1)

    # ---------------------------------------------------------#
    #   计数
    # ---------------------------------------------------------#
    if count:
        classes_nums = np.zeros([num_classes])
        total_points_num = orininal_h * orininal_w
        print('-' * 63)
        print("|%25s | %15s | %15s|" % ("Key", "Value", "Ratio"))
        print('-' * 63)
        for i in range(num_classes):
            num = np.sum(pr == i)
            ratio = num / total_points_num * 100
            if num > 0:
                print("|%25s | %15s | %14.2f%%|" % (str(name_classes[i]), str(num), ratio))
                print('-' * 63)
            classes_nums[i] = num
        print("classes_nums:", classes_nums)

    seg_img = np.reshape(np.array(colors, np.uint8)[np.reshape(pr, [-1])], [orininal_h, orininal_w, -1])
    # ------------------------------------------------#
    #   将新图片转换成Image的形式
    # ------------------------------------------------#
    image = Image.fromarray(np.uint8(seg_img))
    # ------------------------------------------------#
    #   将新图与原图及进行混合
    # ------------------------------------------------#
    # image = Image.blend(old_img, image, 0.7)

    if miou:
        true_image=input("True Mask filename:")
        true_image=Image.open(true_image)
        _,true_label=dataloader.augment(image,true_image,(512,512),work=False)
        true_label=np.array(true_label)
        true_label[true_label <= 127.5] = 1  # 本来像素接近 0 （黑色）的部分标签为 1 ，即需要切割的部分
        true_label[true_label >= 127.5] = 0  # 本来像素接近 255 （白色）的部分标签作为 0
        miou=compute_iou(pr,true_label,2)
        print(f"miou:{miou}")

    return image

#---------------------------------------------------#
#   对输入图像进行resize
#---------------------------------------------------#
def resize_image(image, size):
    iw, ih  = image.size
    w, h    = size

    scale   = min(w/iw, h/ih)
    nw      = int(iw*scale)
    nh      = int(ih*scale)

    image   = image.resize((nw,nh), Image.BICUBIC)
    new_image = Image.new('RGB', size, (128,128,128))
    new_image.paste(image, ((w-nw)//2, (h-nh)//2))

    return new_image, nw, nh


def preprocess_input(image):
    image /= 255.0
    return image


if __name__ == "__main__" :
    while True:
        img = input('Input image filename:')
        try:
            image = Image.open(img)
        except:
            print('Open Error! Try again!')
            continue
        else:
            r_image = detect_image(image, count=False, name_classes=None,miou=True)
            r_image.show()
