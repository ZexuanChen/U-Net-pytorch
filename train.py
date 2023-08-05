import os
import datetime
import numpy as np
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim as optim
from torch.utils.data import DataLoader
import unet
import dataloader
import time

load = bool(True)
dataset_path=''
num_epochs=30
device="cpu"
shuffle=bool(True)
path="GlandCeildata/train"
origin_model="model/model_dice_99.pth"

if __name__ == "__main__":

    # out_chanels填什么？
    model = unet.UNet(3,2).train().to(device)
    #name 是什么？ path
    annotation=[str(i) for i in range(160)]
    dataset=dataloader.MedicalDataSet(annotation,(512,512),1,bool(True),path)
    train_loader=DataLoader(dataset,batch_size=1,shuffle=shuffle)

    # 预训练模块
    # 源代码如果不预训练会进行权重初始分配
    if load:
        # 创建一个与原始模型结构相同的实例
        model = unet.UNet(in_channels=3, out_channels=2)
        # 加载模型的状态字典
        model.load_state_dict(torch.load(origin_model, map_location=torch.device(device)))
        # Step 2: 将模型参数转换到CPU上
        model.to(device)


    # with open(os.path.join(dataset_path, "ImageSets/Segmentation/train.txt"),"r") as f:
    #     train_lines = f.readlines()
    # with open(os.path.join(dataset_path, "ImageSets/Segmentation/val.txt"),"r") as f:
    #     val_lines = f.readlines()
    # num_train   = len(train_lines)
    # num_val     = len(val_lines)

    optimizer = optim.Adam(model.parameters(), lr=0.001)


    def dice_loss(outputs, targets):
        smooth = 1e-5
        numerator = 2.0 * torch.sum(outputs * targets)
        denominator = torch.sum(outputs + targets)
        dice = (numerator + smooth) / (denominator + smooth)
        return (1 - dice) * 10000

    # 定义损失函数
    criterion = dice_loss

    # criterion = nn.CrossEntropyLoss().to(device)

    for epoch in range(num_epochs):
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            print(f"Loss: {loss.item():.4f}")

        epoch_loss = running_loss / len(train_loader)
        # 生成带有当前时间的文件名
        # current_time = time.strftime("%Y%m%d%H%M%S", time.localtime())
        if load:
            file_name = f"model/model_conti_{origin_model.split('.')[0]}_{epoch}.pth"
        else :
            file_name=f"model/model_ini_{epoch}.pth"
        # 保存模型(每一个epoch都会记录一次)
        torch.save(model.state_dict(), file_name)

        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss:.4f}")

