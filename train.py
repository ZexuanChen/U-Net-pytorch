import os
import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim as optim
from torch.utils.data import DataLoader

load = bool(False)
dataset_path=''
num_epochs=30
device="cpu"



if __name__ == "__main__":
    model = Unet().train()

    # 预训练模块
    # 源代码如果不预训练会进行权重初始分配
    if load:
        model = torch.load('model.pth')

    with open(os.path.join(dataset_path, "ImageSets/Segmentation/train.txt"),"r") as f:
        train_lines = f.readlines()
    with open(os.path.join(dataset_path, "ImageSets/Segmentation/val.txt"),"r") as f:
        val_lines = f.readlines()
    num_train   = len(train_lines)
    num_val     = len(val_lines)

    optimizer = optim.Adam(model.parameters(), lr=0.001)
    # 定义损失函数
    criterion = nn.CrossEntropyLoss()

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

        epoch_loss = running_loss / len(train_loader)
        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss:.4f}")

    # 保存模型
    torch.save(model.state_dict(), "your_model.pth")
