# 直接导入dataset.py中的CaptchaDataset类
from dataset import CaptchaDataset
# 直接导入model.py中的CNNModel类
from model import CNNModel

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from torch import optim
import json
import os

if __name__ == '__main__':
    # 打开配置文件
    with open("config.json", "r") as f:
        config = json.load(f)

    # 读取resize_height和resize_width两个参数
    # 它们代表图片数据最终缩放的高和宽，用于创建transform
    height = config["resize_height"]  # 图片的高度
    width = config["resize_width"]  # 图片的宽度
    # 定义数据转换对象transform
    # 使用transforms.Compose定义数据预处理流水线
    # 在transform添加Resize和ToTensor两个数据处理操作
    transform = transforms.Compose([
        transforms.RandomRotation(10), # 添加旋转方案
        transforms.Resize((height, width)),  # 将图片缩放到指定的大小
        transforms.ToTensor()])  # 将图片数据转换为张量

    train_data_path = config["train_data_path"]  # 获取训练数据路径
    characters = config["characters"]  # 验证码字符集
    batch_size = config["batch_size"] # 批量大小
    epoch_num = config["epoch_num"] # 迭代轮数
    digit_num = config["digit_num"] # 字符个数
    learning_rate = config["learning_rate"] #迭代速率
    # 计算类别个数class_num，等于使用的字符数量*字符个数
    class_num = len(characters) * digit_num

    model_save_path = config["model_save_path"] #获取模型的保存路径
    model_name = config["model_name"] #模型名称
    model_save_name = model_save_path + "/" + model_name
    # 创建模型文件夹
    if not os.path.exists(model_save_path):
        os.makedirs(model_save_path)

    print("resize_height = %d"%(height))
    print("resize_width = %d" %(width))
    print("train_data_path = %s"%(train_data_path))
    print("characters = %s" % (characters))
    print("batch_size = %d" % (batch_size))
    print("epoch_num = %d" % (epoch_num))
    print("digit_num = %d" % (digit_num))
    print("class_num = %d" % (class_num))
    print("learning_rate = %lf" % (learning_rate))
    print("model_save_name = %s" % (model_save_name))
    print("")

    # 定义CaptchaDataset对象train_data
    train_data = CaptchaDataset(train_data_path, transform, characters)
    # 使用DataLoader，定义数据加载器train_load
    # 其中参数train_data是训练集
    # batch_size=64代表每个小批量数据的大小是64
    # shuffle = True表示每一轮训练，都会随机打乱数据的顺序
    train_load = DataLoader(train_data,
                            batch_size = batch_size,
                            shuffle = True)
    # 训练集有3000个数据，由于每个小批量大小是64，
    # 3000个数据就会分成47个小批量，前46个小批量包括64个数据，
    # 最后一个小批量包括56个数据。46*64+56=3000

    # 定义设备对象device，这里如果cuda可用则使用GPU，否则使用CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
       
    # 创建一个CNNModel模型对象，并转移到GPU上
    model = CNNModel(height, width, digit_num, class_num).to(device)
    model.train()    
    
    # 需要指定迭代速率。默认情况下是0.001，我们将迭代速率修改0.0001
    # 因为面对更复杂的数据，较小的迭代速率可以使迭代更稳定
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()  # 创建一个交叉熵损失函数

    print("Begin training:")
    # 提升迭代轮数，从50轮训练提升至200轮训练
    for epoch in range(epoch_num):  # 外层循环，代表了整个训练数据集的遍历次数
        # 内层循环代表了，在一个epoch中，以批量的方式，使用train_load对于数据进行遍历
        # batch_idx 表示当前遍历的批次
        # (data, label) 表示这个批次的训练数据和标记。
        for batch_idx, (data, label) in enumerate(train_load):
            # 将数据data和标签label转移到GPU上
            data, label = data.to(device), label.to(device)

            # 使用当前的模型，预测训练数据data，结果保存在output中
            output = model(data)

            # 修改损失值loss的计算方法
            # 将4位验证码的每一位的损失，都累加到一起
            loss = torch.tensor(0.0).to(device)
            for i in range(digit_num): #使用i，循环4位验证码
                # 每一位验证码的模型计算输出为output[:, i, :]
                # 标记为label[:, i]
                # 交叉熵损失函数criterion，计算一位验证码的损失
                # 将4位验证码的损失，累加到loss
                loss += criterion(output[:, i, :], label[:, i])

            loss.backward()  # 计算损失函数关于模型参数的梯度
            optimizer.step()  # 更新模型参数
            optimizer.zero_grad()  # 将梯度清零，以便于下一次迭代

            # 计算训练时每个batch的正确率acc
            predicted = torch.argmax(output, dim=2)
            correct = (predicted == label).all(dim=1).sum().item()
            acc = correct / data.size(0)

            # 对于每个epoch，每训练10个batch，打印一次当前的损失
            if batch_idx % 10 == 0:
                print(f"Epoch {epoch + 1}/{epoch_num} "
                      f"| Batch {batch_idx}/{len(train_load)} "
                      f"| Loss: {loss.item():.4f} "
                      f"| accuracy {correct}/{data.size(0)}={acc:.3f}")

        # 每10轮训练，就保存一次checkpoint模型，用来调试使用
        if (epoch + 1) % 10 == 0:
            checkpoint = model_save_path + "/check.epoch" + str(epoch+1)
            torch.save(model.state_dict(), checkpoint)
            print("checkpoint saved : %s" % (checkpoint))

    # 程序的最后，使用配置中的路径，保存训练结果
    torch.save(model.state_dict(), model_save_name)
    print("model saved : %s" % (model_save_name))










