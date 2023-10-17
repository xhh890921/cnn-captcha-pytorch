from dataset import CaptchaDataset
from model import CNNModel

import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

import json

if __name__ == '__main__':
    with open("config.json", "r") as f:
        config = json.load(f)

    height = config["resize_height"]  # 图片的高度
    width = config["resize_width"]  # 图片的宽度

    # 定义数据转换对象transform
    # 将图片缩放到指定的大小，并将图片数据转换为张量
    transform = transforms.Compose([
        transforms.Resize((height, width)),
        transforms.ToTensor()])

    test_data_path = config["test_data_path"]  # 训练数据储存路径
    characters = config["characters"]  # 验证码使用的字符集
    digit_num = config["digit_num"]
    class_num = len(characters) * digit_num
    test_model_path = config["test_model_path"]

    print("resize_height = %d" % (height))
    print("resize_width = %d" % (width))
    print("test_data_path = %s" % (test_data_path))
    print("characters = %s" % (characters))
    print("digit_num = %d" % (digit_num))
    print("class_num = %d" % (class_num))
    print("test_model_path = %s" % (test_model_path))
    print("")

    # 使用CaptchaDataset构造测试数据集
    test_data = CaptchaDataset(test_data_path, transform, characters)

    # 使用DataLoader读取test_data
    # 不需要设置任何参数，这样会一个一个数据的读取
    test_loader = DataLoader(test_data)

    # 定义设备对象device，这里如果cuda可用则使用GPU，否则使用CPU
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # 创建一个CNNModel模型对象，并转移到GPU上
    model = CNNModel(height, width, digit_num, class_num).to(device)
    model.eval()
    
    # 调用load_state_dict，读取已经训练好的模型文件captcha.digit
    model.load_state_dict(torch.load(test_model_path))

    right = 0  # 设置right变量，保存预测正确的样本数量
    all = 0  # all保存全部的样本数量
    # 遍历test_loader中的数据
    # x表示样本的特征张量，y表示样本的标签
    for (x, y) in test_loader:
        x, y = x.to(device), y.to(device)  # 转移数据至GPU
        pred = model(x)  # 使用模型预测x的结果，保存在pred中
        # 使用pred.argmax(dim=2).squeeze(0)，获取4位验证码数据的预测结果
        # y.squeeze(0)是4验证码的标记结果
        if torch.equal(pred.argmax(dim=2).squeeze(0),
                       y.squeeze(0)):
            right += 1  # 如果相同，那么right加1
        all += 1  # 每次循环，all变量加1

    # 循环结束后，计算模型的正确率
    acc = right * 1.0 / all
    print("test accuracy = %d / %d = %.3lf" % (right, all, acc))
