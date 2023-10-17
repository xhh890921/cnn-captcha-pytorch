from torch.utils.data import Dataset
from PIL import Image
import torch
import os

# 设置CaptchaDataset继承Dataset，用于读取验证码数据
class CaptchaDataset(Dataset):
    # init函数用于初始化
    # 函数传入数据的路径data_dir和数据转换对象transform
    def __init__(self, data_dir, transform):
        self.file_list = list() #保存每个训练数据的路径
        # 使用os.listdir，获取data_dir中的全部文件
        files = os.listdir(data_dir)
        for file in files: #遍历files
            # 将目录路径与文件名组合为文件路径
            path = os.path.join(data_dir, file)
            # 将path添加到file_list列表
            self.file_list.append(path)
        # 将数据转换对象transform保存到类中
        self.transform = transform

        # 设置chars等于字符0到9
        # 表示验证码图片中可能出现的字符
        chars = '0123456789'
        # 创建一个字符到数字的字典
        self.char2int = {}
        for i, char in enumerate(chars):
            self.char2int[char] = i

    def __len__(self):
        # 直接返回数据集中的样本数量
        # 重写该方法后可以使用len(dataset)语法，来获取数据集的大小
        return len(self.file_list)

    # 函数传入索引index，函数应当返回与该索引对应的数据和标签
    # 通过dataset[i]，就可以获取到第i个样本了
    def __getitem__(self, index):
        file_path = self.file_list[index] #获取数据的路径
        # 打开文件，并使用convert('L')，将图片转换为灰色
        # 不需要通过颜色来判断验证码中的字符，转为灰色后，可以提升模型的鲁棒性
        image = Image.open(file_path).convert('L')
        # 使用transform转换数据，将图片数据转为张量数据
        image = self.transform(image)
        # 获取该数据图片中的字符标签
        label_char = os.path.basename(file_path).split('_')[0]

        # 在获取到该数据图片中的字符标签label_char后
        label = list()
        for char in label_char: # 遍历字符串label_char
            # 将其中的字符转为数字，添加到列表label中
            label.append(self.char2int[char])
        # 将label转为张量，作为训练数据的标签
        label = torch.tensor(label, dtype=torch.long)
        print(label)
        return image, label #返回image和label

import torch.nn as nn

# 设置类CNNModel，它继承了torch.nn中的Module模块
class CNNModel(nn.Module):
    # 定义卷积神经网络
    def __init__(self):
        super(CNNModel, self).__init__()
        # 定义第1个卷积层组conv1
        # 其中包括了1个卷积层
        # 1个ReLU激活函数和1个2乘2的最大池化
        self.conv1 = nn.Sequential(
            # 卷积层使用Conv2d定义
            # 包括了1个输入通道，8个输出通道
            # 卷积核的大小是3乘3的
            # 使用padding='same'进行填充
            # 这样可以保证输入和输出的特征图大小相同
            nn.Conv2d(1, 8, kernel_size=3, padding='same'),
            nn.ReLU(),
            nn.MaxPool2d(2))

        # 第2个卷积层组，和conv1具有相同的结
        self.conv2 = nn.Sequential(
            # 包括8个输入通道和16个输出通道
            nn.Conv2d(8, 16, kernel_size=3, padding='same'),
            nn.ReLU(),
            nn.MaxPool2d(2))

        # 第3个卷积层组，和conv1具有相同的结
        self.conv3 = nn.Sequential(
            # 包括16个输入通道和16个输出通道
            nn.Conv2d(16, 16, kernel_size=3, padding='same'),
            nn.ReLU(),
            nn.MaxPool2d(2))

        # 第1个全连接层
        # 全连接层的输入神经元个数与上一层的输出数据的数量有关
        # 具体为16*16*16 = 4096
        # 第1个全连接层的输入神经元有4096个
        # 这一个全连接层会设置128个输出神经元
        self.fc1 = nn.Sequential(
            nn.Linear(4096, 128),
            nn.ReLU())

        # 第2个全连接层
        # 将模型中的最后一个128*10的线性层，修改为128*40的线性层
        self.fc2 = nn.Sequential(
            nn.Linear(128, 4 * 10),
        )
        # 后面训练会使用交叉熵损失函数CrossEntropyLoss
        # softmax函数会定义在损失函数中，所以这里就不显示的定义了

    # 前向传播函数
    # 函数输入一个四维张量x
    # 这四个维度分别是样本数量、输入通道、图片的高度和宽度
    def forward(self, x): # [n, 1, 128, 128]
        # 将输入张量x按照顺序，输入至每一层中进行计算
        # 每层都会使张量x的维度发生变化
        out = self.conv1(x) # [n, 8, 64, 64]
        out = self.conv2(out) # [n, 16, 32, 32]
        out = self.conv3(out) # [n, 16, 16, 16]
        # 使用view函数，将张量的维度从n*16*16*16转为n*4096
        out = out.view(out.size(0), -1) # [n, 4096]
        out = self.fc1(out) # [n, 128]
        # 经过3个卷积层与2个全连接层后，会计算得到n*40的张量
        out = self.fc2(out) # [n, 40]

        # 将这个n*40的张量，使用view函数
        # 重置为n*4*10的张量，从而代表n个4位的数字验证码结果
        out = out.view(out.size(0), 4, -1)
        return out #返回一个n*4*10的结果

from torch.utils.data import DataLoader
import torchvision.transforms as transforms

if __name__ == '__main__':
    # 定义数据转换对象transform
    # 将图片缩放到指定的大小，并将图片数据转换为张量
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor()])

    # 使用CaptchaDataset构造测试数据集
    test_data = CaptchaDataset("./data-4char-1k/", transform)

    # 使用DataLoader读取test_data
    # 不需要设置任何参数，这样会一个一个数据的读取
    test_loader = DataLoader(test_data)

    # 定义设备对象device，这里如果cuda可用则使用GPU，否则使用CPU
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = CNNModel().to(device)  # 转移模型至GPU
    # 调用load_state_dict，读取已经训练好的模型文件captcha.digit
    model.load_state_dict(torch.load('captcha.4digits.3w'))

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
