import torch.nn as nn

# 设置类CNNModel，它继承了torch.nn中的Module模块
class CNNModel(nn.Module):
    # 定义卷积神经网络
    # 修改初始化函数init的参数列表
    # 需要将训练图片的高height、宽width、
    # 图片中的字符数量digit_num，类别数量class_num传入
    def __init__(self, height, width, digit_num, class_num):
        super(CNNModel, self).__init__()
        self.digit_num = digit_num # 将digit_num保存在类中

        # 定义第1个卷积层组conv1
        # 其中包括了1个卷积层
        # 1个ReLU激活函数和1个2乘2的最大池化
        self.conv1 = nn.Sequential(
            # 卷积层使用Conv2d定义
            # 包括了1个输入通道，8个输出通道
            # 卷积核的大小是3乘3的
            # 使用padding='same'进行填充
            # 这样可以保证输入和输出的特征图大小相同
            nn.Conv2d(1, 32, kernel_size=3, padding='same'),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.25))

        # 第2个卷积层组，和conv1具有相同的结
        self.conv2 = nn.Sequential(
            # 包括8个输入通道和16个输出通道
            nn.Conv2d(32, 64, kernel_size=3, padding='same'),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.25))

        # 第3个卷积层组，和conv1具有相同的结
        self.conv3 = nn.Sequential(
            # 包括16个输入通道和16个输出通道
            nn.Conv2d(64, 64, kernel_size=3, padding='same'),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.25))

        # 完成三个卷积层的计算后，计算全连接层的输入数据数量input_num
        # 它等于图片的高和宽，分别除以8，再乘以输出特征图的个数16
        # 除以8的原因是，由于经过了3个2*2的最大池化
        # 因此图片的高和宽，都被缩小到原来的1/8
        input_num = (height//8) * (width//8) * 64
        self.fc1 = nn.Sequential(
            nn.Linear(input_num, 1024),
            nn.ReLU(),
            nn.Dropout(0.25))

        # 将输出层的神经元个数设置为class_num
        self.fc2 = nn.Sequential(
            nn.Linear(1024, class_num),
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

        # 使用初始化时传入的digit_num
        # 也就是将模型的最终输出，修改为n*digit_num*字符种类
        out = out.view(out.size(0), self.digit_num, -1)
        return out

import json
if __name__ == '__main__':
    with open("config.json", "r") as f:
        config = json.load(f)

    height = config["resize_height"]  # 图片的高度
    width = config["resize_width"]  # 图片的宽度
    characters = config["characters"]  # 验证码使用的字符集
    digit_num = config["digit_num"]
    class_num = len(characters) * digit_num

    # 定义一个CNNModelUp1实例
    model = CNNModel(height, width, digit_num, class_num)
    print(model) #将其打印，观察打印结果可以了解模型的结构
    print("")


