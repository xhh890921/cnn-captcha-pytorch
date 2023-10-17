import torch
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
        # 将fc1输出的128个结果，转换为最终的0到9的10个类别
        # 这10个线性输出结果，还会经过softmax函数的计算，得到10个概率值
        self.fc2 = nn.Sequential(
            nn.Linear(128, 10),
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
        out = self.fc2(out) # [n, 10]
        return out #返回一个n*10的结果

#手动的遍历模型中的各个结构，并计算可以训练的参数
def print_parameters(model):
    cnt = 0
    for name, layer in model.named_children(): #遍历每一层
        # 打印层的名称和该层中包含的可训练参数
        print(f"layer({name}) parameters:")
        for p in layer.parameters():
            print(f'\t {p.shape} has {p.numel()} parameters')
            cnt += p.numel() #将参数数量累加至cnt
    #最后打印模型总参数数量
    print('The model has %d trainable parameters\n' % (cnt))

#打印输入张量x经过每一层时的维度变化情况
def print_forward(model, x):
    # x = [n, 1, 128, 128]
    print(f"x: {x.shape}")
    # 经过第1个卷积层，得到[n, 8, 64, 64]
    x = model.conv1(x)
    print(f"after conv1: {x.shape}")
    # 经过第2个卷积层，得到[n, 16, 32, 32]
    x = model.conv2(x)
    print(f"after conv2: {x.shape}")
    # 经过第3个卷积层，得到[n, 16, 16, 16]
    x = model.conv3(x)
    print(f"after conv3: {x.shape}")
    # 将张量的维度从n*16*16*16转为n*4096
    x = x.view(x.size(0), -1)
    print(f"after view: {x.shape}")
    # 经过第1个线性层，得到[n, 128]
    x = model.fc1(x)
    print(f"after fc1: {x.shape}")
    # 经过第2个线性层，得到[n, 10]
    x = model.fc2(x)
    print(f"after fc2: {x.shape}")


if __name__ == '__main__':
    model = CNNModel() #定义一个CNNModel实例
    print(model) #将其打印，观察打印结果可以了解模型的结构
    print("")

    print_parameters(model) #将模型的参数打印出来

    #打印输入张量x经过每一层维度的变化情况
    x = torch.zeros([5, 1, 128, 128])
    print_forward(model, x)

