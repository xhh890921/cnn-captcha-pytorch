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
        # 不需要通过颜色来判断验证码中的字符，
		# 转为灰色后，可以提升模型的鲁棒性
        image = Image.open(file_path).convert('L')
        # 使用transform转换数据，将图片数据转为张量数据
        image = self.transform(image)
        # 获取该数据图片中的字符标签
        label_char = os.path.basename(file_path).split('_')[0]
        # 将label_char转换为张量，保存到label中
        label = torch.tensor(self.char2int[label_char],
                             dtype=torch.long)
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

from torch.utils.data import DataLoader
from torchvision import transforms
from torch import optim

if __name__ == '__main__':
    # 定义数据转换对象transform
    # 将图片缩放到指定的大小，并将图片数据转换为张量
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor()])

    # 定义CaptchaDataset对象train_data
    train_data = CaptchaDataset("./data/", transform)
    # 使用DataLoader，定义数据加载器train_load
    # 其中参数train_data是训练集
    # batch_size=64代表每个小批量数据的大小是64
    # shuffle = True表示每一轮训练，都会随机打乱数据的顺序
    train_load = DataLoader(train_data,
                            batch_size = 64,
                            shuffle = True)
    # 训练集有3000个数据，由于每个小批量大小是64，
    # 3000个数据就会分成47个小批量，前46个小批量包括64个数据，
    # 最后一个小批量包括56个数据。46*64+56=3000

    # 定义设备对象device，这里如果cuda可用则使用GPU，否则使用CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 创建一个CNNModel模型对象，并转移到GPU上
    model = CNNModel().to(device)
    optimizer = optim.Adam(model.parameters())  # 创建一个Adam优化器
    criterion = nn.CrossEntropyLoss()  # 创建一个交叉熵损失函数

    # 进入模型的循环迭代
    for epoch in range(50):  # 外层循环，代表了整个训练数据集的遍历次数
        # 内层循环代表了，在一个epoch中，以批量的方式，使用train_load对于数据进行遍历
        # batch_idx 表示当前遍历的批次
        # (data, label) 表示这个批次的训练数据和标记。
        for batch_idx, (data, label) in enumerate(train_load):
            # 将数据data和标签label转移到GPU上
            data, label = data.to(device), label.to(device)

            # 使用当前的模型，预测训练数据data，结果保存在output中
            output = model(data)

            # 调用criterion，计算预测值output与真实值label之间的损失loss
            loss = criterion(output, label)
            loss.backward()  # 计算损失函数关于模型参数的梯度
            optimizer.step()  # 更新模型参数
            optimizer.zero_grad()  # 将梯度清零，以便于下一次迭代

            # 对于每个epoch，每训练10个batch，打印一次当前的损失
            if batch_idx % 10 == 0:
                print(f"Epoch {epoch + 1}/50 "
                      f"| Batch {batch_idx}/{len(train_load)} "
                      f"| Loss: {loss.item():.4f}")

    # 将训练好的模型保存为文件，文件名是captcha.digit
    torch.save(model.state_dict(), 'captcha.digit')
