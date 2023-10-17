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
        # 将label_char转换为张量，保存到label中
        label = torch.tensor(self.char2int[label_char],
                             dtype=torch.long)
        return image, label #返回image和label


import matplotlib.pyplot as plt
# show_image函数传入图片数据data和标签label
def show_image(data, label):
    # 将每个小批量数据中的8个数据图片和对应的标签显示出来
    for i in range(len(data)):
        plt.subplot(2, 4, i + 1)
        plt.imshow(data[i].squeeze())
        plt.title(label[i].item())
        plt.axis('off')
    plt.show()
    


















from torch.utils.data import DataLoader
from torchvision import transforms

if __name__ == '__main__':
    # 定义数据转换对象transform
    # 使用transforms.Compose定义数据预处理流水线
    # 在transform添加Resize和ToTensor两个数据处理操作
    transform = transforms.Compose([
        transforms.Resize((128, 128)),  # 将图片缩放到指定的大小
        transforms.ToTensor()])  # 将图片数据转换为张量

    # 定义CaptchaDataset对象dataset
    dataset = CaptchaDataset("./data-sample/", transform)
    # 定义数据加载器data_load
    # 其中参数dataset是数据集
    # batch_size=8代表每个小批量数据的大小是8
    # shuffle = True表示每个epoch，都会随机打乱数据的顺序
    data_load = DataLoader(dataset, batch_size = 8, shuffle = True)

    # 编写一个循环，模拟小批量梯度下降迭代时的数据读取
    # 外层循环，代表了整个训练数据集的迭代轮数，3个epoch就是3轮循环
    # 对于每个epoch，都会遍历全部的训练数据
    for epoch in range(3):
        print("epoch = %d"%(epoch))
        # 内层循环代表了，在一个迭代轮次中，以小批量的方式
        # 使用dataloader对数据进行遍历
        # batch_idx表示当前遍历的批次
        # data和label表示这个批次的训练数据和标记
        for batch_idx, (data, label) in enumerate(data_load):
            print("batch_idx = %d label = %s"%(batch_idx, label))
            # 实现一个show_image函数，用来展示每个小批量的数据
            show_image(data, label)

















