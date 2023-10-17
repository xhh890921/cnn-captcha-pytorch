from torch.utils.data import Dataset
from PIL import Image
import torch
import os

# 设置CaptchaDataset继承Dataset，用于读取验证码数据
class CaptchaDataset(Dataset):
    # init函数用于初始化
    # 函数传入数据的路径data_dir和数据转换对象transform
    # 将验证码使用的字符集characters，通过参数传入
    def __init__(self, data_dir, transform, characters):
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

        # 创建一个字符到数字的字典
        self.char2int = {}
        # 在创建字符到数字的字典时，使用外界传入的字符集characters
        for i, char in enumerate(characters):
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
        return image, label #返回image和label
        
        
from torch.utils.data import DataLoader
from torchvision import transforms
import json

if __name__ == '__main__':
    with open("config.json", "r") as f:
        config = json.load(f)

    height = config["resize_height"]  # 图片的高度
    width = config["resize_width"]  # 图片的宽度
    # 定义数据转换对象transform
    # 使用transforms.Compose定义数据预处理流水线
    # 在transform添加Resize和ToTensor两个数据处理操作
    transform = transforms.Compose([
        transforms.Resize((height, width)),  # 将图片缩放到指定的大小
        transforms.ToTensor()])  # 将图片数据转换为张量

    data_path = config["train_data_path"]  # 训练数据储存路径
    characters = config["characters"]  # 验证码使用的字符集
    batch_size = config["batch_size"]
    epoch_num = config["epoch_num"]

    # 定义CaptchaDataset对象dataset
    dataset = CaptchaDataset(data_path, transform, characters)
    # 定义数据加载器data_load
    # 其中参数dataset是数据集
    # batch_size=8代表每个小批量数据的大小是8
    # shuffle = True表示每个epoch，都会随机打乱数据的顺序
    data_load = DataLoader(dataset,
                           batch_size = batch_size,
                           shuffle = True)

    # 编写一个循环，模拟小批量梯度下降迭代时的数据读取
    # 外层循环，代表了整个训练数据集的迭代轮数，3个epoch就是3轮循环
    # 对于每个epoch，都会遍历全部的训练数据
    for epoch in range(epoch_num):
        print("epoch = %d"%(epoch))
        # 内层循环代表了，在一个迭代轮次中，以小批量的方式
        # 使用dataloader对数据进行遍历
        # batch_idx表示当前遍历的批次
        # data和label表示这个批次的训练数据和标记
        for batch_idx, (data, label) in enumerate(data_load):
            print("batch_idx = %d label = %s"%(batch_idx, label))
