# 导入验证码模块ImageCaptcha和随机数模块random
from captcha.image import ImageCaptcha
import random

# 定义函数generate_data，用于生成验证码图片
# num是需要生成的验证码图片数量
# count是验证码图中包含的字符数量
# chars保存验证码中包含的字符
# path是图片结果的保存路径
# width是height是图片的宽和高
def generate_data(num, count, chars, path, width, height):
    # 使用变量i，循环生成num个验证码图片
    for i in range(num):
        # 打印当前的验证码编号
        print("generate %d"%(i))
        # 使用ImageCaptcha，创建验证码生成器generator
        generator = ImageCaptcha(width=width, height=height)
        random_str = "" #保存验证码图片上的字符
        # 向random_str中，循环添加count个字符
        for j in range(count):
            # 每个字符，使用random.choice，随机的从chars中选择
            choose = random.choice(chars)
            random_str += choose
        # 调用generate_image，生成验证码图片img
        img = generator.generate_image(random_str)
        # 在验证码上加干扰点
        generator.create_noise_dots(img, '#000000', 4, 40)
        # 在验证码上加干扰线
        generator.create_noise_curve(img, '#000000')
        # 设置文件名，命名规则为，验证码字符串random_str，加下划线，加数据编号
        file_name = path + random_str + '_' + str(i) + '.jpg'
        img.save(file_name) # 保存文件

import json
import os

if __name__ == '__main__':
    # 使用open函数，打开config.json配置文件
    with open("config.json", "r") as f:
        # 使用json.load读取解析json，结果保存在config
        config = json.load(f)

    # 接着从配置中获取各项参数
    # 具体使用config加中括号中括号中为参数名，这样的方式读取配置内容
    train_data_path = config["train_data_path"]  # 训练数据路径
    test_data_path = config["test_data_path"]  # 测试数据路径

    train_num = config["train_num"]  # 训练样本个数
    test_num = config["test_num"] # 测试样本个数

    characters = config["characters"]  # 验证码使用的字符集
    digit_num = config["digit_num"]  # 图片上的字符数量
    img_width = config["img_width"]  # 图片的宽度
    img_height = config["img_height"]  # 图片的高度


    # 检查数据路径上的文件夹是否存在
    # 如果不存在，则创建保存数据的文件夹
    if not os.path.exists(train_data_path):
        os.makedirs(train_data_path)
    if not os.path.exists(test_data_path):
        os.makedirs(test_data_path)

    # 调用generate_data，生成训练数据
    generate_data(train_num, digit_num, characters,
                  train_data_path, img_width, img_height)
    # 调用generate_data，生成测试数据
    generate_data(test_num, digit_num, characters,
                  test_data_path, img_width, img_height)














