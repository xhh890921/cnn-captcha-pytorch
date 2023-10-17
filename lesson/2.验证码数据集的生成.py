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

if __name__ == '__main__':
    # 保存验证码字符
    chars = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
             'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j',
             'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't',
             'u', 'v', 'w', 'x', 'y', 'z',
             'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J',
             'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T',
             'U', 'V', 'W', 'X', 'Y', 'Z']
    num = 100 #待生成的验证码图片数量
    count = 4 #验证码图片中的字符数
    path = "./data/" #结果路径
    width = 200  # 验证码图片宽度
    height = 100  # 验证码图片高度
    # 调用generate_data，生成验证码数据
    generate_data(num, count, chars, path, width, height)















