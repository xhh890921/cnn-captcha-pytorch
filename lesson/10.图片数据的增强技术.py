# 导入torchvision中的transforms库，它提供了一系列图像预处理的方法
import torchvision.transforms as transforms
# 导入图像处理库Image，用于读取图片
from PIL import Image

image_path = "dog.jpg" #设置原始图片路径image_path
image = Image.open(image_path) #使用Image.open打开图片
print("original image:")
print("height = %d width = %d"%(image.height, image.width))

# 使用transforms.RandomRotation旋转图片
# 其中degrees=30表示，将图像在-30到+30度的范围内，进行随机旋转
rotate_transform = transforms.RandomRotation(degrees=30)
# 在每次应用该方法时，图像有100%的几率被旋转
# 但旋转的角度是随机的，可能是-30到30之间的任何值
# 完成旋转后，将结果保存下来
new_image = rotate_transform(image)
new_image.save("dog_rotated.jpg")
print("RandomRotation:")
print("height = %d width = %d"%(new_image.height, new_image.width))

# 使用RandomHorizontalFlip水平翻转图片
# 参数p指定了图像被水平翻转的概率
# 如果p=1，表示图像有100%的几率，被水平翻转
# 如果p=0，表示图像不会被水平翻转
# p会被设置为0和1之间的值，例如p=0.5，表示了图像有50%的几率被水平翻转
# 这里为了展示水平翻转效果，所以将p设置为1
horizontal_flip = transforms.RandomHorizontalFlip(p=1)
new_image = horizontal_flip(image)
new_image.save("dog_flipped_horizontal.jpg")
print("RandomHorizontalFlip:")
print("height = %d width = %d"%(new_image.height, new_image.width))

# 使用RandomVerticalFlip垂直翻转图片
# 该函数参数p的使用方法与水平翻转图片是一样的
vertical_flip = transforms.RandomVerticalFlip(p=1)
new_image = vertical_flip(image)
new_image.save("dog_flipped_vertical.jpg")
print("RandomVerticalFlip:")
print("height = %d width = %d"%(new_image.height, new_image.width))

# 使用transforms.RandomCrop对图片进行裁剪，函数传入裁剪的目标大小
# 调用后，会从原始图像中，随机的裁剪出一个目标区域
crop_height, crop_width = 300, 300
# 传入300和300，如果原图像尺寸大于300*300，会从原图像裁剪出300*300的大小
# 如果原始图像尺寸小于300*300，则会出现错误
# 因此我们需要确保图像的原始尺寸大于或等于所需的裁剪尺寸
crop_transform = transforms.RandomCrop((crop_height, crop_width))
new_image = crop_transform(image)
new_image.save("dog_cropped.jpg")
print("RandomCrop:")
print("height = %d width = %d"%(new_image.height, new_image.width))

# 使用transforms.Resize进行图片的缩放
# 函数传入图片被缩放后的尺寸new_height和new_weight
new_height, new_width = 250, 250
# 该函数既可以将图片放大，也可以将图片缩小
# 我们一般会使用该方法，将不同大小的训练数据，统一为相同的大小
resize_transform = transforms.Resize((new_height, new_width))
new_image = resize_transform(image)
new_image.save("dog_resized.jpg")
print("Resize:")
print("height = %d width = %d"%(new_image.height, new_image.width))

