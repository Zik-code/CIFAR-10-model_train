import torch
import torchvision
from PIL import Image
from torch import nn
from torch.nn import Sequential, Conv2d, MaxPool2d, Flatten, Linear
from torch.utils.tensorboard import SummaryWriter
from model import  *
# 打开图片，使用绝对路径
image_path = r'D:\Desktop\deep_learning\pytorch入门\images\image.png'
image = Image.open(image_path)
print(image.size)

# 保留颜色通道
image = image.convert('RGB')

# 定义图像变换
# 首先转化尺寸，再转化为tensor类型
transform = torchvision.transforms.Compose([
    torchvision.transforms.Resize((32, 32)),
    torchvision.transforms.ToTensor()
])

# 应用变换
image = transform(image)
print(f'变换后的{image.shape}')
# 增加批量维度，以匹配模型输入要求
# 图像数据，常见的输入形状是 (batch_size, channels, height, width)
# 示例说明：假设 image 是一个形状为 (3, 32, 32) 的张量，代表一张 3 通道、高度为 32、宽度为 32 的图像。
# 当调用 image.unsqueeze(0) 后，得到的新张量形状将变为 (1, 3, 32, 32)，
# 这里的 1 就是新插入的批量维度，表示这个批量中只有一张图像。
# unsqueeze 只能插入一个大小为 1 的维度
image = image.unsqueeze(0)
# 或者使用这种方式来更改图像shape
# image = torch.reshape(image, (1, 3, 32, 32))

print(image.shape)



# 实例化模型
model = zzy()

# 加载模型的状态字典
model_path = r'D:\Desktop\deep_learning\model_train\zzy_9.pth'  # 确保路径正确
model.load_state_dict(torch.load(model_path))
model.eval()

# 进行前向传播
# 不要漏掉 with torch.no_grad():
# 我们的目标仅仅是根据输入数据得到模型的预测结果
# 并不需要更新模型的参数，计算梯度是不必要的开销。
with torch.no_grad():
    output = model(image)

# 获取预测的类别
# _,表示一个占位符，只关心另一个值的输出
_, predicted = torch.max(output.data, 1)

# CIFAR-10 数据集的类别名称
classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# 打印预测结果
print(f"预测的类别是: {classes[predicted.item()]}")