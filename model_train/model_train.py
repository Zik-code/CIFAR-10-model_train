import time
import torch
import torchvision.transforms
from torch.utils.tensorboard import SummaryWriter
from torch import nn
from torch.utils.data import DataLoader

# 下载并加载 CIFAR-10 训练数据集
train_data = torchvision.datasets.CIFAR10(root=r'D:\Desktop\数据集', train=True, transform=torchvision.transforms.ToTensor(), download=True)
# 下载并加载 CIFAR-10 测试数据集
test_data = torchvision.datasets.CIFAR10(root=r'D:\Desktop\数据集', train=False, transform=torchvision.transforms.ToTensor(), download=True)

# 计算训练集和测试集的长度
train_data_size = len(train_data)
test_data_size = len(test_data)
print(f"训练数据集的长度为: {train_data_size}")
print(f"测试数据集的长度为: {test_data_size}")

# 加载数据
train_dataloader = DataLoader(train_data, batch_size=64)
test_dataloader = DataLoader(test_data, batch_size=64)

# 创建网络模型
class zzy(nn.Module):
    def __init__(self):
        super(zzy, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 32, 5, 1, 2),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 32, 5, 1, 2),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 5, 1, 2),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(64 * 4 * 4, 64),
            nn.Linear(64, 10)
        )

    def forward(self, x):
        x = self.model(x)
        return x

# 实例化自定义的网络模型 zzy
zzy1 = zzy()

# 定义损失函数
loss_fn = nn.CrossEntropyLoss()

# 优化器创建
learning_rate = 1e-2
optimier = torch.optim.SGD(zzy1.parameters(), lr=learning_rate)

# 设置训练网络参数
total_train_step = 0
total_test_step = 0
epoch = 100

# 添加 TensorBoard 用于可视化训练过程
writer = SummaryWriter('../logs_train')

start_time = time.time()
# 开始训练循环，共训练 epoch 轮
for i in range(epoch):
    print(f'--------第{i + 1}次训练--------')
    # 遍历训练数据加载器中的每个批次
    for data in train_dataloader:
        # 从数据批次中解包图像和对应的标签
        imgs, target = data
        # 将图像输入到模型中进行前向传播，得到模型的输出
        outputs = zzy1(imgs)
        # 计算模型输出与真实标签之间的损失
        loss = loss_fn(outputs, target)

        # 梯度清零，防止梯度累积
        optimier.zero_grad()
        # 反向传播，计算梯度
        loss.backward()
        # 根据计算得到的梯度更新模型的参数
        optimier.step()

        # 训练步数加 1
        total_train_step += 1
        # 每训练 100 步，打印一次训练信息并将训练损失写入 TensorBoard
        if total_train_step % 100 == 0:
            print(f'训练次数：{total_train_step}，Loss:{loss.item()}')
            # 将训练损失添加到 TensorBoard 中，用于后续可视化
            writer.add_scalar('train_loss', loss.item(), total_train_step)

    # 设置模型为评估模式
    zzy1.eval()
    # 测试步骤开始
    total_test_loss = 0
    total_accuracy = 0
    with torch.no_grad():
        for data in test_dataloader:
            imgs, targets = data
            outputs = zzy1(imgs)
            loss = loss_fn(outputs, targets)
            accuracy = (outputs.argmax(1) == targets).sum()
            total_test_loss += loss.item()
            total_accuracy += accuracy

    print(f'整体的测试损失: {total_test_loss}')
    print(f'整体的正确率: {total_accuracy / test_data_size}')
    # 将测试损失添加到 TensorBoard 中，用于后续可视化
    writer.add_scalar('test_loss', total_test_loss, total_test_step)
    writer.add_scalar('total_accuracy', total_accuracy / test_data_size, total_test_step)
    # 测试步数加 1
    total_test_step += 1

    # 保存模型状态字典
    torch.save(zzy1.state_dict(), f'zzy_{i}.pth')
    print("模型已保存")

end_time = time.time()

print(f'整个训练过程所用时间{end_time-start_time}')
# 关闭 SummaryWriter，释放资源
writer.close()