import torch
from torch import nn

# 模型定义
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
            nn.Linear(64,10)
        )

    def forward(self, x):
        x = self.model(x)
        return x

# 模型前向传播验证
if __name__ == '__main__':
    zzy1 = zzy()
    data = torch.ones((64,3,32,32))
    output = zzy1(data)
    print(output.shape)
