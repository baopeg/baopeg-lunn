"""
LeNet-5 经典卷积神经网络实现
@since v1.0
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class LeNet5(nn.Module):
    """
    LeNet-5 网络结构
    输入: 1x32x32 灰度图像
    输出: 10类分类结果
    """
    def __init__(self):
        super(LeNet5, self).__init__()
        # 卷积层 C1
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5)
        # 池化层 S2
        self.pool1 = nn.AvgPool2d(kernel_size=2, stride=2)
        # 卷积层 C3
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        # 池化层 S4
        self.pool2 = nn.AvgPool2d(kernel_size=2, stride=2)
        # 全连接层 F5
        self.fc1 = nn.Linear(16*5*5, 120)
        # 全连接层 F6
        self.fc2 = nn.Linear(120, 84)
        # 输出层
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        # 特征提取
        x = F.tanh(self.conv1(x))
        x = self.pool1(x)
        x = F.tanh(self.conv2(x))
        x = self.pool2(x)
        # 分类器
        x = x.view(-1, 16*5*5)
        x = F.tanh(self.fc1(x))
        x = F.tanh(self.fc2(x))
        x = self.fc3(x)
        return x

# 示例用法
if __name__ == "__main__":
    net = LeNet5()
    input = torch.randn(1, 1, 32, 32)
    output = net(input)
    print(f"网络输出形状: {output.shape}")