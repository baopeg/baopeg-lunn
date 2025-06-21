import torch
import torch.nn as nn


class AlexNet(nn.Module):

    """AlexNet模型实现 (2012)
    
    特点:
    - 首次使用ReLU激活函数替代Sigmoid
    - 引入Dropout正则化防止过拟合
    - 使用重叠池化(overlapping pooling)
    - 原始论文使用双GPU并行训练
    
    参数:
        num_classes (int): 输出类别数
    """


    def __init__(self, num_classes=1000):
        super(AlexNet, self).__init__()

        self.features = nn.Sequential(
            # 卷积层1: 96@55x55
            nn.Conv2d(3, 96, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.LocalResponseNorm(size=5, alpha=0.0001, beta=0.75),  # LRN层
            nn.MaxPool2d(kernel_size=3, stride=2),  # 重叠池化
            
            # 卷积层2: 256@27x27
            nn.Conv2d(96, 256, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.LocalResponseNorm(size=5, alpha=0.0001, beta=0.75),
            nn.MaxPool2d(kernel_size=3, stride=2),
            
            # 卷积层3-5: 384@13x13
            nn.Conv2d(256, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),  # 256@6x6
        )

        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),  # Dropout层
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            
            nn.Dropout(p=0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
    
if __name__ == "__main__":
    model = AlexNet(num_classes=10)
    print(model)
    # input = torch.randn(1, 3, 227, 227)  # 输入尺寸: 227x227x3
    # output = model(input)
    # print(f"输出尺寸: {output.size()}")  # 预期: torch.Size([1, 10])