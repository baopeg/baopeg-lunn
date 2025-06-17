# AlexNet 实现文档

## 模型概述
AlexNet是2012年ImageNet竞赛冠军模型，由Alex Krizhevsky等人提出。主要创新点包括：
- 首次使用ReLU激活函数
- 使用Dropout防止过拟合
- 引入重叠池化(Overlapping Pooling)
- 使用局部响应归一化(LRN)

## 文件结构
@/LeNet 依据这个文件的模式，我们来学习AlexNet

根据您的要求，我将按照LeNet的模式来组织AlexNet的学习内容。以下是完整的实现方案：
1. 文件结构规划
Apply to readme.md
2. AlexNet模型实现 (cnn/AlexNet/AlexNet.py)
3. 训练脚本 (cnn/AlexNet/train.py)
Apply
4. README文档 (cnn/AlexNet/readme.md)
AlexNet/
├── AlexNet.py # 模型架构实现
├── train.py # 训练脚本
└── readme.md # 本文档

## 训练说明
1. **数据集**：使用CIFAR-10（原始论文使用ImageNet）
2. **输入尺寸**：227×227×3
3. **训练命令**：
   ```bash
   python train.py
   ```
4. **预期性能**：
   - 训练时间：~30分钟（GPU）
   - 测试准确率：>80%（CIFAR-10）

## 关键参数
| 参数 | 值 | 说明 |
|------|----|------|
| 学习率 | 0.01 | SGD优化器 |
| 动量 | 0.9 | SGD优化器 |
| Dropout率 | 0.5 | 全连接层 |
| 批量大小 | 128 | |

## 使用示例
```python
from AlexNet import AlexNet

# 初始化模型
model = AlexNet(num_classes=10)

# 加载预训练权重
model.load_state_dict(torch.load('alexnet_cifar10.pth'))
```

## 注意事项
1. 原始AlexNet输入为224×224，但实际计算需要227×227
2. LRN层在现代网络中已较少使用
3. 训练时建议使用GPU加速

