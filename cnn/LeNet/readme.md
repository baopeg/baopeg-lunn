## LeNet-5 手写数字识别项目

### 项目概述
LeNet-5是由Yann LeCun提出的经典卷积神经网络，本实现包含：
- 原始论文网络结构复现
- MNIST数据集训练流程
- 模型保存与加载功能
- 测试准确率评估

### 环境依赖
```text
Python >= 3.6
torch == 2.0.0
torchvision == 0.15.0
tqdm (进度条显示)
```

### 快速开始
1. 安装依赖：
```bash
pip install -r requirements.txt
```

2. 训练模型：
```bash
python train.py --epochs 10 --batch_size 64
```

3. 测试模型：
```bash
python train.py --mode test --model_path model.pth
```

### 文件结构
```text
神经网络学习/
├── train.py        # 训练测试主程序
├── lenet.py        # 网络结构定义
├── utils.py        # 数据预处理工具
└── model/          # 训练好的模型存储
```

### 参数说明
| 参数 | 说明 |
|------|------|
| --epochs | 训练轮数 (默认10) |
| --lr | 学习率 (默认0.001) |
| --save_dir | 模型保存路径 |
