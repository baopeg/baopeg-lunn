import torch
import torchvision
from torch import optim,nn
from tqdm import tqdm
from LeNet5 import LeNet5
import argparse

"""
训练函数
参数:
    model: 神经网络模型
    device: 训练设备(CPU/GPU)
    train_loader: 训练数据加载器
    optimizer: 优化器
    criterion: 损失函数
返回:
    平均训练损失
"""
def train(model,device,train_loader,optimizer,criterion):
    model.train()  # 设置为训练模式
    total_loss = 0
    for data,target in tqdm(train_loader):  # 使用tqdm显示进度条
        data,target = data.to(device),target.to(device)  # 数据转移到指定设备
        output= model(data)  # 前向传播
        loss = criterion(output,target)  # 计算损失
        optimizer.zero_grad()  # 梯度清零
        loss.backward()  # 反向传播
        optimizer.step()  # 参数更新
        total_loss += loss.item()

    return total_loss / len(train_loader)  # 返回平均损失

"""
测试函数
参数:
    model: 神经网络模型
    device: 测试设备(CPU/GPU)
    test_loader: 测试数据加载器
返回:
    测试准确率
"""
def test(model,device,test_loader):
    model.eval()  # 设置为评估模式
    correct = 0

    with torch.no_grad():  # 禁用梯度计算
        for data,target in tqdm(test_loader):
            data,target = data.to(device),target.to(device)
            output = model(data)
            pred = output.argmax(dim=1)  # 获取预测类别
            correct += pred.eq(target).sum().item()  # 统计正确预测数

    return correct / len(test_loader.dataset)  # 返回准确率

def main():
    # 参数解析器配置
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs',type=int,default=10, help='训练轮数')
    parser.add_argument('--batch_size',type=int,default=64, help='批大小')
    parser.add_argument('--lr',type=float,default=0.001, help='学习率')
    parser.add_argument('--save_dir',type=str,default='model', help='模型保存目录')
    args = parser.parse_args()

    # 设备配置(自动检测GPU)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 数据预处理管道
    transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize((32,32)),  # 调整图像尺寸为32x32
        torchvision.transforms.ToTensor(),  # 转换为张量
    ])

    # 加载MNIST数据集
    train_set = torchvision.datasets.MNIST(root='./data',train=True,download=True,transform=transform)
    test_set = torchvision.datasets.MNIST(root='./data',train=False,download=True,transform=transform)

    # 创建数据加载器
    train_loader = torch.utils.data.DataLoader(train_set,batch_size=args.batch_size,shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_set,batch_size=args.batch_size)

    # 初始化模型、优化器和损失函数
    model = LeNet5().to(device)  # 实例化LeNet5并转移到设备
    optimizer = optim.Adam(model.parameters(),lr=args.lr)  # Adam优化器
    criterion = nn.CrossEntropyLoss()  # 交叉熵损失函数

    # 训练循环
    for epoch in range(1,args.epochs+1):
        train_loss = train(model,device,train_loader,optimizer,criterion)
        test_acc = test(model,device,test_loader)
        print(f'Epoch {epoch}: Loss={train_loss:.4f}, Accuracy={test_acc:.2%}')

    # 保存训练好的模型
    torch.save(model.state_dict(),f'{args.save_dir}/lenet.pth')

if __name__ == '__main__':
    main()

