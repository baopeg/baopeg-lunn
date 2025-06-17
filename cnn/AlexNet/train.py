import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from tqdm import tqdm
from torch.utils.data import DataLoader
from AlexNet import AlexNet
import time

# 设备配置
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")

# 数据预处理
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(227),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 加载CIFAR-10数据集 (ImageNet太大，使用CIFAR-10替代)
trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

trainloader = DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)
testloader = DataLoader(testset, batch_size=128, shuffle=False, num_workers=2)

# 初始化模型
model = AlexNet(num_classes=10).to(device)

# 损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=0.0005)

# 训练函数
def train(epoch):
    model.train()
    running_loss = 0.0
    for i, data in tqdm(enumerate(trainloader,0)):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        if i % 100 == 99:
            print(f'[{epoch + 1}, {i + 1}] 损失: {running_loss / 100:.3f}')
            running_loss = 0.0

# 测试函数
def test():
    # 1. 设置模型为评估模式
    model.eval()
    
    # 2. 初始化统计变量
    correct = 0
    total = 0
    
    # 3. 禁用梯度计算
    with torch.no_grad():
        # 4. 遍历测试数据集
        for data in tqdm(testloader):
            # 5. 获取批处理数据
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            
            # 6. 前向传播计算输出
            outputs = model(images)
            
            # 7. 获取预测结果
            _, predicted = torch.max(outputs.data, 1)
            
            # 8. 统计正确预测数
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    # 9. 计算准确率
    accuracy = 100 * correct / total
    print(f'测试准确率: {accuracy:.2f}%')
    return accuracy

# 主训练循环
if __name__ == "__main__":
    start_time = time.time()
    
    for epoch in range(10):
        print(f"Epoch {epoch+1}/10")
        train(epoch)
        accuracy = test()
    
    print(f"训练完成! 总耗时: {time.time() - start_time:.2f}秒")
    
    # 保存模型
    torch.save(model.state_dict(), 'alexnet_cifar10.pth')
    print("模型已保存为 alexnet_cifar10.pth")