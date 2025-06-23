import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm
from ResNet import ResNet18

# 1. 数据准备
def prepare_data(batch_size=32):
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    train_data = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    val_data = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size)
    return train_loader, val_loader

# 2. 训练函数
def train_model(model, train_loader, val_loader, epochs=10, lr=0.001):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    for epoch in range(epochs):
        # 训练阶段
        model.train()
        train_loss, correct = 0, 0
        for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            correct += (outputs.argmax(1) == labels).sum().item()
        
        # 验证阶段
        model.eval()
        val_loss, val_correct = 0, 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                val_loss += criterion(outputs, labels).item()
                val_correct += (outputs.argmax(1) == labels).sum().item()
        
        # 打印统计信息
        print(f"\nTrain Loss: {train_loss/len(train_loader):.4f} | "
              f"Train Acc: {correct/len(train_data)*100:.2f}%")
        print(f"Val Loss: {val_loss/len(val_loader):.4f} | "
              f"Val Acc: {val_correct/len(val_data)*100:.2f}%")

if __name__ == "__main__":
    # 初始化模型和数据
    model = ResNet18(num_classes=10)
    train_loader, val_loader = prepare_data()
    
    # 开始训练
    train_model(model, train_loader, val_loader, epochs=10)