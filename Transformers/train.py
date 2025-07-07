import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from model import Transformer  # 导入我们实现的Transformer模型
from config import Config  # 导入配置参数
from data_loader import get_dataloaders  # 导入数据加载器
import time
import matplotlib.pyplot as plt

# 设置随机种子保证可复现性
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

def create_mask(src, tgt, pad_idx):
    """
    创建源序列和目标序列的掩码
    
    参数:
    src: 源序列张量 (batch_size, src_len)
    tgt: 目标序列张量 (batch_size, tgt_len)
    pad_idx: 填充符的索引
    
    返回:
    src_mask: 源序列掩码 (batch_size, 1, 1, src_len)
    tgt_mask: 目标序列掩码 (batch_size, 1, tgt_len, tgt_len)
    """
    # 源序列掩码：忽略填充位置
    src_mask = (src != pad_idx).unsqueeze(1).unsqueeze(2)  # 添加维度 (batch_size, 1, 1, src_len)
    
    # 目标序列填充掩码
    tgt_pad_mask = (tgt != pad_idx).unsqueeze(1).unsqueeze(2)  # (batch_size, 1, 1, tgt_len)
    
    # 目标序列前瞻掩码（防止看到未来信息）
    seq_len = tgt.size(1)
    nopeak_mask = (1 - torch.triu(torch.ones(1, seq_len, seq_len), diagonal=1)).bool()  # 上三角矩阵
    
    # 组合填充掩码和前瞻掩码
    tgt_mask = tgt_pad_mask & nopeak_mask.to(config.device)
    
    return src_mask, tgt_mask

def train_model(model, train_loader, val_loader, criterion, optimizer, config):
    """
    训练Transformer模型
    
    参数:
    model: Transformer模型实例
    train_loader: 训练数据加载器
    val_loader: 验证数据加载器
    criterion: 损失函数
    optimizer: 优化器
    config: 配置对象
    """
    # 记录训练过程中的损失和准确率
    train_losses = []
    val_losses = []
    val_accuracies = []
    
    # 训练开始时间
    start_time = time.time()
    
    # 训练循环
    for epoch in range(config.epochs):
        model.train()  # 设置为训练模式
        epoch_train_loss = 0
        total_samples = 0
        
        # 遍历训练批次
        for i, batch in enumerate(train_loader):
            # 获取源序列和目标序列
            src = batch.src.to(config.device)  # (batch_size, src_len)
            tgt = batch.tgt.to(config.device)  # (batch_size, tgt_len)
            
            # 创建掩码
            src_mask, tgt_mask = create_mask(src, tgt, config.pad_idx)
            
            # 前向传播
            # 输入目标序列的前n-1个token，预测后n-1个token
            outputs = model(src, tgt[:, :-1], src_mask, tgt_mask[:, :, :-1, :-1])
            
            # 计算损失
            # 目标序列从第2个token开始作为标签
            loss = criterion(outputs.reshape(-1, outputs.size(-1)), tgt[:, 1:].reshape(-1))
            
            # 反向传播
            optimizer.zero_grad()  # 清空梯度
            loss.backward()  # 反向计算梯度
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config.max_grad_norm)  # 梯度裁剪
            optimizer.step()  # 更新参数
            
            # 记录损失
            epoch_train_loss += loss.item() * src.size(0)
            total_samples += src.size(0)
            
            # 每100个批次打印一次进度
            if i % 100 == 0:
                print(f"Epoch {epoch+1}/{config.epochs} | Batch {i}/{len(train_loader)} | Loss: {loss.item():.4f}")
        
        # 计算平均训练损失
        avg_train_loss = epoch_train_loss / total_samples
        train_losses.append(avg_train_loss)
        
        # 验证步骤
        avg_val_loss, val_accuracy = validate_model(model, val_loader, criterion, config)
        val_losses.append(avg_val_loss)
        val_accuracies.append(val_accuracy)
        
        # 打印epoch结果
        epoch_time = time.time() - start_time
        print(f"Epoch {epoch+1}/{config.epochs} | "
              f"Train Loss: {avg_train_loss:.4f} | "
              f"Val Loss: {avg_val_loss:.4f} | "
              f"Val Acc: {val_accuracy:.2f}% | "
              f"Time: {epoch_time//60:.0f}m {epoch_time%60:.0f}s")
        
        # 保存模型检查点
        if (epoch + 1) % config.save_interval == 0:
            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss
            }
            torch.save(checkpoint, f"transformer_epoch_{epoch+1}.pt")
            print(f"Saved checkpoint at epoch {epoch+1}")
    
    # 训练结束后保存最终模型
    torch.save(model.state_dict(), "transformer_final.pt")
    print("Training complete. Final model saved.")
    
    # 绘制训练曲线
    plot_training_curves(train_losses, val_losses, val_accuracies)

def validate_model(model, val_loader, criterion, config):
    """
    在验证集上评估模型性能
    
    返回:
    avg_val_loss: 平均验证损失
    accuracy: 验证集准确率(%)
    """
    model.eval()  # 设置为评估模式
    total_loss = 0
    total_correct = 0
    total_tokens = 0
    
    with torch.no_grad():  # 禁用梯度计算
        for batch in val_loader:
            src = batch.src.to(config.device)
            tgt = batch.tgt.to(config.device)
            
            # 创建掩码
            src_mask, tgt_mask = create_mask(src, tgt, config.pad_idx)
            
            # 前向传播
            outputs = model(src, tgt[:, :-1], src_mask, tgt_mask[:, :, :-1, :-1])
            
            # 计算损失
            loss = criterion(outputs.reshape(-1, outputs.size(-1)), tgt[:, 1:].reshape(-1))
            total_loss += loss.item() * src.size(0)
            
            # 计算准确率
            # 获取预测结果中概率最大的索引
            predictions = outputs.argmax(dim=-1)
            # 获取真实标签（忽略填充符）
            labels = tgt[:, 1:]
            # 计算非填充位置上的正确预测数
            non_pad_mask = labels != config.pad_idx
            total_correct += (predictions == labels)[non_pad_mask].sum().item()
            total_tokens += non_pad_mask.sum().item()
    
    # 计算平均损失和准确率
    avg_val_loss = total_loss / len(val_loader.dataset)
    accuracy = (total_correct / total_tokens) * 100 if total_tokens > 0 else 0
    
    return avg_val_loss, accuracy

def plot_training_curves(train_losses, val_losses, val_accuracies):
    """绘制训练过程中的损失和准确率曲线"""
    plt.figure(figsize=(12, 5))
    
    # 绘制损失曲线
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    
    # 绘制准确率曲线
    plt.subplot(1, 2, 2)
    plt.plot(val_accuracies, label='Validation Accuracy', color='green')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title('Validation Accuracy')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('training_curves.png')
    plt.show()

if __name__ == "__main__":
    # 初始化配置
    config = Config()
    print(f"Using device: {config.device}")
    
    # 获取数据加载器
    train_loader, val_loader = get_dataloaders(
        batch_size=config.batch_size
    )
    print(f"Loaded data: {len(train_loader.dataset)} training samples, {len(val_loader.dataset)} validation samples")
    
    # 初始化模型
    model = Transformer(
        src_vocab_size=config.src_vocab_size,
        tgt_vocab_size=config.tgt_vocab_size,
        d_model=config.d_model,
        num_heads=config.num_heads,
        num_layers=config.num_layers,
        d_ff=config.d_ff,
        max_seq_len=config.max_seq_len,
        dropout=config.dropout
    ).to(config.device)
    
    # 打印模型参数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model initialized. Total parameters: {total_params:,} | Trainable parameters: {trainable_params:,}")
    
    # 损失函数（忽略填充符位置）
    criterion = nn.CrossEntropyLoss(ignore_index=config.pad_idx)
    
    # 优化器（Adam优化器，带权重衰减）
    optimizer = optim.Adam(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    
    # 学习率调度器（每epoch衰减学习率）
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=config.lr_decay)
    
    # 开始训练
    print("Starting training...")
    train_model(model, train_loader, val_loader, criterion, optimizer, config)