import math
import torch
import torch.nn as nn

class PositionalEncoding(nn.Module):
    """
    位置编码层：为输入序列添加位置信息
    
    参数:
    d_model (int): 词嵌入的维度（通常512）
    max_len (int): 最大序列长度（默认5000）
    
    用途:
    - 使用正弦/余弦函数生成位置编码
    - 帮助模型理解序列中词的位置关系
    """
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        # 初始化位置编码矩阵 (max_len, d_model)
        pe = torch.zeros(max_len, d_model)
        
        # 生成位置序列 [0, 1, 2, ..., max_len-1]
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        
        # 计算角度变化率 (d_model/2)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        # 偶数位置使用正弦函数
        pe[:, 0::2] = torch.sin(position * div_term)
        # 奇数位置使用余弦函数
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # 增加batch维度 (1, max_len, d_model)
        pe = pe.unsqueeze(0)
        
        # 注册为缓冲区（不参与训练）
        self.register_buffer('pe', pe)

    def forward(self, x):
        """添加位置编码到输入张量"""
        # x.shape: (batch_size, seq_len, d_model)
        # 截取与输入序列长度匹配的位置编码
        return x + self.pe[:, :x.size(1)]

class MultiHeadAttention(nn.Module):
    """
    多头注意力机制
    
    参数:
    d_model (int): 输入维度（通常512）
    num_heads (int): 注意力头数（通常8）
    
    用途:
    - 将输入分割为多个头并行计算注意力
    - 增强模型捕捉不同位置关系的能力
    """
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        # 确保d_model能被num_heads整除
        assert d_model % num_heads == 0, "d_model必须能被num_heads整除"
        self.head_dim = d_model // num_heads  # 每个头的维度（通常64）
        
        # 线性变换层（Q/K/V）
        self.wq = nn.Linear(d_model, d_model)  # 查询向量变换
        self.wk = nn.Linear(d_model, d_model)  # 键向量变换
        self.wv = nn.Linear(d_model, d_model)  # 值向量变换
        self.wo = nn.Linear(d_model, d_model)  # 输出变换
        
    def forward(self, q, k, v, mask=None):
        batch_size = q.size(0)  # 获取batch大小
        
        # 线性变换并重塑为多头结构
        # Q/K/V.shape: (batch_size, seq_len, d_model) -> 
        # (batch_size, num_heads, seq_len, head_dim)
        Q = self.wq(q).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.wk(k).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.wv(v).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        
        # 计算缩放点积注意力分数
        # attn_scores.shape: (batch_size, num_heads, seq_len_q, seq_len_k)
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        # 应用掩码（用于填充位置或未来位置）
        if mask is not None:
            # 将掩码位置的值设为极小数，softmax后接近0
            attn_scores = attn_scores.masked_fill(mask == 0, -1e9)
            
        # 计算注意力权重
        attn_probs = torch.softmax(attn_scores, dim=-1)
        
        # 注意力加权求和
        output = torch.matmul(attn_probs, V)  # (batch_size, num_heads, seq_len, head_dim)
        
        # 拼接多头输出
        # 转置回 (batch_size, seq_len, num_heads, head_dim)
        output = output.transpose(1, 2).contiguous()
        # 重塑为 (batch_size, seq_len, d_model)
        output = output.view(batch_size, -1, self.d_model)
        
        # 最终线性变换
        return self.wo(output)

class FeedForward(nn.Module):
    """
    前馈神经网络
    
    参数:
    d_model (int): 输入/输出维度（通常512）
    d_ff (int): 隐藏层维度（通常2048）
    
    用途:
    - 提供非线性变换能力
    - 增强模型表示能力
    """
    def __init__(self, d_model, d_ff=2048):
        super().__init__()
        # 两层线性变换
        self.linear1 = nn.Linear(d_model, d_ff)  # 扩展维度
        self.linear2 = nn.Linear(d_ff, d_model)  # 恢复维度
        self.relu = nn.ReLU()  # 激活函数

    def forward(self, x):
        # ReLU激活提供非线性
        return self.linear2(self.relu(self.linear1(x)))

class EncoderLayer(nn.Module):
    """
    Transformer编码器层
    
    参数:
    d_model (int): 输入维度（通常512）
    num_heads (int): 注意力头数（通常8）
    d_ff (int): 前馈网络隐藏层维度（通常2048）
    dropout (float): Dropout概率（默认0.1）
    
    结构:
    1. 多头自注意力
    2. 残差连接 + 层归一化
    3. 前馈网络
    4. 残差连接 + 层归一化
    """
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        # 多头自注意力层
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        # 前馈网络层
        self.ffn = FeedForward(d_model, d_ff)
        
        # 层归一化
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        # Dropout层
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask):
        # 自注意力计算
        attn_output = self.self_attn(x, x, x, mask)
        # 残差连接 + Dropout + 层归一化
        x = self.norm1(x + self.dropout(attn_output))
        
        # 前馈网络计算
        ffn_output = self.ffn(x)
        # 残差连接 + Dropout + 层归一化
        return self.norm2(x + self.dropout(ffn_output))

class DecoderLayer(nn.Module):
    """
    Transformer解码器层
    
    参数:
    d_model (int): 输入维度（通常512）
    num_heads (int): 注意力头数（通常8）
    d_ff (int): 前馈网络隐藏层维度（通常2048）
    dropout (float): Dropout概率（默认0.1）
    
    结构:
    1. 带掩码的多头自注意力
    2. 残差连接 + 层归一化
    3. 编码器-解码器注意力
    4. 残差连接 + 层归一化
    5. 前馈网络
    6. 残差连接 + 层归一化
    """
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        # 掩码自注意力层
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        # 编码器-解码器注意力层
        self.cross_attn = MultiHeadAttention(d_model, num_heads)
        # 前馈网络层
        self.ffn = FeedForward(d_model, d_ff)
        
        # 三层归一化
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        # Dropout层
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, enc_output, src_mask, tgt_mask):
        # 自注意力（带目标序列掩码）
        attn_output = self.self_attn(x, x, x, tgt_mask)
        # 残差连接 + Dropout + 层归一化
        x = self.norm1(x + self.dropout(attn_output))
        
        # 编码器-解码器注意力
        # 查询：解码器输出，键/值：编码器输出
        attn_output = self.cross_attn(x, enc_output, enc_output, src_mask)
        # 残差连接 + Dropout + 层归一化
        x = self.norm2(x + self.dropout(attn_output))
        
        # 前馈网络
        ffn_output = self.ffn(x)
        # 残差连接 + Dropout + 层归一化
        return self.norm3(x + self.dropout(ffn_output))

class Transformer(nn.Module):
    """
    完整的Transformer模型
    
    参数:
    src_vocab_size (int): 源语言词汇表大小
    tgt_vocab_size (int): 目标语言词汇表大小
    d_model (int): 模型维度（通常512）
    num_heads (int): 注意力头数（通常8）
    num_layers (int): 编码器/解码器层数（通常6）
    d_ff (int): 前馈网络隐藏层维度（通常2048）
    max_seq_len (int): 最大序列长度（默认100）
    dropout (float): Dropout概率（默认0.1）
    
    结构:
    1. 词嵌入层
    2. 位置编码
    3. 编码器堆叠
    4. 解码器堆叠
    5. 输出线性层
    """
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model, num_heads, num_layers, d_ff, max_seq_len, dropout=0.1):
        super().__init__()
        # 源语言词嵌入
        self.encoder_embed = nn.Embedding(src_vocab_size, d_model)
        # 目标语言词嵌入
        self.decoder_embed = nn.Embedding(tgt_vocab_size, d_model)
        # 位置编码层
        self.pos_encoding = PositionalEncoding(d_model, max_seq_len)
        
        # 创建编码器层堆叠
        self.encoder_layers = nn.ModuleList([
            EncoderLayer(d_model, num_heads, d_ff, dropout) 
            for _ in range(num_layers)
        ])
        
        # 创建解码器层堆叠
        self.decoder_layers = nn.ModuleList([
            DecoderLayer(d_model, num_heads, d_ff, dropout) 
            for _ in range(num_layers)
        ])
        
        # 输出线性层（预测目标语言词汇）
        self.fc_out = nn.Linear(d_model, tgt_vocab_size)
        # Dropout层
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, src, tgt, src_mask, tgt_mask):
        """
        前向传播
        
        输入:
        src: 源序列 (batch_size, src_len)
        tgt: 目标序列 (batch_size, tgt_len)
        src_mask: 源序列掩码 (batch_size, 1, 1, src_len)
        tgt_mask: 目标序列掩码 (batch_size, 1, tgt_len, tgt_len)
        
        输出:
        目标序列的预测概率 (batch_size, tgt_len, tgt_vocab_size)
        """
        # 编码器处理
        # 词嵌入 + 位置编码 + Dropout
        src_emb = self.dropout(self.pos_encoding(self.encoder_embed(src)))
        enc_output = src_emb
        # 逐层处理编码器
        for layer in self.encoder_layers:
            enc_output = layer(enc_output, src_mask)
        
        # 解码器处理
        # 词嵌入 + 位置编码 + Dropout
        tgt_emb = self.dropout(self.pos_encoding(self.decoder_embed(tgt)))
        dec_output = tgt_emb
        # 逐层处理解码器
        for layer in self.decoder_layers:
            dec_output = layer(dec_output, enc_output, src_mask, tgt_mask)
        
        # 输出层（预测目标词概率）
        return self.fc_out(dec_output)