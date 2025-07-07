import torch
from torchtext.datasets import Multi30k
from torchtext.data import Field, BucketIterator
import spacy

def get_dataloaders(batch_size, device, max_length=100):
    """
    获取训练和验证数据加载器
    
    参数:
    batch_size (int): 每个批次的样本数
    device (torch.device): 计算设备 (CPU/GPU)
    max_length (int): 最大序列长度 (默认100)
    
    返回:
    train_loader, valid_loader: 训练集和验证集数据加载器
    SRC, TRG: 源语言和目标语言的Field对象
    
    用途:
    - 加载和预处理多语言数据集
    - 构建词汇表
    - 创建批处理迭代器
    """
    # 加载spacy分词器（需要提前安装spacy语言模型）
    # 德语: python -m spacy download de_core_news_sm
    # 英语: python -m spacy download en_core_web_sm
    try:
        spacy_de = spacy.load('de_core_news_sm')  # 德语分词器
        spacy_en = spacy.load('en_core_web_sm')   # 英语分词器
    except:
        raise ImportError("请先安装spacy语言模型: `python -m spacy download de_core_news_sm en_core_web_sm`")
    
    def tokenize_de(text):
        """
        德语分词函数
        
        参数:
        text (str): 输入文本
        
        返回:
        list: 分词后的token列表
        
        处理:
        - 转换为小写
        - 反转词序（提升翻译效果）
        """
        return [tok.text.lower() for tok in spacy_de.tokenizer(text)][::-1]
    
    def tokenize_en(text):
        """
        英语分词函数
        
        参数:
        text (str): 输入文本
        
        返回:
        list: 分词后的token列表
        
        处理:
        - 转换为小写
        """
        return [tok.text.lower() for tok in spacy_en.tokenizer(text)]
    
    # 定义源语言（德语）处理字段
    # tokenize=tokenize_de: 使用自定义德语分词器
    # init_token='<sos>': 添加序列开始标记
    # eos_token='<eos>': 添加序列结束标记
    # lower=True: 转换为小写
    # include_lengths=True: 包含序列实际长度信息
    SRC = Field(tokenize=tokenize_de,
                init_token='<sos>',
                eos_token='<eos>',
                lower=True,
                include_lengths=True)
    
    # 定义目标语言（英语）处理字段
    TRG = Field(tokenize=tokenize_en,
                init_token='<sos>',
                eos_token='<eos>',
                lower=True)
    
    # 加载Multi30k数据集
    # exts=('.de', '.en'): 源文件和目标文件扩展名
    # fields=(SRC, TRG): 指定处理字段
    train_data, valid_data, test_data = Multi30k.splits(
        exts=('.de', '.en'),
        fields=(SRC, TRG)
    )
    
    # 构建词汇表
    # min_freq=2: 只保留出现至少2次的词
    # specials: 添加特殊标记
    SRC.build_vocab(train_data, min_freq=2, 
                   specials=['<unk>', '<pad>', '<sos>', '<eos>'])
    TRG.build_vocab(train_data, min_freq=2, 
                   specials=['<unk>', '<pad>', '<sos>', '<eos>'])
    
    # 打印词汇表信息
    print(f"源语言词汇表大小: {len(SRC.vocab)}")
    print(f"目标语言词汇表大小: {len(TRG.vocab)}")
    print(f"最常见的10个源语言词: {SRC.vocab.freqs.most_common(10)}")
    print(f"最常见的10个目标语言词: {TRG.vocab.freqs.most_common(10)}")
    
    # 创建批处理迭代器
    # BucketIterator: 自动将相似长度样本分组，减少填充
    # sort_within_batch=True: 批次内按长度排序
    # sort_key: 排序依据（序列长度）
    # batch_size: 批次大小
    # device: 计算设备
    train_loader, valid_loader = BucketIterator.splits(
        (train_data, valid_data),
        batch_size=batch_size,
        sort_within_batch=True,
        sort_key=lambda x: len(x.src),
        device=device
    )
    
    return train_loader, valid_loader, SRC, TRG