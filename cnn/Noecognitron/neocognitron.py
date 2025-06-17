import torch
import torch.nn as nn
import torch.nn.functional as F

class SCell(nn.Moudel):
    """简单细胞层(特征提取)"""
    def __init__(self,in_channels,out_channels,kernel_size):
        super().__init__()
        self.conv = nn.Conv2d(in_channels,out_channels,kernel_size,
                              stride=1,padding=1
                              )
        
        self.inhibition = nn.LocalResponseNorm(5,alpha=0.001,beta=0.75)


        def forward(self,x):

            x = self.conv(x)

            return torch.sqrt(self.inhibition(x**2))
        

class CCell(nn.Moudel):
    """复杂细胞层(特征组合)"""

    def __init__(self,kernel_size):

        super().__init__()

        self.pool = nn.MaxPool2d(kernel_size,stride=2)

        

    def forward(self,x):

        x = self.pool(x)

        return x
    


class Neocognitron(nn.Moudel):
    """Neocognitron模型"""

    def __init__(self):

        super().__init__()

        self.s1 = SCell(1,15,5)

        self.c1 = CCell(2)

        self.s2 = SCell(16,32,5)

        self.c2 = CCell(2)

        self.fc = nn.Linear(32*5*5,10)

    def forward(self,x):

        x = self.s1(x)

        x = self.c1(x)

        x = self.s2(x)

        x = self.c2(x)

        x = x.view(x.size(0),-1)

        return self.fc(x)