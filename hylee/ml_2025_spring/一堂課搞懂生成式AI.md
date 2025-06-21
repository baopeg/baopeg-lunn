# 一堂课搞懂生成式AI

## 1. 有什么样的行为

Breezy Voice 语言合成模型

**就算人生一团乱，只要一步一步努力去除噪音，也能拼凑出美丽图景**

gamma AI生成PPT

机器展示撕开reasoning的过程

以前： 问题 -> 机器 -> 直接给出答案

现在模型：问题 -> 机器 -> 思维链 -> 答案

ai agent：规划能力、使用工具、规划能力、从经验学习

deep research 深度思考

claude computer use / chatgpt operator  
 
任务+屏幕截图 操作鼠标和键盘，来实现操作页面的目的。

开发机器学习模型也需要很多步骤



## 2. 运作原理

输入 -》 人工智慧 -》 输出

策略：根据固定的次序，每次只产生一次token

自回归模型

只有解码的transform：

深度不够，长度来凑

testing time scaling

simple test-time scaling

mamba另一种类似的模型 

复杂物体由有限的基本单位组成，选择有限token

文法树

## 3. 怎么产生出来的

类神经网络：架构 vs 参数

架构（天资）：开发者决定，超参数  
参数（后天努力的结果） ：训练资料决定

找出参数，训练模型

机器学习中的分类问题（做选择）

专才   ->    通才

prompt 提示词

过去，不同语言不可以共用系统，一种语言只能对应一种系统，不现实，
希望能有一个通用翻译，可以翻译多种语言。输入语言，变成内部机器自己的言语，然后翻译成目标语言

通用机器学习模型：第一形态 elmo、bert 、ermie
第二形态：gpt-3参数需要微调，架构相同，参数不同。 
第三形态：chatgpt lamma、gimini、deepseek

## 4. 怎么赋予新的能力  

以前:从零开始
现在：已经具备基本知识

机器的终身学习技术：life-long-learning

改变基础模型的参数：微调fine-tune，谨慎使用，回答会变得混乱。

模型编辑，model editing 

 模型融合，model merging


