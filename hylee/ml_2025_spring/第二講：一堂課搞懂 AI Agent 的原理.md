# AI Agent原理

人为设定目标
agent观察目前状况 boservation
action 执行


## 根据经验调整行为

不更新模型参数调整模型的行为。根据反馈的内容调整模型的行为。

过去发生的事情都给模型反馈，造成上下文长度限制。每次思考都需要思考过往所有。
解决办法，添加memory组件和read组件。根据检索系统检索记忆单元。RAG。 

平均正确率

告诉要做什么而不是不要做什么   

## 语言模型的判断能力
 
内外知识互相拉扯，在做rag的时候。


什么样的外部知识比较容易说服AI
结论：外部知识与模型本身的信念的差距相近比较容易介绍。

就算工具可靠，但不代表AI就不会犯错。

使用工具与模型本省能力简的平衡。

使用工具不一定比较有效率。

AI 能不能做计划呢？

运行过程中做输入和输出。

plan  要有计划，更要能改变计划的弹性。  观察到之后，思考是否需要调整计划。 

