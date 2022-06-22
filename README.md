# 《基于PyTorch的自然语言处理》阅读笔记
&emsp;&emsp;本书主要是基于作者（Delip Rao、Brian McMahan）在O'Reilly的AI and Strata会议上的2天NLP培训内容，书中列出了很多值得阅读的NLP经典论文，推荐具有深度学习和PyTorch基础的同学阅读，本项目保留原始项目 [PyTorchNLPBook](https://github.com/joosthub/PyTorchNLPBook) 的所有文件，并新增了第2章的代码，整理本书涉及到的所有数据集和依赖包，方便大家阅读和运行代码。

## 运行环境
Python 3.8 Windows环境

### 创建本地虚拟环境
在项目的根目录下执行如下命令：
```shell
python -m venv venv
```

### 安装相关的依赖包
在项目的根目录下执行如下命令：
```shell
pip install -r requirements.txt
```

### 安装Jupyter Kernel
在项目的根目录下执行如下命令：
```shell
python -m ipykernel install --user --name nlpbook
```

### 安装PyTorch
```shell
pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu113
```

### 下载数据文件
已经整理好了所有需要下载的数据集，并提供相关的wheel安装包，百度网盘地址如下：  
链接：https://pan.baidu.com/s/17pbW6ernwJrm95pPrL1iIg  
提取码：5dvx  
<pre>
whls---------------------------------------需要安装的依赖包
nltk_data----------------------------------nltk的tokenizers包
data---------------------------------------本书数据集
</pre>

- data：本书数据集按照需要可放置于每个章节对应的`data`目录下，在其中已说明本章需要用到的数据集  
- nltk_data：将文件夹目录直接放置在venv目录下，主要用于nltk的语言解析
- whls：额外依赖包，包括`pytorch`和`spacy`的语言依赖包，使用`pip install`命令安装即可

## 个人总结
&emsp;&emsp;《基于PyTorch的自然语言处理》这本书总体上写得不错，但是翻译得太烂了，和论文中的术语完全不对应，强烈推荐看英文原版。本书内容的难度曲线较为陡峭，从第3章开始，代码量就逐步上升了，建议精读第3章的代码之后，后续章节的代码就会很好理解（都是类似的模块化代码）。  
&emsp;&emsp;本书主要围绕着NLP模型和解决场景进行讲解，逐步介绍基础的Perceptron（感知机模型）、MLP模型（多层感知机模型）、CNN（卷积神经网络）、CBOW模型（连续词袋模型）、RNN（递归神经网络）、ElmanRNN（普通RNN）和基于注意力机制的Encoder-Decoder模型：
1. 第1章主要介绍了基于文本处理的PyTorch基本知识，包括随机梯度下降法、`one-hot`编码、TF/TF-IDF、计算图和张量的基本操作。关于PyTorch的学习，建议大家学习Datawhale开源社区的[深入浅出PyTorch课程](https://datawhalechina.github.io/thorough-pytorch/)
2. 第2章主要介绍了NLP的基本术语和思想，包括语料库、词/词类型等术语，并介绍了很多的文本处理方法，主要基于`spack`库，例如文本分类、n元模型生成、词形还原、词性标注、广度分块（名词短语分块）等操作。笔者在原有教程中补充了本章的代码部分，详见第2章的代码
3. 第3章比较重要，理解了整体代码思路和各个模块的作用，能提升后续章节的代码理解。主要介绍了神经网络的基本概念，包括感知器模型、激活函数（`sigmodi`、`Tanh`、`ReLU`、`Softmax`）、损失函数（`MESLoss`、`CrossEntropyLoss`、`BCELoss`）、监督学习相关知识（模型评估、早停、规范化），介绍了基于感知器模型的`Yelp`数据集的餐馆评论分类预测案例，需要重点理解`Vocabulary`（词汇表）、`Vectorizer`（矢量化）和`DataLoader`（数据转换器）的相关代码
4. 第4章主要介绍了基于MLP模型和CNN模型的文本处理方法，通过对姓氏分类案例，讲解了用于自然语言处理的前馈网络方法
5. 第5章主要介绍了`Embedding`（词嵌入）的使用，通过文档分类案例，讲解了CBOW（连续词袋模型）和基于预处理词向量的方法
6. 第6章主要介绍了`ElmanRNN`模型，通过对姓氏国籍分类案例，讲解基于字符的RNN模型的处理方法
7. 第7章主要介绍了基于门控的`ElmanRNN`（即GRU）模型，通过对无条件/条件姓氏生成模型的案例，讲解了`GRU`序列模型的处理方法
8. 第8章主要介绍了`Seq2Seq`模型中的基于注意力的编码器-解码器模型，通过对神经机器翻译案例，讲解了`Encoder-Decoder`序列模型的处理方法
9. 第9章主要介绍了NLP的应用场景和在2018年的发展趋势（本书写作的时间比较长，距离中文版出版已经有3年了），主要包括对话与交互系统、话语理解、信息提取与文本挖掘、文件分析与检索等场景，NLP系统的设计模式主要包括在线与离线系统（垃圾邮件检测系统）、交互系统与非交互系统、单峰与多峰系统（新闻记录系统）、端到端系统与分段系统（机器翻译、摘要、语音合成）、封闭域与开放域系统（文件标签系统、语音识别系统）、单语言与多语言系统