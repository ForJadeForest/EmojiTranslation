# Emoji-Translation

## 问题描述

本赛题要求将带有emoji表情的文本翻译成原始的文本句子。

## 方案介绍

- 直接采用Seq2Seq方案

### Seq2Seq模型选择

- 经过对比，我们发现`bart`模型的效果会更加的好，同时训练的速度也比较符合我们的预期。
- 同时为了更好的效果，我们也训练了一个`bart-large model`。

### 训练配置

#### Tokenizer

- 由于本次比赛数据集与预训练中的风格，token都差异较大。比如最重要的emoji，大部分的预训练模型的训练集都是不包含emoji的。同时还有一些其他的词，比如一些繁体字等等。所以第一件事就是要添加生词。在最终我们在预训练的基础上添加1000+的生词，其中包含emoji以及一些符号表情，还有一些不常用的词语。
- 同时，我们注意到，在huggingface上的预训练的Bart的tokenizer会自动忽视空格。而空格在数据中大量分布。因此我们额外加入了一个token：`[BLANK]`，用于标注空格，同时将文中所有的空格字符替换为`[BLANK]`.

- 依据数据分析结果可知，文本中的最大长度不超过100，为了避免测试时遇见较长的句子，我们将tokenizer的最大长度设定为128。

#### Hyper Parameters

**对于Bart-base模型采用五折交叉验证训练**

##### Bart-base

- `lr:5e-5 `
- `warm_up_ratio: 0.1`
- `batch size: 256`

**对于Bart-large模型，我们其特别容易产生过拟合的现象。训练比较困难，而且训练时间也大大增加，因此我们只训练一个Bart-large模型**

##### Bart-large

- 为了减少计算时间，将max-length降低为100
- 为了减轻过拟合，学习率降低为`1e-6`
- `epoch: 25`
- `warm_up_ratio: 0.09`
- 生成的结果采用beam-search的方式，取beam_num为4



### 后处理

- 在测试集上会存在一些[unk]的字符，为了解决好这一问题，我们采用一个copy的机制，也就是说，对于这些[unk]的字符，要么采用直接复制，要么采用直接删除的方法。对于符号或者一些文本，我们采用直接复制的方法。对于一些[unk]的emoji表情，采用直接删除的方法。

### Ensemble

采用ensemble的方法去提升准确度。取5折训练出来的`bart-base`模型加上训练的一个`bart-large`模型。一共6个模型的生成结果。采用最简单的投票方法。同时，考虑到出现同票的结果。我们做了两个改进。

- `bart-large`的模型生成的结果会更加的好，因此如果出现同票直接选择`bart-large`模型的输出结果。
- 同时这也存在一定的不合理，比如`bart-large`模型有时候生成的结果是里面独一无二的。也就是只有一票。对于这种情况，我们决定采用计算句子的困惑度方式来进行判断。我们尝试使用别人预训练好的`bert`模型来进行尝试，发现效果比较一般。我们推测为是预训练的语料和本次比赛的语料差距较大。因此，需要在本次语料上重新进行 fine-tune。


### 文件配置
- `train.py` 训练单个model
- `train_bart_large.py` 用于训练`bart-large` model
- `finetune-mlm.py` 用于训练计算困惑度的`bert-base` model
- `k-fold_train.py` 用于训练k折model
- `pl_train.py` 用于使用伪标签进行训练
- `predict.py` 用于生成结果
- `predict_filter.py` 生成结果并使用后处理
- `utils.py` 一些工具函数
- `ensemble.py` 用于进行ensemble