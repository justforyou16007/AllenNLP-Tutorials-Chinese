# Creating Your Own Models

创建模型的步骤：

- 创建一个`DatasetReader`
- 创建一个`Model`

一般来说，为了实现新模型，您需要实现一个 `DatasetReader`子类来读入您的数据集以及`Model`与您要实现的模型相对应的 子类。（如果已经有一个`DatasetReader`要使用的数据集，当然可以重用那个。）在本教程中，我们还将实现一个自定义`PyTorch Module`，但一般情况下不需要这样做。

我们的简单tagger模型使用LSTM来捕获输入句子中单词之间的依赖关系，但是没有很好的方法来捕获标记之间的依赖关系。对于像[命名实体识别](https://en.wikipedia.org/wiki/Named-entity_recognition)这样的任务来说。我们将尝试构建一个NER模型，该模型可以胜过[CoNLL 2003数据集](https://www.clips.uantwerpen.be/conll2003/ner/)上的简单标记器，由于许可原因，您必须自己获取。

简单标记器 在验证数据集上使用span-based F1损失获得大约88％，我们想做得更好。
on the validation dataset. We'd like to do better.

解决此问题的一种方法是 在标记模型的末尾添加CRF层。（如果你不熟悉CRF，[这个概述文章](https://arxiv.org/abs/1011.4088) 是有用的，就像[这个PyTorch教程一样](http://pytorch.org/tutorials/beginner/nlp/advanced_tutorial.html)。）

我们使用的线性链CRF的转移损失是一个 `num_tags` x `num_tags` 大小的转移矩阵， `transitions[i, j]`代表 从第`i`个标签紧跟第`j`个标签的概率。且除了要预测词的标签外，还要预测出句首句尾。

再者，我们的CRF会接受一组可选的不合理转换作为约束条件。例如，我们的NER数据有不同的标记，分别表示开始、中间和结束实体类型的。我们不允许使用“个人实体的开始”标签后面是“位置结束实体标记”。

CRF只是我们模型的一部分，我们会以模块的方式实现它。

## Implementing the CRF Module

为了实现一个CRF的PyTorch模型，我们需要继承`torch.nn.Module`类并复写：

```python
    def forward(self, *input):
        ...
```

计算输入的对数概率。

- 初始化模块`__init__()`

为了初始化这个模块，我们只需要一些标签和可选的一些约束(表示为被允许tag索引对列表 `(from_tag_index, to_tag_index)`):

```python
    def __init__(self,
                 num_tags: int,
                 constraints: List[Tuple[int, int]] = None) -> None:
        super().__init__()
        self.num_tags = num_tags

        # transitions[i, j]是从状态i到状态j的转移概率.
        self.transitions = torch.nn.Parameter(torch.Tensor(num_tags, num_tags))

        # _constraint_mask表示有效转换(基于提供的约束).
        if constraints is None:
            self._constraint_mask = None
        else:
            constraint_mask = torch.Tensor(num_tags, num_tags).fill_(0.)
            for i, j in constraints:
                constraint_mask[i, j] = 1.
		# torch.nn.Parameter可以暂时存储网络的参数且默认require_grad = False
            self._constraint_mask = torch.nn.Parameter(constraint_mask, requires_grad=False)

        # 还需要计算从“开始”状态转换到“结束”状态的概率。
        self.start_transitions = torch.nn.Parameter(torch.Tensor(num_tags))
        self.end_transitions = torch.nn.Parameter(torch.Tensor(num_tags))
```

- 输入模块是一个 `(sequence_length, num_tags)`形状的概率张量表示预测到的每个标签在每个位置的概率。一个 `(sequence_length,)` 长度的概率张量。 (事实上，我们实际上提供

  _batches由多个序列组成，但我正在掩盖这些细节。)

- 在某个序列位置生成某个标签的概率取决于该位置的输入逻辑和对应于在上一个位置标记。
  
- 计算总体概率需要对所有可能的标记序列求和，但是我们可以使用聪明的动态编程技巧来有效地做到这一点。
- 我们也添加了一个`viterbi_tags()` 方法允许一些输入概率获取转移概率矩阵并使用[Viterbi algorithm](https://en.wikipedia.org/wiki/Viterbi_algorithm)以及提供的约束条件为给定的输入计算最可能的标记序列。

## Implementing the CRF Tagger Model

`crftagger`与`simpletagger`模型非常相似，所以我们可以把它作为一个起点。我们需要做以下更改：

- 给我们的模型一个包含`CRF`的`conditionalRandomField`模块。
- 用`Viterbi-generated`最有可能的标签替换softmax概率。
- 使用负CRF对数概率替换softmax和交叉熵。

我们用`"crf_tagger"`注册新模型。

## Creating a Dataset Reader

[CoNLL data](https://www.clips.uantwerpen.be/conll2003/ner/)的格式如下：

```
   U.N.         NNP  I-NP  I-ORG
   official     NN   I-NP  O
   Ekeus        NNP  I-NP  I-PER
   heads        VBZ  I-VP  O
   for          IN   I-PP  O
   Baghdad      NNP  I-NP  I-LOC
   .            .    O     O
```

其中每行从左至右依次为词、语音标记、句法标记和命名体实体标记。空行表示句子结尾。

```
-DOCSTART- -X- O O
```

指示文档的结尾。（我们的读者只关心句子，不关心文档。）

我们使用[`itertools.groupby`](https://docs.python.org/3/library/itertools.html#itertools.groupby)将输入转换为包含 "dividers"和"sentences"的组，然后我们把每一行分成四列，为词创建一个 `TextField` 并为标签创建一个 `SequenceLabelField`（NER标签）。

## Creating a Config File

由于 `CrfTagger` 模型与 `SimpleTagger` 模型非常相似，因此我们可以使用类似的配置文件。我们只需要

一些变化：

- 将`model.type` 改为 `"crf_tagger"`
- 把 `"dataset_reader.type"` 改成 `"conll2003"`
- 添加一个值为“ner”的 `"dataset_reader.tag_label"` 字段（指示ner标签是我们预测的内容）。

即便我们不需要，但我们也做了一些其他的改变：

- 根据[Peters, Ammar, Bhagavatula, and Power 2017](https://www.semanticscholar.org/paper/Semi-supervised-sequence-tagging-with-bidirectiona-Peters-Ammar/73e59cb556351961d1bdd4ab68cbbefc5662a9fc)，我们使用了对短语编码的GRU字符编码
- 我们使用预训练模型[GloVe vectors](https://nlp.stanford.edu/projects/glove/)对我们的词向量化
- 我们添加了一个正则化器，“l2”惩罚应用于`transitions`帮助避免参数过拟合
  
- 我们添加了`test_data_path`和一组 `evaluate_on_test` 设置为`True`。这是为了确认我们的词向量层加载了对应了测试数据集中词的GloVe模型，第一个是为了测试模型，第二个是为了评估我们的模型。谨慎使用这个标志，当你做真正的科学研究时，你不想太频繁地在测试集上评估。

## Putting It All Together

现在我们已经准备好训练这个模型了。在这种情况下我们的新类是 `allennlp`库的一部分，即我们可以用`allennlp train`来训练模型：

```bash
$ allennlp train \
    tutorials/getting_started/walk_through_allennlp/crf_tagger.json \
    -s /tmp/crf_model
```

如果要在Allennlp代码库之外创建自己的模型，您需要在定义类的地方注册模块。否则，它们永远不会注册，然后Allennlp无法根据配置文件调用它们。

可以使用`--include-packages`标志指定一个或多个额外的包。

例如，假设您的模型在模块`myallennlp.model`中，而您的数据集读取器在模块`myallennlp.dataset_reader`中。

接下来我们只需要：

```bash
$ allennlp train \
    /path/to/your/model/configuration \
    -s /path/to/serialization/dir \
    --include-package myallennlp
```

而且（只要您的包位于python查找包的路径上），您的自定义类都将被正确注册和使用。