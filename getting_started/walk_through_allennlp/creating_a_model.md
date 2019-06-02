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

为了初始化这个模块

To initialize this module, we just need the number of tags and optionally some constraints
(represented as a list of allowed pairs `(from_tag_index, to_tag_index)`):

```python
    def __init__(self,
                 num_tags: int,
                 constraints: List[Tuple[int, int]] = None) -> None:
        super().__init__()
        self.num_tags = num_tags

        # transitions[i, j] is the logit for transitioning from state i to state j.
        self.transitions = torch.nn.Parameter(torch.Tensor(num_tags, num_tags))

        # _constraint_mask indicates valid transitions (based on supplied constraints).
        if constraints is None:
            self._constraint_mask = None
        else:
            constraint_mask = torch.Tensor(num_tags, num_tags).fill_(0.)
            for i, j in constraints:
                constraint_mask[i, j] = 1.

            self._constraint_mask = torch.nn.Parameter(constraint_mask, requires_grad=False)

        # Also need logits for transitioning from "start" state and to "end" state.
        self.start_transitions = torch.nn.Parameter(torch.Tensor(num_tags))
        self.end_transitions = torch.nn.Parameter(torch.Tensor(num_tags))
```

I'm not going to get into the exact mechanics of how the log-likelihood is calculated;
you should read the aforementioned overview paper
(and look at our implementation)
if you want the details. The key points are

- the input to this module is a `(sequence_length, num_tags)` tensor of logits
  representing the likelihood of each tag at each position in some sequence
  and a `(sequence_length,)` tensor of gold tags. (In fact, we actually provide
  _batches_ consisting of multiple sequences, but I'm glossing over that detail.)
- The likelihood of producing a certain tag at a certain sequence position depends on both
  the input logits at that position and the transition parameters corresponding to the
  tag at the previous position
- Computing the overall likelihood requires summing across all possible tag sequences,
  but we can use clever dynamic programming tricks to do so efficiently.
- We also add a `viterbi_tags()` method that accepts some input logits,
  gets the transition probabilities, and uses the
  [Viterbi algorithm](https://en.wikipedia.org/wiki/Viterbi_algorithm)
  and the supplied constraints
  to compute the most likely sequence of tags for a given input.

## Implementing the CRF Tagger Model

The `CrfTagger` is quite similar to the `SimpleTagger` model,
so we can take that as a starting point. We need to make the following changes:

- give our model a `crf` attribute containing an appropriately initialized
  `ConditionalRandomField` module
- replace the softmax class probabilities with the Viterbi-generated most likely tags
- replace the softmax + cross-entropy loss function
  with the negative of the CRF log-likelihood

We can then register the new model as `"crf_tagger"`.

## Creating a Dataset Reader

The [CoNLL data](https://www.clips.uantwerpen.be/conll2003/ner/) is formatted like

```
   U.N.         NNP  I-NP  I-ORG
   official     NN   I-NP  O
   Ekeus        NNP  I-NP  I-PER
   heads        VBZ  I-VP  O
   for          IN   I-PP  O
   Baghdad      NNP  I-NP  I-LOC
   .            .    O     O
```

where each line contains a token, a part-of-speech tag, a syntactic chunk tag, and a named-entity tag.
An empty line indicates the end of a sentence, and a line

```
-DOCSTART- -X- O O
```

indicates the end of a document. (Our reader is concerned only with sentences
and doesn't care about documents.)

You can poke at the code yourself, but at a high level we use
[`itertools.groupby`](https://docs.python.org/3/library/itertools.html#itertools.groupby)
to chunk our input into groups of either "dividers" or "sentences".
Then for each sentence we split each row into four columns,
create a `TextField` for the token, and create a `SequenceLabelField`
for the tags (which for us will be the NER tags).

## Creating a Config File

As the `CrfTagger` model is quite similar to the `SimpleTagger` model,
we can get away with a similar configuration file. We need to make only
a couple of changes:

- change the `model.type` to `"crf_tagger"`
- change the `"dataset_reader.type"` to `"conll2003"odule, we just n`
- add a `"dataset_reader.tag_label"` field with value "ner" (to indicate that the NER labels are what we're predicting)

We don't *need* to, but we also make a few other changes

- following [Peters, Ammar, Bhagavatula, and Power 2017](https://www.semanticscholar.org/paper/Semi-supervised-sequence-tagging-with-bidirectiona-Peters-Ammar/73e59cb556351961d1bdd4ab68cbbefc5662a9fc), we use a
  Gated Recurrent Unit (GRU) character encoder
  as well as a GRU for our phrase encoder
- we also start with pretrained [GloVe vectors](https://nlp.stanford.edu/projects/glove/) for our token embeddings
- we add a regularizer that applies a L2 penalty just to the `transitions`
  parameters to help avoid overfitting
- we add a `test_data_path` and set `evaluate_on_test` to true.
  This is mostly to ensure that our token embedding layer loads the GloVe
  vectors corresponding to tokens in the test data set, so that they are not
  treated as out-of-vocabulary at evaluation time. The second flag just evaluates
  the model on the test set when training stops. Use this flag cautiously,
  when you're doing real science you don't want to evaluate on your test set too often.

## Putting It All Together

At this point we're ready to train the model.
In this case our new classes are part of the `allennlp` library,
which means we can just use `allennlp train`:

```bash
$ allennlp train \
    tutorials/getting_started/walk_through_allennlp/crf_tagger.json \
    -s /tmp/crf_model
```

If you were to create your own model outside of the allennlp codebase,
you would need to load the modules where you've defined your classes.
Otherwise they never get registered and then AllenNLP is unable to
instantiate them based on the configuration file.

You can specify one or more extra packages using the
`--include-packages` flag. For example, imagine that
your model is in the module `myallennlp.model`
and your dataset reader is in the module `myallennlp.dataset_reader`.

Then you would just

```bash
$ allennlp train \
    /path/to/your/model/configuration \
    -s /path/to/serialization/dir \
    --include-package myallennlp
```

and (as long as your package is somewhere on the PATH
where Python looks for packages), your custom classes
will all get registered and used correctly.