# Configuring Experiments

## Registrable and from_params

当我们自定义网络时，训练需要使用**`--include-package`**参数来让allennlp找到我们自定义的模型和数据读取器。也可以在python中调用**`allennlp.commands.main()`**来实现(**不建议**)

## Datasets and Instances and Fields

我们要在**`Dataset`**上训练和评价模型，**`Dataset`**是一个**`Instance`**的集合，在我们的**tagging实验**中，每个数据集都是已经被标记好的句子集合，并且每个**`Instance`**都是被标记好的句子集合。

一个**`Instance`**包含了许多**`Field`**（可能会有**`TextField`**或者其他**`LabelField`**），每个**`Field`**都将**`Instance`**表示为适合喂给模型的数组。

在我们的标记实验中每个**`Istance`**都包含了一个**`TextField`**代表句子中的词或者token，**`SequenceLabField`**代表想对应的部分句子的标签。

想要将文本所有句子放入Dataset中就需要有配置文件规定的DatasetReader。

## DatasetReaders

我们的配置文件第一部分就定义了**`dataset_reader`**：

```json
  "dataset_reader": {
    "type": "sequence_tagging",
    "word_tag_delimiter": "/",
    "token_indexers": {
      "tokens": {
        "type": "single_id",
        "lowercase_tokens": true
      },
      "token_characters": {
        "type": "characters"
      }
    }
  },
```

在这里我们明确规定要使用注册过的名为**`sequence_tagging`**的**`DatasetReader`**这是**`SequenceTaggingDatasetReader`**的子类。这个reader假定的数据集格式为**`WORD###TAG [TAB] WORD###TAG [TAB]....`**，每个词和它的标签用**`word_tag_delimiter`**分隔开，每对词和标签使用**`token_delimiter`**。因此我们的数据集类似于下面这个样子：

```bash
The/at detectives/nns placed/vbd Barco/np under/in arrest/nn
```

这就是为什么我们的（**`word_tag_delimiter`**是**`sequence_tagging`**的一个参数，因此可以在配置文件中使用）：

```json
"word_tag_delimiter": "/",
```

查看**`SequenceTaggingDatasetReader.read()`**的源码会发现，它将每个句子转换为token组成的**`TextField`**和把标签转换为**`SequenceLabelField`**。后者不是可配置的，但前者需要一个**`TokenIndexer`**字典，它指示如何将tokens转换成数组。

我们的配置文件主要在两个token_indexer中

```js
"token_indexers": {
  "tokens": {
    "type": "single_id",
    "lowercase_tokens": true
  },
  "token_characters": {
    "type": "characters"
  }
}
```
`tokens`是一个`SingleIdTokenIndexer`，它将每个token（word）表示为单个整型序号（即可以通过这个序号索引到对应的token）;这个配置还有一个参数：`lowercase_tokens`，即是将tokens在编码前转换为小写字母表示。

然后是`token_characters`，它是`TokenCharactersIndexer`：可以将每个token表示为一个整型编码的列表（one-hot编码）。

注意：这会给我们两种对于token的不同编码，每个编码都有一个名字，即`tokens`和`token_characters`，这两个会在模型中被引用。

## Training and Validation Data

下一部分指定模型的训练数据和测试数据：

```js
  "train_data_path": "https://allennlp.s3.amazonaws.com/datasets/getting-started/sentences.small.train",
  "validation_data_path": "https://allennlp.s3.amazonaws.com/datasets/getting-started/sentences.small.dev",
```

两个分别表示训练数据位置和测试数据位置，地址接下来会被传入cache中，allennlp会自动下载数据。

## The Model

这个部分用来配置我们的模型：

```js
  "model": {
    "type": "simple_tagger",
```

这里使用的是`SimpleTagger`模型，源码：

```python
allennlp.models.simple_tagger.SimpleTagger(
text_field_embedder:allennlp.modules.text_field_embedders.text_field_embedder.TextFieldEmbedder, 
encoder: allennlp.modules.seq2seq_encoders.seq2seq_encoder.Seq2SeqEncoder）
```

你会发现其中有上述两个参数（其他参数未列出），第一个是`text_field_embedder=TextFieldEmbedder`，第二个是`encoder=Seq2SeqEncoder`。它们会将输入序列转换为输出序列（编码），并通过线性层将输出转换为预测标签的概率。

## The Text Field Embedder

我们首先看一下text field embedder的配置文件：

```js
  "text_field_embedder": {
    "tokens": {
      "type": "embedding",
      "embedding_dim": 50
    },
    "token_characters": {
      "type": "character_encoding",
      "embedding": {
        "embedding_dim": 8
      },
      "encoder": {
        "type": "cnn",
        "embedding_dim": 8,
        "num_filters": 50,
        "ngram_filter_sizes": [
          5
        ]
      },
      "dropout": 0.2
    }
  },
```

这个文件是为了配置上文提到的`TextFieldEmbedder`，先将`tokens`（包含了整数编码的小写体词）输入到`Embedding`中，这个模块会把词编码到50维（通过`embedding_dim`控制）上;

接下来`token_characters`的作用是将上面的输出喂给`TokenCharactersEncoder`，它输出8维的向量，接着输入到`CnnEnconder`中，通过`num_filters`输出50维的编码。`dropout`规定了CNN编码器的dropout参数（0.2代表在训练中随机丢失20%的连接）。

输出为将`tokens`编码产生的50维向量和`token_characters`产生的50维向量连接，产生的100维的向量。

因为`TextFields` 和 `TextFieldEmbedder` 都这样配置，这样实验两种编码方式的不同是很简单的，甚至可以使用预训练模型得到已经做好了的embeddings，而不需要去改变一行代码。

### The Seq2SeqEncoder

 要将`TextFieldEmbedder`的输出交给 "stacked encoder"处理就需要`Seq2SeqEncoder`：

```js
    "encoder": {
      "type": "lstm",
      "input_size": 100,
      "hidden_size": 100,
      "num_layers": 2,
      "dropout": 0.5,
      "bidirectional": true
    }
```

这是由`torch.nn.LSTM`集成的`"lstm"`编码器并且它的参数只是简单的从PyTorch constructor中产生的。它的输入大小需要匹配前面产生的100维向量，但是连接在它之后的线性层不需要再配置

## Training the Model

剩下的就是训练模型了：

```js
  "iterator": {"type": "basic", "batch_size": 32},
```

我们使用一个名为`BasicIterator`的迭代器。

```js
  "trainer": {
    "optimizer": "adam",
    "num_epochs": 40,
    "patience": 10,
    "cuda_device": -1
  }
}
```

最后我们使用`torch.optim.Adam`优化我们的网络参数，`patience`参数的作用是当模型训练了`patience`轮后如果还有任何优化就停止训练， `cuda_device` 的作用是选择要使用的设备，`-1`代表最后一个设备。

### Next Steps

下个教程 [Creating a Model](creating_a_model.md) 。

