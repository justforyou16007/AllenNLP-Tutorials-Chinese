# Model's Training,Evaluating and Predicting

## Train a Model

```bash
allennlp train tutorials/getting_started/walk_through_allennlp/simple_tagger.json --serialization-dir /tmp/tutorials/getting_started
```

> `serialization-dir`：model的存储位置
>
> `simple_tagger.json`：训练所需配置文件

## Evaluating a Model

```bash
allennlp evaluate /tmp/tutorials/getting_started/model.tar.gz https://allennlp.s3.amazonaws.com/datasets/getting-started/sentences.small.test
```

> `model.tar.gz`：训练时保存的模型
>
> `sentences.small.test`：测试数据集

## Making Predictions

```bash
$ cat <<EOF >> inputs.txt
{"sentence": "I am reading a tutorial."}
{"sentence": "Natural language processing is easy."}
EOF
```

> `inputs.txt`：要预测的两句话

```bash
$ allennlp predict /tmp/tutorials/getting_started/model.tar.gz inputs.txt
... lots of logging omitted
{"tags": ["ppss", "bem", "vbg", "at", "nn", "."], "class_probabilities": [[ ... ]]}
{"tags": ["jj", "nn", "nn", "bez", "jj", "."], "class_probabilities": [[ ... ]]}
```

> 可以看到输出了两句话的词性。

