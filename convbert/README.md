# Turkish ConvBERT model

We release a ConvBERT base model for Turkish, that were trained on the same data as *BERTurk*.

> Pre-trained language models like BERT and its variants have recently achieved impressive performance
> in various natural language understanding tasks. However, BERT heavily relies on the global 
> self-attention block and thus suffers large memory footprint and computation cost. Although all its
> attention heads query on the whole input sequence for generating the attention map from a global
> perspective, we observe some heads only need to learn local dependencies, which means the existence
> of computation redundancy. We therefore propose a novel span-based dynamic convolution to replace these
> self-attention heads to directly model local dependencies. The novel convolution heads, together with
> the rest self-attention heads, form a new mixed attention block that is more efficient at both global
> and local context learning. We equip BERT with this mixed attention design and build a ConvBERT model.
> Experiments have shown that ConvBERT significantly outperforms BERT and its variants in various
> downstream tasks, with lower training cost and fewer model parameters. Remarkably, ConvBERTbase model
> achieves 86.4 GLUE score, 0.7 higher than ELECTRAbase, while using less than 1/4 training cost.
> Code and pre-trained models will be released.

More details about ConvBERT can be found in the
["ConvBERT: Improving BERT with Span-based Dynamic Convolution"](https://arxiv.org/abs/2008.02496)
or in the [official ConvBERT repository](https://github.com/yitu-opensource/ConvBert) on GitHub.

Details about the pre-training process can be found in the [cheatsheet](CHEATSHEET.md).
