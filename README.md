# 🇹🇷 BERTurk

[![DOI](https://zenodo.org/badge/237817454.svg)](https://zenodo.org/badge/latestdoi/237817454)

We present community-driven cased and uncased models BERT models for Turkish 🎉

Some datasets used for pretraining and evaluation are contributed from the
awesome Turkish NLP community, as well as the decision for the model name: BERTurk.

# Changelog

* 12.05.2020: Release of ELECTRA (small and base) models, see [here](electra/README.md)
* 25.03.2020: Release of *BERTurk* uncased model and *BERTurk* models with larger vocab size (128k, cased and uncased)
* 11.03.2020: Release of the cased distilled *BERTurk* model: *DistilBERTurk*.
              Available on the [Hugging Face model hub](https://huggingface.co/dbmdz/distilbert-base-turkish-cased)
* 17.02.2020: Release of the cased *BERTurk* model.
              Available on the [Hugging Face model hub](https://huggingface.co/dbmdz/bert-base-turkish-cased)
* 10.02.2020: Training corpus update, new TensorBoard links, new results for cased model
* 02.02.2020: Initial version of this repo.

# Stats

The current version of the model is trained on a filtered and sentence
segmented version of the Turkish [OSCAR corpus](https://traces1.inria.fr/oscar/),
a recent Wikipedia dump, various [OPUS corpora](http://opus.nlpl.eu/) and a
special corpus provided by [Kemal Oflazer](http://www.andrew.cmu.edu/user/ko/).

The final training corpus has a size of 35GB and 44,04,976,662 tokens.

Thanks to Google's TensorFlow Research Cloud (TFRC) we can train both cased and
uncased models on a TPU v3-8. You can find the TensorBoard outputs for
the training here:

* [TensorBoard cased model](https://tensorboard.dev/experiment/ZgFk8LclQOKdW0pYWviLMg/)
* [TensorBoard uncased model](https://tensorboard.dev/experiment/5LlD11cWRwexyqKSEPPXGA/)

We also provide cased and uncased models that aŕe using a larger vocab size (128k instead of 32k).

A detailed cheatsheet of how the models were trained, can be found [here](CHEATSHEET.md).

# *DistilBERTurk*

The distilled version of a cased model, so called *DistilBERTurk*, was trained
on 7GB of the original training data, using the cased version of *BERTurk*
as teacher model.

*DistilBERTurk* was trained with the official Hugging Face implementation from
[here](https://github.com/huggingface/transformers/tree/master/examples/distillation).

The cased model was trained for 5 days on 4 RTX 2080 TI.

More details about distillation can be found in the
["DistilBERT, a distilled version of BERT: smaller, faster, cheaper and lighter"](https://arxiv.org/abs/1910.01108)
paper by Sanh et al. (2019).

# ELECTRA

In addition to the *BERTurk* models, we also trained ELECTRA small and base models. A detailed overview can be found
in the [ELECTRA section](electra/README.md).

# Evaluation

We use the [token classification example](https://github.com/huggingface/transformers/tree/master/examples/token-classification)
from 🤗/Transformers for evaluation on both PoS and NER datasets.
We report averaged Accuracy (PoS tagging) and F-Score (NER) on 5 runs (initialized with 5 different seeds).

*BERTurk* and ELECTRA model checkpoint selection: We evaluated 5 different checkpoints for our cased and uncased models based on
the development score for PoS tagging and NER. The model with the best results is used for the final and released model.

## PoS tagging

The Turkish [IMST dataset](https://github.com/UniversalDependencies/UD_Turkish-IMST) 
from Universal Dependencies is used for PoS tagging evaluation. We use the `dev` branch and
commit `a6c955`. Result on development set is reported in brackets.


| Model                  | Run 1             | Run 2             | Run 3             | Run 4             | Run 5             | Avg.
| ---------------------- | ----------------- | ----------------- | ----------------- | ----------------- | ----------------- | ---------------------
| ELECTRA small          | (0.9567) / 0.9584 | (0.9578) / 0.9589 | (0.9564) / 0.9591 | (0.9544) / 0.9585 | (0.9545) / 0.9582 | (0.9560) / 0.9586
| ELECTRA base           | (0.9707) / 0.9734 | (0.9710) / 0.9734 | (0.9712) / 0.9745 | (0.9728) / 0.9719 | (0.9711) / 0.9727 | (0.9714) / **0.9732**
| mBERT                  | (0.9573) / 0.9580 | (0.9554) / 0.9584 | (0.9556) / 0.9591 | (0.9594) / 0.9572 | (0.9580) / 0.9586 | (0.9571) / 0.9583
| BERTurk (32k)          | (0.9701) / 0.9712 | (0.9731) / 0.9717 | (0.9728) / 0.9730 | (0.9719) / 0.9729 | (0.9728) / 0.9708 | (0.9722) / 0.9719
| BERTurk (128k)         | (0.9707) / 0.9732 | (0.9716) / 0.9712 | (0.9702) / 0.9722 | (0.9675) / 0.9715 | (0.9711) / 0.9729 | (0.9703) / 0.9722
| BERTurk uncased (32k)  | (0.9707) / 0.9703 | (0.9711) / 0.9713 | (0.9715) / 0.9705 | (0.9717) / 0.9719 | (0.9718) / 0.9697 | (0.9714) / 0.9707
| BERTurk uncased (128k) | (0.9716) / 0.9726 | (0.9715) / 0.9710 | (0.9704) / 0.9720 | (0.9715) / 0.9702 | (0.9704) / 0.9693 | (0.9711) / 0.9710
| DistilBERTurk          | (0.9648) / 0.9654 | (0.9649) / 0.9642 | (0.9654) / 0.9660 | (0.9646) / 0.9650 | (0.9637) / 0.9642 | (0.9646) / 0.9650
| XLM-RoBERTa            | (0.9611) / 0.9620 | (0.9629) / 0.9623 | (0.9617) / 0.9602 | (0.9602) / 0.9618 | (0.9614) / 0.9629 | (0.9614) / 0.9619

## NER

NER dataset is similar to the one used in [this paper](https://www.aclweb.org/anthology/P11-3019/).
We converted the dataset into CoNLL-like format and used a 80/10/10 training, development and test split.
Result on development set is reported in brackets.

| Model                  | Run 1             | Run 2             | Run 3             | Run 4             | Run 5             | Avg.
| ---------------------- | ----------------- | ----------------- | ----------------- | ----------------- | ----------------- | ---------------------
| ELECTRA small          | (0.9447) / 0.9468 | (0.9421) / 0.9439 | (0.9421) / 0.9471 | (0.9428) / 0.9434 | (0.9439) / 0.9447 | (0.9431) / 0.9452
| ELECTRA base           | (0.9564) / 0.9566 | (0.9552) / 0.9557 | (0.9579) / 0.9567 | (0.9563) / 0.9570 | (0.9568) / 0.9577 | (0.9565) / **0.9567**
| mBERT                  | (0.9441) / 0.9420 | (0.9448) / 0.9421 | (0.9439) / 0.9421 | (0.9444) / 0.9421 | (0.9434) / 0.9436 | (0.9441) / 0.9424
| BERTurk (32k)          | (0.9574) / 0.9550 | (0.9534) / 0.9552 | (0.9539) / 0.9570 | (0.9550) / 0.9543 | (0.9594) / 0.9531 | (0.9558) / 0.9549
| BERTurk (128k)         | (0.9479) / 0.9494 | (0.9569) / 0.9599 | (0.9546) / 0.9571 | (0.9549) / 0.9579 | (0.9557) / 0.9534 | (0.9540) / 0.9555
| BERTurk uncased (32k)  | (0.9529) / 0.9511 | (0.9531) / 0.9520 | (0.9533) / 0.9543 | (0.9530) / 0.9522 | (0.9523) / 0.9511 | (0.9529) / 0.9521
| BERTurk uncased (128k) | (0.9512) / 0.9531 | (0.9502) / 0.9518 | (0.9517) / 0.9520 | (0.9513) / 0.9525 | (0.9530) / 0.9546 | (0.9515) / 0.9528
| DistilBERTurk          | (0.9418) / 0.9392 | (0.9411) / 0.9415 | (0.9382) / 0.9400 | (0.9411) / 0.9427 | (0.9417) / 0.9427 | (0.9408) / 0.9412
| XLM-RoBERTa            | (0.9536) / 0.9541 | (0.9517) / 0.9521 | (0.9527) / 0.9530 | (0.9493) / 0.9530 | (0.9529) / 0.9516 | (0.9520) / 0.9527

# Citation

You can use the following BibTeX entry for citation:

```bibtex
@software{stefan_schweter_2020_3770924,
  author       = {Stefan Schweter},
  title        = {BERTurk - BERT models for Turkish},
  month        = apr,
  year         = 2020,
  publisher    = {Zenodo},
  version      = {1.0.0},
  doi          = {10.5281/zenodo.3770924},
  url          = {https://doi.org/10.5281/zenodo.3770924}
}
```

# Acknowledgments

Thanks to [Kemal Oflazer](http://www.andrew.cmu.edu/user/ko/) for providing us
additional large corpora for Turkish. Many thanks to Reyyan Yeniterzi for providing
us the Turkish NER dataset for evaluation.

Research supported with Cloud TPUs from Google's TensorFlow Research Cloud (TFRC).
Thanks for providing access to the TFRC ❤️
