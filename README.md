# 🇹🇷 BERTurk

We present community-driven cased and uncased models BERT models for Turkish 🎉

Some datasets used for pretraining and evaluation are contributed from the
awesome Turkish NLP community, as well as the decision for the model name: BERTurk.

# Changelog

* 10.03.2020: Release of the cased distilled *BERTurk* model: *DistilBERTurk*.
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

## *DistilBERTurk*

The distilled version of a cased model, so called *DistilBERTurk*, was trained
on 7GB of the original training data, using the cased version of *BERTurk*
as teacher model.

*DistilBERTurk* was trained with the official Hugging Face implementation from
[here](https://github.com/huggingface/transformers/tree/master/examples/distillation).

The cased model was trained for 5 days on 4 RTX 2080 TI.

More details about distillation can be found in the
["DistilBERT, a distilled version of BERT: smaller, faster, cheaper and lighter"](https://arxiv.org/abs/1910.01108)
paper by Sanh et al. (2019).

# Evaluation

We use [FARM](https://github.com/deepset-ai/FARM) for evaluation on both PoS and NER datasets.
All configuration files can be found in the `configs` folder of this repository. We report
averaged Accuracy (PoS tagging) and F-Score (NER) on 5 runs (initialized with 5 different seeds).

We evaluated 5 different checkpoints for our cased and uncased models based on the development
score for PoS tagging and NER. The model with the best results is used for the final and released
model.

## PoS tagging

The Turkish [IMST dataset](https://github.com/UniversalDependencies/UD_Turkish-IMST) 
from Universal Dependencies is used for PoS tagging evaluation. We use the `dev` branch and
commit `a6c955`. Result on development set is reported in brackets.


| Model                                  | Run 1           | Run 2           | Run 3           | Run 4           | Run 5           | Avg.
| -------------------------------------- | --------------- | --------------- | --------------- | --------------- | --------------- | -------------------
| mBERT (base, cased)                    | (95.20) / 95.55 | (95.28) / 95.16 | (95.41) / 95.52 | (95.19) / 95.41 | (95.17) / 95.28 | (95.25) / 95.38
| XLM-R (large, cased)                   | (94.88) / 95.27 | (95.12) / 95.37 | (95.01) / 95.38 | (94.98) / 95.64 | (95.44) / 95.36 | (95.09) / 95.40
| BERTurk, 1.9M steps (base, cased)      | (96.82) / 97.05 | (96.96) / 96.81 | (96.89) / 96.88 | (96.95) / 97.06 | (96.76) / 96.84 | (96.88) / **96.93**
| DistilBERTurk (base, cased)            | (96.19) / 96.14 | (96.11) / 96.19 | (96.13) / 96.44 | (96.18) / 96.18 | (96.08) / 96.26 | (96.14) / 96.24

## NER

NER dataset is similar to the one used in [this paper](https://www.aclweb.org/anthology/P11-3019/).
We converted the dataset into CoNLL-like format and used a 80/10/10 training, development and test split.
 Result on development set is reported in brackets.

| Model                                  | Run 1           | Run 2           | Run 3           | Run 4           | Run 5           | Avg.
| -------------------------------------- | --------------- | --------------- | --------------- | --------------- | --------------- | -------------------
| mBERT (base, cased)                    | (93.51) / 93.79 | (93.63) / 93.44 | (94.11) / 93.67 | (93.95) / 93.40 | (94.08) / 93.76 | (93.86) / 93.61
| XLM-R (large, cased)                   | (94.86) / 94.51 | (94.79) / 94.08 | (94.57) / 94.32 | (94.91) / 94.09 | (94.97) / 94.47 | (94.82) / 94.29
| BERTurk, 1.9M steps (base, cased)      | (95.12) / 94.80 | (95.07) / 95.00 | (95.33) / 94.69 | (95.03) / 94.87 | (95.22) / 94.91 | (95.15) / **94.85**
| DistilBERTurk (base, cased)            | (99.56) / 93.26 | (92.01) / 93.26 | (88.15) / 93.04 | (92.50) / 92.97 | (91.20) / 93.30 | (92.68) / 93.17

# Acknowledgments

Thanks to [Kemal Oflazer](http://www.andrew.cmu.edu/user/ko/) for providing us
additional large corpora for Turkish. Many thanks to Reyyan Yeniterzi for providing
us the Turkish NER dataset for evaluation.

Research supported with Cloud TPUs from Google's TensorFlow Research Cloud (TFRC).
Thanks for providing access to the TFRC ❤️
