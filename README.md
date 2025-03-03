# 🇹🇷 BERTurk

<p align="center">
  <img alt="Logo provided by Merve Noyan" title="Awesome logo from Merve Noyan" src="https://raw.githubusercontent.com/stefan-it/turkish-bert/master/merve_logo.png">
</p>

[![DOI](https://zenodo.org/badge/237817454.svg)](https://zenodo.org/badge/latestdoi/237817454)

We present community-driven BERT, DistilBERT, ELECTRA and ConvBERT models for Turkish 🎉

Some datasets used for pretraining and evaluation are contributed from the
awesome Turkish NLP community, as well as the decision for the BERT model name: BERTurk.

Logo is provided by [Merve Noyan](https://twitter.com/mervenoyann).

# Changelog

* 21.12.2024: New evaluations with Flair are added.
* 23.09.2021: Release of uncased ELECTRA and ConvBERT models and cased ELECTRA model, all trained on mC4 corpus.
* 24.06.2021: Release of new ELECTRA model, trained on Turkish part of mC4 dataset. Repository got new awesome logo from Merve Noyan.
* 16.03.2021: Release of *ConvBERTurk* model and more evaluations on different downstream tasks.
* 12.05.2020: Release of ELEC**TR**A ([small](https://huggingface.co/dbmdz/electra-small-turkish-cased-discriminator)
              and [base](https://huggingface.co/dbmdz/electra-base-turkish-cased-discriminator)) models, see [here](electra/README.md).
* 25.03.2020: Release of *BERTurk* uncased model and *BERTurk* models with larger vocab size (128k, cased and uncased).
* 11.03.2020: Release of the cased distilled *BERTurk* model: *DistilBERTurk*.
              Available on the [Hugging Face model hub](https://huggingface.co/dbmdz/distilbert-base-turkish-cased)
* 17.02.2020: Release of the cased *BERTurk* model.
              Available on the [Hugging Face model hub](https://huggingface.co/dbmdz/bert-base-turkish-cased)
* 10.02.2020: Training corpus update, new TensorBoard links, new results for cased model.
* 02.02.2020: Initial version of this repo.

# Pretraining Corpora Stats

The current version of the model is trained on a filtered and sentence
segmented version of the Turkish [OSCAR corpus](https://traces1.inria.fr/oscar/),
a recent Wikipedia dump, various [OPUS corpora](http://opus.nlpl.eu/) and a
special corpus provided by [Kemal Oflazer](http://www.andrew.cmu.edu/user/ko/).

The final training corpus has a size of 35GB and 4,404,976,662 tokens.

Thanks to Google's TensorFlow Research Cloud (TFRC) we can train both cased and
uncased models on a TPU v3-8. You can find the TensorBoard outputs for
the training here:

* [TensorBoard cased model](https://tensorboard.dev/experiment/ZgFk8LclQOKdW0pYWviLMg/)
* [TensorBoard uncased model](https://tensorboard.dev/experiment/5LlD11cWRwexyqKSEPPXGA/)

We also provide cased and uncased models that aŕe using a larger vocab size (128k instead of 32k).

A detailed cheatsheet of how the models were trained, can be found [here](CHEATSHEET.md).

## C4 Multilingual dataset (mC4)

We've also trained an ELECTRA (cased) model on the recently released Turkish part of the
[multiligual C4 (mC4) corpus](https://github.com/allenai/allennlp/discussions/5265) from the AI2 team.

After filtering documents with a broken encoding, the training corpus has a size of 242GB resulting
in 31,240,963,926 tokens.

We used the original 32k vocab (instead of creating a new one).

# Turkish Model Zoo

Here's an overview of all available models, incl. their training corpus size:

| Model name                 | Model hub link                                                                      | Pre-training corpus size |
|----------------------------|-------------------------------------------------------------------------------------|--------------------------|
| ELECTRA Small (cased)      | [here](https://huggingface.co/dbmdz/electra-small-turkish-cased-discriminator)      | 35GB                     |
| ELECTRA Base (cased)       | [here](https://huggingface.co/dbmdz/electra-base-turkish-cased-discriminator)       | 35GB                     |
| ELECTRA Base mC4 (cased)   | [here](https://huggingface.co/dbmdz/electra-base-turkish-mc4-cased-discriminator)   | 242GB                    |
| ELECTRA Base mC4 (uncased) | [here](https://huggingface.co/dbmdz/electra-base-turkish-mc4-uncased-discriminator) | 242GB                    |
| BERTurk (cased, 32k)       | [here](https://huggingface.co/dbmdz/bert-base-turkish-cased)                        | 35GB                     |
| BERTurk (uncased, 32k)     | [here](https://huggingface.co/dbmdz/bert-base-turkish-uncased)                      | 35GB                     |
| BERTurk (cased, 128k)      | [here](https://huggingface.co/dbmdz/bert-base-turkish-128k-cased)                   | 35GB                     |
| BERTurk (uncased, 128k)    | [here](https://huggingface.co/dbmdz/bert-base-turkish-128k-uncased)                 | 35GB                     |
| DistilBERTurk (cased)      | [here](https://huggingface.co/dbmdz/distilbert-base-turkish-cased)                  | 35GB                     |
| ConvBERTurk (cased)        | [here](https://huggingface.co/dbmdz/convbert-base-turkish-cased)                    | 35GB                     |
| ConvBERTurk mC4 (cased)    | [here](https://huggingface.co/dbmdz/convbert-base-turkish-mc4-cased)                | 242GB                    |
| ConvBERTurk mC4 (uncased)  | [here](https://huggingface.co/dbmdz/convbert-base-turkish-mc4-uncased)              | 242GB                    |
| BERT5urk                   | [here](stefan-it/bert5urk)                                                          | 262GB                    |

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

In addition to the *BERTurk* models, we also trained ELEC**TR**A small and base models. A detailed overview can be found
in the [ELECTRA section](electra/README.md).

# ConvBERTurk

In addition to the BERT and ELECTRA based models, we also trained a ConvBERT model. The ConvBERT architecture is presented
in the ["ConvBERT: Improving BERT with Span-based Dynamic Convolution"](https://arxiv.org/abs/2008.02496) paper.

We follow a different training procedure: instead of using a two-phase approach, that pre-trains the model for 90% with 128
sequence length and 10% with 512 sequence length, we pre-train the model with 512 sequence length for 1M steps on a v3-32 TPU.

More details about the pre-training can be found [here](convbert/README.md).

# mC4 ELECTRA

In addition to the ELEC**TR**A base model, we also trained an ELECTRA model on the Turkish part of the mC4 corpus. We use a
sequence length of 512 over the full training time and train the model for 1M steps on a v3-32 TPU.

# BERT5urk

BERT5urk is a new 1.42B encoder-decoder model based on the [efficient](https://arxiv.org/abs/2109.10686) [T5 architecture](https://arxiv.org/abs/1910.10683) and
pretrained with the [UL2 objective](https://arxiv.org/abs/2205.05131).

The model was pretrained with the awesome [T5X](https://github.com/google-research/t5x) library for 2M steps with a
batch size of 128 and an input and output sequence length of 512 for 16.56 days on a v3-32 TPU Pod.

The Turkish part of the amazing [FineWeb2](https://huggingface.co/datasets/HuggingFaceFW/fineweb-2), filtered for a
minimum language score of 0.99 resulting in a 262GB pretraining corpus.

# Evaluation

In 2024 we ran new evaluations on PoS tagging, NER and sentiment classification datasets. Prior evaluation results can be found [here](OLD_EVALUATIONS.md).

All evaluations are performed with the awesome Flair library and the evaluation code and configs can be found in the
[`experiments](experiments) folder of this repository.

## PoS Tagging

The Model Zoo is evaluated on (the concatenation) of the following PoS Tagging datasets from Universal Dependencies:

* [Atis](https://github.com/UniversalDependencies/UD_Turkish-Atis)
* [BOUN](https://github.com/UniversalDependencies/UD_Turkish-BOUN)
* [FrameNet](https://github.com/UniversalDependencies/UD_Turkish-FrameNet)
* [IMST](https://github.com/UniversalDependencies/UD_Turkish-IMST)
* [Tourism](https://github.com/UniversalDependencies/UD_Turkish-Tourism)

We perform a hyper-parameter search over the following configurations:

| Parameter     | Values         |
|---------------|----------------|
| Batch Size    | `[16, 8]`      |
| Learning Rate | `[3e-5, 5e-5]` |
| Epoch         | `[3]`          |

And report averaged Accuracy over 5 runs (with different seeds):

| Model Name                                                                                                | Best Configuration | Best Development Score | Best Test Score |
|-----------------------------------------------------------------------------------------------------------|--------------------|-----------------------:|----------------:|
| [BERTurk (cased, 128k)](https://huggingface.co/dbmdz/bert-base-turkish-128k-cased)                        | `bs16-e3-lr5e-05`  |           93.93 ± 0.04 |    94.50 ± 0.07 |
| [BERTurk (uncased, 128k)](https://huggingface.co/dbmdz/bert-base-turkish-128k-uncased)                    | `bs8-e3-lr5e-05`   |           93.84 ± 0.04 |    94.41 ± 0.13 |
| [BERTurk (cased, 32k)](https://huggingface.co/dbmdz/bert-base-turkish-cased)                              | `bs16-e3-lr5e-05`  |           93.95 ± 0.05 |    94.57 ± 0.04 |
| [BERTurk (uncased, 32k)](https://huggingface.co/dbmdz/bert-base-turkish-uncased)                          | `bs16-e3-lr5e-05`  |           93.84 ± 0.04 |    94.38 ± 0.03 |
| [ConvBERTurk (cased)](https://huggingface.co/dbmdz/convbert-base-turkish-cased)                           | `bs8-e3-lr5e-05`   |           94.03 ± 0.07 |    94.58 ± 0.06 |
| [ConvBERTurk mC4 (cased)](https://huggingface.co/dbmdz/convbert-base-turkish-mc4-cased)                   | `bs8-e3-lr5e-05`   |       **94.04** ± 0.05 |    94.59 ± 0.06 |
| [ConvBERTurk mC4 (uncased)](https://huggingface.co/dbmdz/convbert-base-turkish-mc4-uncased)               | `bs8-e3-lr5e-05`   |           93.90 ± 0.08 |    94.52 ± 0.04 |
| [DistilBERTurk (cased)](https://huggingface.co/dbmdz/distilbert-base-turkish-cased)                       | `bs8-e3-lr5e-05`   |           93.52 ± 0.03 |    94.19 ± 0.04 |
| [ELECTRA Base (cased)](https://huggingface.co/dbmdz/electra-base-turkish-cased-discriminator)             | `bs16-e3-lr5e-05`  |           93.89 ± 0.05 |    94.45 ± 0.05 |
| [ELECTRA Base mC4 (cased)](https://huggingface.co/dbmdz/electra-base-turkish-mc4-cased-discriminator)     | `bs16-e3-lr5e-05`  |           93.88 ± 0.05 |    94.53 ± 0.11 |
| [ELECTRA Base mC4 (uncased)](https://huggingface.co/dbmdz/electra-base-turkish-mc4-uncased-discriminator) | `bs8-e3-lr5e-05`   |           93.80 ± 0.09 |    94.41 ± 0.04 |
| [ELECTRA Small (cased)](https://huggingface.co/dbmdz/electra-small-turkish-cased-discriminator)           | `bs8-e3-lr5e-05`   |           93.15 ± 0.04 |    93.88 ± 0.06 |
| [BERT5urk](https://huggingface.co/stefan-it/bert5urk)                                                     | `bs8-e3-lr5e-05`   |           93.75 ± 0.04 |    94.33 ± 0.06 |

## Named Entity Recognition

The Model Zoo is evaluated on the Turkish split of the WikiANN dataset, using the following hyper-parameter search:

| Parameter     | Values         |
|---------------|----------------|
| Batch Size    | `[16, 8]`      |
| Learning Rate | `[3e-5, 5e-5]` |
| Epoch         | `[10]`         |

Averaged F1-Score over 5 runs (with different seeds):

| Model Name                                                                                                | Best Configuration | Best Development Score | Best Test Score |
|-----------------------------------------------------------------------------------------------------------|--------------------|-----------------------:|----------------:|
| [BERTurk (cased, 128k)](https://huggingface.co/dbmdz/bert-base-turkish-128k-cased)                        | `bs8-e10-lr3e-05`  |           93.92 ± 0.07 |    93.92 ± 0.16 |
| [BERTurk (uncased, 128k)](https://huggingface.co/dbmdz/bert-base-turkish-128k-uncased)                    | `bs16-e10-lr3e-05` |           93.59 ± 0.05 |    93.29 ± 0.11 |
| [BERTurk (cased, 32k)](https://huggingface.co/dbmdz/bert-base-turkish-cased)                              | `bs8-e10-lr3e-05`  |           93.36 ± 0.04 |    93.26 ± 0.14 |
| [BERTurk (uncased, 32k)](https://huggingface.co/dbmdz/bert-base-turkish-uncased)                          | `bs8-e10-lr3e-05`  |           93.13 ± 0.19 |    92.96 ± 0.06 |
| [ConvBERTurk (cased)](https://huggingface.co/dbmdz/convbert-base-turkish-cased)                           | `bs8-e10-lr3e-05`  |       **93.93** ± 0.07 |    93.93 ± 0.05 |
| [ConvBERTurk mC4 (cased)](https://huggingface.co/dbmdz/convbert-base-turkish-mc4-cased)                   | `bs8-e10-lr3e-05`  |           93.89 ± 0.07 |    93.57 ± 0.06 |
| [ConvBERTurk mC4 (uncased)](https://huggingface.co/dbmdz/convbert-base-turkish-mc4-uncased)               | `bs8-e10-lr3e-05`  |           93.68 ± 0.13 |    93.58 ± 0.15 |
| [DistilBERTurk (cased)](https://huggingface.co/dbmdz/distilbert-base-turkish-cased)                       | `bs8-e10-lr5e-05`  |           91.80 ± 0.05 |    91.17 ± 0.03 |
| [ELECTRA Base (cased)](https://huggingface.co/dbmdz/electra-base-turkish-cased-discriminator)             | `bs8-e10-lr3e-05`  |           93.58 ± 0.12 |    93.60 ± 0.09 |
| [ELECTRA Base mC4 (cased)](https://huggingface.co/dbmdz/electra-base-turkish-mc4-cased-discriminator)     | `bs16-e10-lr3e-05` |           93.51 ± 0.09 |    93.42 ± 0.11 |
| [ELECTRA Base mC4 (uncased)](https://huggingface.co/dbmdz/electra-base-turkish-mc4-uncased-discriminator) | `bs16-e10-lr5e-05` |           93.01 ± 0.12 |    92.94 ± 0.13 |
| [ELECTRA Small (cased)](https://huggingface.co/dbmdz/electra-small-turkish-cased-discriminator)           | `bs8-e10-lr5e-05`  |           91.42 ± 0.09 |    91.07 ± 0.09 |
| [BERT5urk](https://huggingface.co/stefan-it/bert5urk)                                                     | `bs8-e10-lr5e-05`  |       **93.93** ± 0.10 |    93.66 ± 0.10 |

## Sentiment Classification

The Model Zoo is additionally evaluated on the [OffensEval-TR 2020](stefan-it/offenseval2020_tr) dataset for sentiment
classification.

The following parameters are used for a hyper-parameter search:

| Parameter     | Values         |
|---------------|----------------|
| Batch Size    | `[16, 8]`      |
| Learning Rate | `[3e-5, 5e-5]` |
| Epoch         | `[3]`          |

Averaged Macro F1-Score over 5 runs (with different seeds) is reported:

| Model Name                                                                                                | Best Configuration | Best Development Score | Best Test Score |
|-----------------------------------------------------------------------------------------------------------|--------------------|-----------------------:|----------------:|
| [BERTurk (cased, 128k)](https://huggingface.co/dbmdz/bert-base-turkish-128k-cased)                        | `bs16-e3-lr3e-05`  |           81.30 ± 0.61 |    81.72 ± 0.47 |
| [BERTurk (uncased, 128k)](https://huggingface.co/dbmdz/bert-base-turkish-128k-uncased)                    | `bs16-e3-lr3e-05`  |           80.31 ± 0.54 |    82.16 ± 0.27 |
| [BERTurk (cased, 32k)](https://huggingface.co/dbmdz/bert-base-turkish-cased)                              | `bs16-e3-lr5e-05`  |           79.64 ± 0.50 |    80.65 ± 0.40 |
| [BERTurk (uncased, 32k)](https://huggingface.co/dbmdz/bert-base-turkish-uncased)                          | `bs16-e3-lr3e-05`  |           80.87 ± 0.22 |    81.68 ± 0.37 |
| [ConvBERTurk (cased)](https://huggingface.co/dbmdz/convbert-base-turkish-cased)                           | `bs16-e3-lr3e-05`  |       **82.22** ± 0.41 |    82.29 ± 0.34 |
| [ConvBERTurk mC4 (cased)](https://huggingface.co/dbmdz/convbert-base-turkish-mc4-cased)                   | `bs16-e3-lr3e-05`  |           82.16 ± 0.46 |    82.10 ± 0.30 |
| [ConvBERTurk mC4 (uncased)](https://huggingface.co/dbmdz/convbert-base-turkish-mc4-uncased)               | `bs16-e3-lr3e-05`  |           81.69 ± 0.29 |    81.81 ± 0.37 |
| [DistilBERTurk (cased)](https://huggingface.co/dbmdz/distilbert-base-turkish-cased)                       | `bs16-e3-lr3e-05`  |           78.54 ± 0.55 |    79.12 ± 0.17 |
| [ELECTRA Base (cased)](https://huggingface.co/dbmdz/electra-base-turkish-cased-discriminator)             | `bs16-e3-lr3e-05`  |           79.76 ± 0.24 |    81.69 ± 0.38 |
| [ELECTRA Base mC4 (cased)](https://huggingface.co/dbmdz/electra-base-turkish-mc4-cased-discriminator)     | `bs8-e3-lr3e-05`   |           80.34 ± 0.67 |    82.14 ± 0.27 |
| [ELECTRA Base mC4 (uncased)](https://huggingface.co/dbmdz/electra-base-turkish-mc4-uncased-discriminator) | `bs16-e3-lr5e-05`  |           80.46 ± 0.80 |    81.52 ± 0.56 |
| [ELECTRA Small (cased)](https://huggingface.co/dbmdz/electra-small-turkish-cased-discriminator)           | `bs16-e3-lr5e-05`  |           77.25 ± 0.47 |    79.89 ± 0.28 |
| [BERT5urk](https://huggingface.co/stefan-it/bert5urk)                                                     | `bs8-e3-lr0.00015` |           82.20 ± 0.88 |    82.78 ± 0.44 |

## Overall

The following table shows the performance of all models over all datasets:

| Model Name                                                                                                | Overall Development | Overall Test |
|-----------------------------------------------------------------------------------------------------------|--------------------:|-------------:|
| [BERTurk (cased, 128k)](https://huggingface.co/dbmdz/bert-base-turkish-128k-cased)                        |               89.72 |        90.05 |
| [BERTurk (uncased, 128k)](https://huggingface.co/dbmdz/bert-base-turkish-128k-uncased)                    |               89.25 |        89.95 |
| [BERTurk (cased, 32k)](https://huggingface.co/dbmdz/bert-base-turkish-cased)                              |               88.98 |        89.49 |
| [BERTurk (uncased, 32k)](https://huggingface.co/dbmdz/bert-base-turkish-uncased)                          |               89.28 |        89.67 |
| [ConvBERTurk (cased)](https://huggingface.co/dbmdz/convbert-base-turkish-cased)                           |           **90.06** |        90.27 |
| [ConvBERTurk mC4 (cased)](https://huggingface.co/dbmdz/convbert-base-turkish-mc4-cased)                   |               90.03 |        90.09 |
| [ConvBERTurk mC4 (uncased)](https://huggingface.co/dbmdz/convbert-base-turkish-mc4-uncased)               |               89.76 |        89.97 |
| [DistilBERTurk (cased)](https://huggingface.co/dbmdz/distilbert-base-turkish-cased)                       |               87.95 |        88.16 |
| [ELECTRA Base (cased)](https://huggingface.co/dbmdz/electra-base-turkish-cased-discriminator)             |               89.08 |        89.91 |
| [ELECTRA Base mC4 (cased)](https://huggingface.co/dbmdz/electra-base-turkish-mc4-cased-discriminator)     |               89.24 |        90.03 |
| [ELECTRA Base mC4 (uncased)](https://huggingface.co/dbmdz/electra-base-turkish-mc4-uncased-discriminator) |               89.09 |        89.62 |
| [ELECTRA Small (cased)](https://huggingface.co/dbmdz/electra-small-turkish-cased-discriminator)           |               87.27 |        88.28 |
| [BERT5urk](https://huggingface.co/stefan-it/bert5urk)                                                     |               89.96 |        90.26 |

# Model usage

All trained models can be used from the [DBMDZ](https://github.com/dbmdz) Hugging Face [model hub page](https://huggingface.co/dbmdz)
using their model name.

Example usage with 🤗/Transformers:

```python
tokenizer = AutoTokenizer.from_pretrained("dbmdz/bert-base-turkish-cased")

model = AutoModel.from_pretrained("dbmdz/bert-base-turkish-cased")
```

This loads the *BERTurk* cased model. The recently introduced ELEC**TR**A base model can be loaded with:

```python
tokenizer = AutoTokenizer.from_pretrained("dbmdz/electra-base-turkish-cased-discriminator")

model = AutoModelWithLMHead.from_pretrained("dbmdz/electra-base-turkish-cased-discriminator")
```

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

We would like to thank [Merve Noyan](https://twitter.com/mervenoyann) for the
awesome logo!

Research supported with Cloud TPUs from the awesome [TRC program](https://sites.research.google/trc/about/).

Many thanks for providing access to the TPUs ❤️
