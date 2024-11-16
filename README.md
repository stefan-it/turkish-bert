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

* 1x.12.2024: New evaluations with Flair are added.
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

# Stats

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

| Model name                 | Model hub link                                                                      | Pre-training corpus size
| -------------------------- | ----------------------------------------------------------------------------------- | ------------------------
| ELECTRA Small (cased)      | [here](https://huggingface.co/dbmdz/electra-small-turkish-cased-discriminator)      | 35GB
| ELECTRA Base (cased)       | [here](https://huggingface.co/dbmdz/electra-base-turkish-cased-discriminator)       | 35GB
| ELECTRA Base mC4 (cased)   | [here](https://huggingface.co/dbmdz/electra-base-turkish-mc4-cased-discriminator)   | 242GB
| ELECTRA Base mC4 (uncased) | [here](https://huggingface.co/dbmdz/electra-base-turkish-mc4-uncased-discriminator) | 242GB
| BERTurk (cased, 32k)       | [here](https://huggingface.co/dbmdz/bert-base-turkish-cased)                        | 35GB
| BERTurk (uncased, 32k)     | [here](https://huggingface.co/dbmdz/bert-base-turkish-uncased)                      | 35GB
| BERTurk (cased, 128k)      | [here](https://huggingface.co/dbmdz/bert-base-turkish-128k-cased)                   | 35GB
| BERTurk (uncased, 128k)    | [here](https://huggingface.co/dbmdz/bert-base-turkish-128k-uncased)                 | 35GB
| DistilBERTurk (cased)      | [here](https://huggingface.co/dbmdz/distilbert-base-turkish-cased)                  | 35GB
| ConvBERTurk (cased)        | [here](https://huggingface.co/dbmdz/convbert-base-turkish-cased)                    | 35GB
| ConvBERTurk mC4 (cased)    | [here](https://huggingface.co/dbmdz/convbert-base-turkish-mc4-cased)                | 242GB
| ConvBERTurk mC4 (uncased)  | [here](https://huggingface.co/dbmdz/convbert-base-turkish-mc4-uncased)              | 242GB

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

# Evaluation

In 2024 we ran new evaluations on Pos Tagging and NER datasets. Prior evaluation results can be found [here](OLD_EVALUATIONS.md).

# Model usage

All trained models can be used from the [DBMDZ](https://github.com/dbmdz) Hugging Face [model hub page](https://huggingface.co/dbmdz)
using their model name. The following models are available:

* *BERTurk* models with 32k vocabulary: `dbmdz/bert-base-turkish-cased` and `dbmdz/bert-base-turkish-uncased`
* *BERTurk* models with 128k vocabulary: `dbmdz/bert-base-turkish-128k-cased` and `dbmdz/bert-base-turkish-128k-uncased`
* *ELECTRA* small and base cased models (discriminator): `dbmdz/electra-small-turkish-cased-discriminator` and `dbmdz/electra-base-turkish-cased-discriminator`
* *ELECTRA* base cased and uncased models, trained on Turkish part of mC4 corpus (discriminator): `dbmdz/electra-small-turkish-mc4-cased-discriminator` and `dbmdz/electra-small-turkish-mc4-uncased-discriminator`
* *ConvBERTurk* model with 32k vocabulary: `dbmdz/convbert-base-turkish-cased`
* *ConvBERTurk* base cased and uncased models, trained on Turkish part of mC4 corpus: `dbmdz/convbert-base-turkish-mc4-cased` and `dbmdz/convbert-base-turkish-mc4-uncased`

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

Research supported with Cloud TPUs from Google's TensorFlow Research Cloud (TFRC).
Thanks for providing access to the TFRC ❤️
