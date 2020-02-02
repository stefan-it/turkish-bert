# 🇹🇷 BERT

We present cased and uncased models BERT models for Turkish 🎉

**Note**: the model evaluation and training is currently in progress,
so the results here could change quite quickly.

# Stats

The current version of the model was trained on a filtered and sentence
segmented version of the Turkish [OSCAR corpus](https://traces1.inria.fr/oscar/).

The final training corpus has a size of 26GB and 32,50,152,191 tokens.

# Evaluation

## PoS tagging

The Turkish [IMST dataset](https://github.com/UniversalDependencies/UD_Turkish-IMST) 
from Universal Dependencies is used for evaluation. We use the `dev` branch and
commit `a6c955`.

The `run_ner` script with default parameters (`batch_size` is set to 16) from the
awesome [Transformers](https://github.com/huggingface/transformers) library is used
for training the PoS tagging models. Averaged F-score over 5 runs is reported 
(each run with a differend `seed`), score on development set in brackets:

| Model                | Run 1           | Run 2           | Run 3           | Run 4           | Run 5           | Avg.
| -------------------- | --------------- | --------------- | --------------- | --------------- | --------------- | -------------------
| mBERT (base, cased)  | (94.09) / 93.95 | (94.40) / 94.42 | (94.19) / 94.09 | (94.26) / 94.31 | (94.49) / 94.35 | (94.29) / 94.22
| XLM-R (large, cased) | (95.72) / 95.89 | (96.11) / 96.02 | (95.95) / 96.20 | (96.02) / 96.14 | (96.07) / 95.87 | (95.97) / 96.02
| TrBERT (base, cased) | (96.09) / 96.28 | (95.86) / 96.18 | (96.10) / 96.20 | (95.94) / 96.03 | (95.83) / 96.15 | (95.96) / **96.17**

# Changelog

* 02.02.2020: Initial version of this repo.