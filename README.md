# 🇹🇷 BERT

We present cased and uncased models BERT models for Turkish 🎉

**Note**: the model evaluation and training is currently in progress,
so the results here could change quite quickly.

# Changelog

* 10.02.2020: Training corpus update, new TensorBoard links, new results for cased model
* 02.02.2020: Initial version of this repo.

# Stats

The current version of the model is trained on a filtered and sentence
segmented version of the Turkish [OSCAR corpus](https://traces1.inria.fr/oscar/),
a recent Wikipedia dump, various [OPUS corpora](http://opus.nlpl.eu/) and a
special corpus provided by [Kemal Oflazer](http://www.andrew.cmu.edu/user/ko/).

The final training corpus has a size of 35GB and 44,04,976,662 tokens.

Thanks to Google's TensorFlow Research Cloud (TFRC) we can train both cased and
uncased models on a TPU v3-8. You can find the current TensorBoard outputs for
the training (currently running):

* [TensorBoard cased model](https://tensorboard.dev/experiment/ZgFk8LclQOKdW0pYWviLMg/)
* [TensorBoard uncased model](https://tensorboard.dev/experiment/5LlD11cWRwexyqKSEPPXGA/)

# Evaluation

## PoS tagging

The Turkish [IMST dataset](https://github.com/UniversalDependencies/UD_Turkish-IMST) 
from Universal Dependencies is used for evaluation. We use the `dev` branch and
commit `a6c955`.

The `run_ner` script with default parameters (`batch_size` is set to 16) from the
awesome [Transformers](https://github.com/huggingface/transformers) library is used
for training the PoS tagging models. Averaged F-score over 5 runs is reported 
(each run with a differend `seed`), score on development set in brackets:

| Model                                  | Run 1           | Run 2           | Run 3           | Run 4           | Run 5           | Avg.
| -------------------------------------- | --------------- | --------------- | --------------- | --------------- | --------------- | -------------------
| mBERT (base, cased)                    | (94.09) / 93.95 | (94.40) / 94.42 | (94.19) / 94.09 | (94.26) / 94.31 | (94.49) / 94.35 | (94.29) / 94.22
| XLM-R (large, cased)                   | (95.72) / 95.89 | (96.11) / 96.02 | (95.95) / 96.20 | (96.02) / 96.14 | (96.07) / 95.87 | (95.97) / 96.02
| Turkish BERT, 1.4M steps (base, cased) | (95.83) / 96.16 | (96.05) / 96.22 | (96.15) / 96.25 | (96.19) / 96.49 | (95.72) / 96.23 | (95.99) / **96.27**

# Acknowledgments

Thanks to [Kemal Oflazer](http://www.andrew.cmu.edu/user/ko/) for providing us
additional large corpora for Turkish.

Research supported with Cloud TPUs from Google's TensorFlow Research Cloud (TFRC).
Thanks for providing access to the TFRC ❤️
