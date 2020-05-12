# Cheatsheet for training the 🇹🇷 ELECTRA model from scratch

## Preprocessing

Our ELECTRA models use the same vocab file as *BERTurk*. For a detailed description of how to generate
a BERT-like vocab, please refer to this [cheatsheet](https://github.com/stefan-it/turkish-bert/blob/master/CHEATSHEET.md).

In order to create the so called "pre-training" data for ELECTRA, only the vocab file and training corpus is needed.

We use the same training corpus as for *BERTurk*. We split the corpus into smaller chunks to enable better multi-processing.

After cloning the ELECTRA repository:

```bash
git clone https://github.com/google-research/electra.git
cd electra
```

The creation of the "pre-training" data can be started with:

```bash
python3 build_pretraining_dataset.py --corpus-dir ../corpus/ \
--vocab-file ../vocab.txt \
--output-dir ../output-512 --max-seq-length 512 \
--num-processes 20 --no-lower-case
```

The `corpus` folder contains 20 shards, so we can use 20 processes (specified via `--num-processes` option).
We use a sequence length of `512`, as we want to train an ELECTRA base model. When training an ELECTRA small model, you should
specify `128` here. We also want to train a cased model, so the `--no-lower-case` option is used here.

Pre-processing will take ~3 to 4 hours, depending on your hardware. After that you need to upload the `output-512` folder to your
Google Cloud Bucket. This can be done via:

```bash
gsutil -m -o GSUtil:parallel_composite_upload_threshold=150M cp -r output-512 gs://tr-electra
```

## Pre-training

All necessary hyper-parameters for pre-training need to be adjusted in the `configure_pretraining.py` script
in the ELECTRA repository.

We include both the configuration files for small and base ELECTRA models that we've used for training:

* ELECTRA small configuration: [here](configure_pretraining_small.py) and
* ELECTRA base configuration: [here](configure_pretraining_base.py)

Then pre-training can be started with:

```bash
python3 run_pretraining.py --data-dir gs://tr-electra --model-name electra-base-turkish-cased
```

We train both small and base models for 1M steps on a v3-8 TPU. Training a small model took ~9 hours,
base model took ~8 days.
