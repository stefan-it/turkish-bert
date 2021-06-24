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

# mC4 dataset

We use the Turkish part of the multilingual C4 dataset, that was recently released by the awesome AI2 team.

To download only the Turkish part of the dataset, just run:

```bash
GIT_LFS_SKIP_SMUDGE=1 git clone https://huggingface.co/datasets/allenai/c4
cd c4
git lfs pull --include "multilingual/c4-tr.*.json.gz"
```

We then extract the archive, parse the JSONL-like format, fix encoding errors and perform sentence
splitting - in just one script (`tr-extractor.py`):

```python
import json
import gzip
import sys

from nltk.tokenize import sent_tokenize

filename = sys.argv[1]
out_filename = sys.argv[2]

f_out = open(out_filename, "wt")

with gzip.open(filename, "rt") as f_p:
    for line in f_p:
        data = json.loads(line)

        text = data["text"]

        if "text" in data and "�" not in data["text"]:
            for sentence in sent_tokenize(text, "turkish"):
                f_out.write(sentence + "\n")
```

We use NLTK as sentence splitter, because it is way faster than spacy.

You can run this script in a "real" multi-threaded scenario, using `find` and `xargs`:

```bash
find . -iname "c4-tr.tfrecord*" -type f | xargs -I % -P 32 -n 1 python3 tr-extractor.py % ../%.extracted
```

Then the `*.extracted` files can be moved into a `corpus` folder in order to use it with the
`build_pretraining_dataset.py` script as shown in the previous sections of this readme.

The model was trained on a v3-32 TPU with a sequence length of 512 for 1M steps. Training took
around 3.5 days.
