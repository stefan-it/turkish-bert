# Cheatsheet for training the 🇹🇷 BERT model from scratch

# Step 0: Collect corpora

For the 🇹🇷 BERT model we collect ~ 35GB text from various sorces like
[OPUS](http://opus.nlpl.eu/), Wikipedia, [Leipzig Corpora Collection](https://wortschatz.uni-leipzig.de/en/download)
or from the [OSCAR corpus](https://traces1.inria.fr/oscar/).

We split the training corpus into 1G shards using `split -C 1G`.

# Vocab generation

We use the awesome 🤗 [Tokenizers library](https://github.com/huggingface/tokenizers)
to create a BERT-compatible vocab.

The vocab is created on the complete training corpus (not just a single shard).

## Cased model

For the cased model we use the following snippet to generate the vocab:

```python
from tokenizers import BertWordPieceTokenizer

tokenizer = BertWordPieceTokenizer(
    clean_text=True, 
    handle_chinese_chars=False,
    strip_accents=False,
    lowercase=False, 
)

trainer = tokenizer.train( 
    "tr_final",
    vocab_size=32000,
    min_frequency=2,
    show_progress=True,
    special_tokens=['[PAD]', '[UNK]', '[CLS]', '[SEP]', '[MASK]'],
    limit_alphabet=1000,
    wordpieces_prefix="##"
)

tokenizer.save("./", "cased")
```

## Uncased model

For the uncased model we use the following snippet to generate the vocab:

```python
from tokenizers import BertWordPieceTokenizer

tokenizer = BertWordPieceTokenizer(
    clean_text=True,
    handle_chinese_chars=False,
    strip_accents=False,  # We need to investigate that further (stripping helps?)
    lowercase=True,
)

trainer = tokenizer.train(
    "tr_final",
    vocab_size=32000,
    min_frequency=2,
    show_progress=True,
    special_tokens=['[PAD]', '[UNK]', '[CLS]', '[SEP]', '[MASK]'],
    limit_alphabet=1000,
    wordpieces_prefix="##"
)

tokenizer.save("./", "cased")
```

# BERT preprocessing

In this step, the `create_pretraining_data.py` script from the
[BERT repo](https://github.com/google-research/bert) is used to create the
necessary input format (TFrecords) to train a model from scratch.

We need to clone the BERT repo first:

```bash
git clone https://github.com/google-research/bert.git
```

## Cased model

We did split our huge training corpus into smaller shards (1G per shard):

```bash
split -C 1G tr_final tr-
```

Then we move all shards into a separate folder:

```bash
mkdir cased_shards
mv tr-* cased_shards
```

The preprocessing step for each step will approx. consume 50-60GB of RAM and will
take 4-5 hours (depending on your machine). If you have a high memory machine,
you can parallelize this step using awesome `xargs` magic ;)

You can set the number of parallel processes with:

```bash
export NUM_PROC=5
```

Then you can start the preprocessing with:

```bash
cd bert # go to the BERT repo

find ../cased_shards -type f | xargs -I% -P $NUM_PROC -n 1 \
python3 create_pretraining_data.py --input_file % --output_file %.tfrecord \
--vocab_file ../cased-vocab.txt --do_lower_case=False -max_seq_length=512 \
--max_predictions_per_seq=75 --masked_lm_prob=0.15 --random_seed=12345 \
--dupe_factor=5
```

So in this example we use 5 parallel processes. Furthermore, we use a sequence
length of 512. You could start with a sequence length of 128, train the model
for a few steps and then fine-tune the model with a sequence length of 512.

# BERT Pretraining

## Uploading TFRecords

The previously created TFRecords are copied into a separate folder:

```bash
mkdir cased_tfrecords
mv cased_shards/*.tfrecord cased_tfrecords
```

Then this folder can be uploaded to a Google Storage Bucket using the `gsutil`
command:

```bash
gsutil -m -o GSUtil:parallel_composite_upload_threshold=150M cp -r cased_tfrecords gs://trbert
```

**Notice**: You must create a Google Storage Bucket first. Please also make
sure that the service user (e.g. `service-<id>@cloud-tpu.iam.gserviceaccount.com`)
has "Storage Administrator" permissions in order to write files to the bucket.

## TPU + VM instance

We use a v3-8 TPU from the Google's TensorFlow Research Cloud (TFRC). A TPU
instance can be created with:

```bash
gcloud compute tpus create bert --zone=<zone> --accelerator-type=v3-8 \
--network=default --range=192.168.1.0/29 --version=1.15
```

Please make sure, that you've set the correct `--zone` to avoid extra costs.

The following command is used to create a Google Cloud VM:

```bash
gcloud compute instances create bert --zone=<zone> --machine-type=n1-standard-2 \
--image-project=ml-images --image-family=tf-1-15 --scopes=cloud-platform
```

## SSH into VM

Just ssh into the previously created VM and open a `tmux` session:

```bash
gcloud compute ssh bert

# First login: takes a bit of time...
tmux
```

Clone the BERT repository (first time) and go to the BERT repo:

```
git clone https://github.com/google-research/bert.git
cd bert
```

## Config

The pretraining scripts needs a json-based configuration file with the correct
vocab size. We just use the original BERT base configuration file from the
Transformers library and adjusted the vocab size (32000 in our case):

```json
{
  "attention_probs_dropout_prob": 0.1,
  "hidden_act": "gelu",
  "hidden_dropout_prob": 0.1,
  "hidden_size": 768,
  "initializer_range": 0.02,
  "intermediate_size": 3072,
  "max_position_embeddings": 512,
  "num_attention_heads": 12,
  "num_hidden_layers": 12,
  "type_vocab_size": 2,
  "vocab_size": 32000
}
```

Store this configuration file to `config.json` in the `bert` repo folder.

## Pretraining

Then the pretraining command can be run to train a BERT model from scratch:

```bash
python3 run_pretraining.py --input_file=gs://trbert/cased_tfrecords/*.tfrecord \
--output_dir=gs://trbert/bert-base-turkish-cased --bert_config_file=config.json \
--max_seq_length=512 --max_predictions_per_seq=75 --do_train=True \
--train_batch_size=128 --num_train_steps=3000000 --learning_rate=1e-4 \
--save_checkpoints_steps=100000 --keep_checkpoint_max=20 --use_tpu=True \
--tpu_name=bert --num_tpu_cores=8
```

This will train the model for 3M steps. Checkpoints are saved after 100k steps.
The last 20 checkpoints will be kept.
