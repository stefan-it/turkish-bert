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
