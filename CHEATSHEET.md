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
