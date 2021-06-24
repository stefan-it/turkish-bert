# Cheatsheet for training the 🇹🇷 ConvBERT model from scratch

We use the same pre-training data as for training the Turkish ELECTRA model.

The detailed procedure can be found in the ELECTRA
[cheatsheet](../electra/CHEATSHEET.md).

In [`configure_pretraining.py`](configure_pretraining.py) we provide the configuration file, that was
used for training the ConvBERT model using the official implementation. It
can simply be replaced with the original `configure_pretraining.py` script.

The model was trained for 1M steps with a sequence length of 512 with a
batch size of 256. Training was performed on a v3-32 TPU.
