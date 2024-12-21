# Flair Fine-Tuning

The awesome Flair version is used for running all fine-tuning experiments.

All necessary dependencies (including a pinned Flair version for reproducability) can be installed via:

```bash
$ pip3 install -r requirement.txt
```

## YAML-based configuration

We use a YAML-based configuration approach for fine-tuning. The [`generate_configs.py`](configs/generate_configs.py)
script generated all necessary YAML configurations for all models and datasets.

In order to run a fine-tuning for a specific model and dataset, the corresponding YAML configuration file needs to be
set as environment variable, e.g. `export CONFIG=configs/distilberturk_cased/pos/uds.yaml`.

## Fine-Tuning

After the `CONFIG` variable is set, the fine-tuning can be started by running:

```bash
$ python3 fine_tuner.py
```

## Results

After fine-tuning, the benchmark results can be parsed with:

```bash
$ python3 flair-log-parser.py "flair-pos-distilberturk_cased-bs*"
```
