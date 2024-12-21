# Evaluation

For evaluation we use latest Flair 0.8.1 version with a fine-tuning approach for PoS Tagging and NER downstream tasks. In order
to evaluate models on a Turkish question answering dataset, we use the [question answering example](https://github.com/huggingface/transformers/tree/master/examples/question-answering)
from the awesome 🤗 Transformers library.

We use the following hyperparameters for training PoS and NER models with Flair:

| Parameter       | Value
| --------------- | -----
| `batch_size`    | 16
| `learning_rate` | 5e-5
| `num_epochs`    | 10

For the question answering task, we use the same hyperparameters as used in the ["How Good Is Your Tokenizer?"](https://arxiv.org/abs/2012.15613)
paper.

The script `train_flert_model.py` in this repository can be used to fine-tune models on PoS Tagging an NER datasets.

We pre-train models with 5 different seeds and reported averaged accuracy (PoS tagging), F1-score (NER) or EM/F1 (Question answering).

For some downstream tasks, we perform "Almost Stochastic Order" tests as proposed in the
["Deep Dominance - How to Properly Compare Deep Neural Models"](https://www.aclweb.org/anthology/P19-1266/) paper.
The heatmap figures are heavily inspired by the ["CharacterBERT"](https://arxiv.org/abs/2010.10392) paper.

## PoS tagging

We use two different PoS Tagging datasets for Turkish from the Universal Dependencies project:

* [IMST dataset](https://github.com/UniversalDependencies/UD_Turkish-IMST) 
* [BOUN dataset](https://github.com/UniversalDependencies/UD_Turkish-BOUN)

We use the `dev` branch for training/dev/test splits.

### Evaluation on IMST dataset

| Model                      | Development Accuracy | Test Accuracy
| -------------------------- | -------------------- | -------------
| BERTurk (cased, 128k)      | 96.614 ± 0.58        | 96.846 ± 0.42
| BERTurk (cased, 32k)       | 97.138 ± 0.18        | 97.096 ± 0.07
| BERTurk (uncased, 128k)    | 96.964 ± 0.11        | 97.060 ± 0.07
| BERTurk (uncased, 32k)     | 97.080 ± 0.05        | 97.088 ± 0.05
| ConvBERTurk                | 97.208 ± 0.10        | 97.346 ± 0.07
| ConvBERTurk mC4 (cased)    | 97.148 ± 0.07        | 97.426 ± 0.03
| ConvBERTurk mC4 (uncased)  | 97.308 ± 0.09        | 97.338 ± 0.08
| DistilBERTurk              | 96.362 ± 0.05        | 96.560 ± 0.05
| ELECTRA Base               | 97.122 ± 0.06        | 97.232 ± 0.09
| ELECTRA Base mC4 (cased)   | 97.166 ± 0.07        | 97.380 ± 0.05
| ELECTRA Base mC4 (uncased) | 97.058 ± 0.12        | 97.210 ± 0.11
| ELECTRA Small              | 95.196 ± 0.09        | 95.578 ± 0.10
| XLM-R (base)               | 96.618 ± 0.10        | 96.492 ± 0.06
| mBERT (cased)              | 95.504 ± 0.10        | 95.754 ± 0.05

![UD IMST Development Results - PoS tagging](figures/ud_imst_dev.png)

![UD IMST Test Results - PoS tagging](figures/ud_imst_test.png)

Almost Stochastic Order tests (using the default alpha of 0.05) on test set:

![UD IMST Almost Stochastic Order tests - Test set](figures/ud_imst_asd.png)

### Evaluation on BOUN dataset

| Model                      | Development Accuracy | Test Accuracy
| -------------------------- | -------------------- | -------------
| BERTurk (cased, 128k)      | 90.828 ± 0.71        | 91.016 ± 0.60
| BERTurk (cased, 32k)       | 91.460 ± 0.10        | 91.490 ± 0.10
| BERTurk (uncased, 128k)    | 91.010 ± 0.15        | 91.286 ± 0.09
| BERTurk (uncased, 32k)     | 91.322 ± 0.19        | 91.544 ± 0.09
| ConvBERTurk                | 91.250 ± 0.14        | 91.524 ± 0.07
| ConvBERTurk mC4 (cased)    | 91.552 ± 0.10        | 91.724 ± 0.07
| ConvBERTurk mC4 (uncased)  | 91.202 ± 0.16        | 91.484 ± 0.12
| DistilBERTurk              | 91.166 ± 0.10        | 91.044 ± 0.09
| ELECTRA Base               | 91.354 ± 0.04        | 91.534 ± 0.11
| ELECTRA Base mC4 (cased)   | 91.402 ± 0.14        | 91.746 ± 0.11
| ELECTRA Base mC4 (uncased) | 91.100 ± 0.13        | 91.178 ± 0.15
| ELECTRA Small              | 91.020 ± 0.11        | 90.850 ± 0.12
| XLM-R (base)               | 91.828 ± 0.08        | 91.862 ± 0.16
| mBERT (cased)              | 91.286 ± 0.07        | 91.492 ± 0.11

![UD BOUN Development Results - PoS tagging](figures/ud_boun_dev.png)

![UD BOUN Test Results - PoS tagging](figures/ud_boun_test.png)

## NER

We use the Turkish dataset split from the [XTREME Benchmark](https://arxiv.org/abs/2003.11080).

These training/dev/split were introduced in the ["Massively Multilingual Transfer for NER"](https://arxiv.org/abs/1902.00193)
paper and are based on the famous WikiANN dataset, that is presentend in the
["Cross-lingual Name Tagging and Linking for 282 Languages"](https://www.aclweb.org/anthology/P17-1178/) paper.

| Model                      | Development F1-score | Test F1-score
| -------------------------- | -------------------- | -------------
| BERTurk (cased, 128k)      | 93.796 ± 0.07        | 93.8960 ± 0.16
| BERTurk (cased, 32k)       | 93.470 ± 0.11        | 93.4706 ± 0.09
| BERTurk (uncased, 128k)    | 93.604 ± 0.12        | 93.4686 ± 0.08
| BERTurk (uncased, 32k)     | 92.962 ± 0.08        | 92.9086 ± 0.14
| ConvBERTurk                | 93.822 ± 0.14        | 93.9286 ± 0.07
| ConvBERTurk mC4 (cased)    | 93.778 ± 0.15        | 93.6426 ± 0.15
| ConvBERTurk mC4 (uncased)  | 93.586 ± 0.07        | 93.6206 ± 0.13
| DistilBERTurk              | 92.012 ± 0.09        | 91.5966 ± 0.06
| ELECTRA Base               | 93.572 ± 0.08        | 93.4826 ± 0.17
| ELECTRA Base mC4 (cased)   | 93.600 ± 0.13        | 93.6066 ± 0.12
| ELECTRA Base mC4 (uncased) | 93.092 ± 0.15        | 92.8606 ± 0.36
| ELECTRA Small              | 91.278 ± 0.08        | 90.8306 ± 0.09
| XLM-R (base)               | 92.986 ± 0.05        | 92.9586 ± 0.14
| mBERT (cased)              | 93.308 ± 0.09        | 93.2306 ± 0.07

![XTREME Development Results - NER](figures/xtreme_dev.png)

![XTREME Test Results - NER](figures/xtreme_test.png)

## Question Answering

We use the Turkish Question Answering dataset from [this website](https://tquad.github.io/turkish-nlp-qa-dataset/)
and report EM and F1-Score on the development set (as reported from Transformers).

| Model                      | Development EM       | Development F1-score
| -------------------------- | -------------------- | -------------
| BERTurk (cased, 128k)      | 60.38 ± 0.61         | 78.21 ± 0.24
| BERTurk (cased, 32k)       | 58.79 ± 0.81         | 76.70 ± 1.04
| BERTurk (uncased, 128k)    | 59.60 ± 1.02         | 77.24 ± 0.59
| BERTurk (uncased, 32k)     | 58.92 ± 1.06         | 76.22 ± 0.42
| ConvBERTurk                | 60.11 ± 0.72         | 77.64 ± 0.59
| ConvBERTurk mC4 (cased)    | 60.65 ± 0.51         | 78.06 ± 0.34
| ConvBERTurk mC4 (uncased)  | 61.28 ± 1.27         | 78.63 ± 0.96
| DistilBERTurk              | 43.52 ± 1.63         | 62.56 ± 1.44
| ELECTRA Base               | 59.24 ± 0.70         | 77.70 ± 0.51
| ELECTRA Base mC4 (cased)   | 61.28 ± 0.94         | 78.17 ± 0.33
| ELECTRA Base mC4 (uncased) | 59.28 ± 0.87         | 76.88 ± 0.61
| ELECTRA Small              | 38.05 ± 1.83         | 57.79 ± 1.22
| XLM-R (base)               | 58.27 ± 0.53         | 76.80 ± 0.39
| mBERT (cased)              | 56.70 ± 0.43         | 75.20 ± 0.61

![TSQuAD Development Results EM - Question Answering](figures/tsquad_em.png)

![TSQuAD Development Results F1 - Question Answering](figures/tsquad_f1.png)
