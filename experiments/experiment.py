import logging

from dataclasses import dataclass

from flair import set_seed

from flair.data import MultiCorpus
from flair.datasets import  NER_MULTI_XTREME
from flair.embeddings import TransformerWordEmbeddings, TransformerDocumentEmbeddings
from flair.models import SequenceTagger, TextClassifier
from flair.trainers import ModelTrainer
from flair.trainers.plugins.loggers.tensorboard import TensorboardLogger

from pathlib import Path

from typing import List

from offenseval2020_tr_dataset import OFFENSEVAL_TR_2020
from ud_datasets import UD_GENERIC


logger = logging.getLogger("flair")
logger.setLevel(level="INFO")


@dataclass
class ExperimentConfiguration:
    batch_size: int
    learning_rate: float
    epoch: int
    context_size: int
    seed: int
    base_model: str
    base_model_short: str
    task: str
    datasets: List[str]
    layers: str = "-1"
    subtoken_pooling: str = "first"
    use_crf: bool = False
    use_tensorboard: bool = True


def run_experiment_text_classification(experiment_configuration: ExperimentConfiguration) -> str:
    set_seed(experiment_configuration.seed)

    if experiment_configuration.task == "sentiment":
        corpus = OFFENSEVAL_TR_2020()
        label_type = "class"

    label_dict = corpus.make_label_dictionary(label_type=label_type)

    document_embeddings = TransformerDocumentEmbeddings(experiment_configuration.base_model, fine_tune=True)

    classifier = TextClassifier(document_embeddings, label_dictionary=label_dict, label_type=label_type)

    trainer = ModelTrainer(classifier, corpus)

    output_path_parts = [
        "flair",
        experiment_configuration.task,
        experiment_configuration.base_model_short,
        f"bs{experiment_configuration.batch_size}",
        f"e{experiment_configuration.epoch}",
        f"lr{experiment_configuration.learning_rate}",
        str(experiment_configuration.seed)
    ]

    output_path = "-".join(output_path_parts)

    plugins = []

    if experiment_configuration.use_tensorboard:
        logger.info("TensorBoard logging is enabled")

        tb_path = Path(f"{output_path}/runs")
        tb_path.mkdir(parents=True, exist_ok=True)

        plugins.append(TensorboardLogger(log_dir=str(tb_path), comment=output_path))

    trainer.fine_tune(
        output_path,
        reduce_transformer_vocab=False,  # set this to False for slow version
        mini_batch_size=experiment_configuration.batch_size,
        max_epochs=experiment_configuration.epoch,
        learning_rate=experiment_configuration.learning_rate,
        main_evaluation_metric=("macro avg", "f1-score"),
        use_final_model_for_eval=False,
    )

    # Finally, print model card for information
    classifier.print_model_card()

    return output_path

def run_experiment_token_classification(experiment_configuration: ExperimentConfiguration) -> str:
    set_seed(experiment_configuration.seed)

    corpora = []

    label_type = experiment_configuration.task

    if experiment_configuration.task == "pos":
        label_type = "upos"

        for dataset in experiment_configuration.datasets:
            # E.g. UD_Turkish-Atis/tr_atis@765b19c20edd89124d2a295fc8b6a330bfd8cdc2
            ud_dataset, ud_dataset_prefix = dataset.split("/")
            ud_dataset_prefix, ud_dataset_revision = ud_dataset_prefix.split("@")
            corpora.append(
                UD_GENERIC(ud_name=ud_dataset, ud_dataset_prefix=ud_dataset_prefix, revision=ud_dataset_revision)
            )
    elif experiment_configuration.task == "ner":
        label_type = "ner"

        for dataset in experiment_configuration.datasets:
            # E.g. xtreme/tr
            loader, language = dataset.split("/")
            if loader == "xtreme":
                corpora.append(
                    NER_MULTI_XTREME(languages=language)
                )

    corpora: MultiCorpus = MultiCorpus(corpora=corpora, sample_missing_splits=False)

    label_dictionary = corpora.make_label_dictionary(label_type=label_type)
    logger.info("Label Dictionary: {}".format(label_dictionary.get_items()))

    embeddings = TransformerWordEmbeddings(
        model=experiment_configuration.base_model,
        layers=experiment_configuration.layers,
        subtoken_pooling=experiment_configuration.subtoken_pooling,
        fine_tune=True,
        use_context=experiment_configuration.context_size,
    )

    tagger = SequenceTagger(
        hidden_size=256,
        embeddings=embeddings,
        tag_dictionary=label_dictionary,
        tag_type=label_type,
        use_crf=experiment_configuration.use_crf,
        use_rnn=False,
        reproject_embeddings=False,
    )

    trainer = ModelTrainer(tagger, corpora)

    output_path_parts = [
        "flair",
        experiment_configuration.task,
        experiment_configuration.base_model_short,
        f"bs{experiment_configuration.batch_size}",
        f"e{experiment_configuration.epoch}",
        f"lr{experiment_configuration.learning_rate}",
        str(experiment_configuration.seed)
    ]

    output_path = "-".join(output_path_parts)

    plugins = []

    if experiment_configuration.use_tensorboard:
        logger.info("TensorBoard logging is enabled")

        tb_path = Path(f"{output_path}/runs")
        tb_path.mkdir(parents=True, exist_ok=True)

        plugins.append(TensorboardLogger(log_dir=str(tb_path), comment=output_path))

    trainer.fine_tune(
        output_path,
        learning_rate=experiment_configuration.learning_rate,
        mini_batch_size=experiment_configuration.batch_size,
        max_epochs=experiment_configuration.epoch,
        shuffle=True,
        embeddings_storage_mode='none',
        weight_decay=0.,
        use_final_model_for_eval=False,
        plugins=plugins,
    )

    # Finally, print model card for information
    tagger.print_model_card()

    return output_path

def run_experiment(experiment_configuration: ExperimentConfiguration) -> str:
    if experiment_configuration.task in ["pos", "ner"]:
        return run_experiment_token_classification(experiment_configuration)
    elif experiment_configuration.task in ["sentiment"]:
        return run_experiment_text_classification(experiment_configuration)
