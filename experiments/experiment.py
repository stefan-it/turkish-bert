import logging

from dataclasses import dataclass

from flair import set_seed

from flair.datasets import NER_MULTI_XTREME
from flair.embeddings import TransformerWordEmbeddings
from flair.models import SequenceTagger
from flair.trainers import ModelTrainer
from flair.trainers.plugins.loggers.tensorboard import TensorboardLogger

from pathlib import Path

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
    dataset_name: str
    layers: str = "-1"
    subtoken_pooling: str = "first"
    use_crf: bool = False
    use_tensorboard: bool = True


def run_experiment(experiment_configuration: ExperimentConfiguration) -> str:
    set_seed(experiment_configuration.seed)

    task_name = experiment_configuration.dataset_name.split("/")[0]
    dataset_name = experiment_configuration.dataset_name.split("/")[-1]

    label_type = "ner"
    corpus = None

    if task_name == "pos":
        label_type = "upos"
    else:
        corpus = NER_MULTI_XTREME(languages="tr")

    label_dictionary = corpus.make_label_dictionary(label_type=label_type)
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

    trainer = ModelTrainer(tagger, corpus)

    output_path_parts = [
        "flair",
        task_name,
        dataset_name.replace("-", "_"),
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
