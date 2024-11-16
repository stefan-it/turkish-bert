from pathlib import Path

from omegaconf import OmegaConf


dataset_template = \
    """
batch_sizes:
  - 16
  - 8
learning_rates:
  - !!float 3e-5
  - !!float 5e-5
epochs:
  - 3
context_sizes:
  - 0
seeds:
  - 1
  - 2
  - 3
  - 4
  - 5
layers: "-1"
subword_poolings:
  - "first"
use_crf: !!bool false
use_tensorboard: !!bool true
cuda: "0"
"""

model_mapping = {
    "berturk_128k_cased":      "dbmdz/bert-base-turkish-128k-cased",
    "berturk_128k_uncased":    "dbmdz/bert-base-turkish-128k-uncased",
    "berturk_cased":           "dbmdz/bert-base-turkish-cased",
    "berturk_uncased":         "dbmdz/bert-base-turkish-uncased",
    "convbert_base_cased":     "dbmdz/convbert-base-turkish-cased",
    "convbert_base_mc4cased":  "dbmdz/convbert-base-turkish-mc4-uncased",
    "distilberturk_cased":     "dbmdz/distilbert-base-turkish-cased",
    "electra_base_cased":      "dbmdz/electra-base-turkish-cased-discriminator",
    "electra_base_mc4cased":   "dbmdz/electra-base-turkish-mc4-cased-discriminator",
    "electra_base_mc4uncased": "dbmdz/electra-base-turkish-mc4-uncased-discriminator",
    "electra_small_cased":     "dbmdz/electra-small-turkish-cased-discriminator",
}

pos_datasets = {
    "UD_Turkish-Atis":     ["tr_atis", "765b19c20edd89124d2a295fc8b6a330bfd8cdc2"],
    "UD_Turkish-BOUN":     ["tr_boun", "d1f2725233c17c188355004a252244c440db4c55"],
    "UD_Turkish-FrameNet": ["tr_framenet", "36a9b3f719cb26e0313f77238fc3e94128f57e3b"],
    "UD_Turkish-IMST":     ["tr_imst", "cc03792f9d725991deb95514d2309b7eb7b3945b"],
    "UD_Turkish-Tourism":  ["tr_tourism", "4f9b9898ee9d262920ff36535f1a8bf0ef60a7c0"],
}

ner_datasets = {
    "wikiann": "xtreme/tr",
}

for model_short_name, model_name in model_mapping.items():
    # Create model folder
    model_path = Path(f"./{model_short_name}")
    model_path.mkdir(parents=True, exist_ok=True)

    # Create subfolders for ner and pos
    for task in ["pos", "ner"]:
        task_path = model_path / task
        task_path.mkdir(parents=True, exist_ok=True)

        if task == "pos":
            concatenated_pos_datasets = []

            for pos_dataset_name, metadata in pos_datasets.items():
                dataset_short_name = metadata[0]
                dataset_revision = metadata[1]
                concatenated_pos_datasets.append(f"{pos_dataset_name}/{dataset_short_name}@{dataset_revision}")

            conf = OmegaConf.create(dataset_template)

            conf["task"] = "pos"
            conf["datasets"] = concatenated_pos_datasets
            conf["hf_model"] = model_name
            conf["model_short_name"] = model_short_name

            with open(str(task_path / "uds.yaml"), "wt") as f_out:
                OmegaConf.save(config=conf, f=f_out)
        elif task == "ner":
            for dataset_name, metadata in ner_datasets.items():
                conf["task"] = "ner"
                conf["datasets"] = [f"{metadata}"]
                conf["hf_model"] = model_name
                conf["model_short_name"] = model_short_name

                with open(str(task_path / f"{dataset_name}.yaml"), "wt") as f_out:
                    OmegaConf.save(config=conf, f=f_out)

    #with open(f"{config_name}.yaml", "wt") as f_out:
    #    f_out.write(config_template + "\n")
