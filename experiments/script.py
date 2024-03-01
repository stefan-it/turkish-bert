import flair
import json
import os

from experiment import ExperimentConfiguration, run_experiment

from huggingface_hub import login, HfApi

from pathlib import Path

# Hugging Face Model Hub configuration
config_file     = os.environ.get("CONFIG")
hf_token        = os.environ.get("HF_TOKEN")
hf_hub_org_name = os.environ.get("HUB_ORG_NAME")
do_upload       = os.environ.get("HF_UPLOAD", False)

login(token=hf_token, add_to_git_credential=True)
api = HfApi()

with open(config_file, "rt") as f_p:
    json_config = json.load(f_p)

seeds            = json_config["seeds"]
batch_sizes      = json_config["batch_sizes"]
epochs           = json_config["epochs"]
learning_rates   = json_config["learning_rates"]
subword_poolings = json_config["subword_poolings"]
context_sizes    = json_config["context_sizes"]
dataset_name     = json_config["dataset_name"]
hf_model         = json_config["hf_model"]
model_short_name = json_config["model_short_name"]

cuda = json_config["cuda"]
flair.device = f'cuda:{cuda}'

for seed in seeds:
    for batch_size in batch_sizes:
        for epoch in epochs:
            for learning_rate in learning_rates:
                for subword_pooling in subword_poolings:
                    for context_size in context_sizes:
                        experiment_configuration = ExperimentConfiguration(
                            batch_size=batch_size,
                            learning_rate=learning_rate,
                            epoch=epoch,
                            context_size=context_size,
                            seed=seed,
                            subtoken_pooling=subword_pooling,
                            base_model=hf_model,
                            base_model_short=model_short_name,
                            dataset_name=dataset_name,
                        )
                        output_path = run_experiment(experiment_configuration=experiment_configuration)

                        if not do_upload:
                            continue

                        repo_url = api.create_repo(
                            repo_id=f"{hf_hub_org_name}/{output_path}",
                            token=hf_token,
                            private=True,
                            exist_ok=True,
                        )

                        if experiment_configuration.use_tensorboard:
                            api.upload_folder(
                                folder_path=f"{output_path}/runs",
                                path_in_repo="./runs",
                                repo_id=f"{hf_hub_org_name}/{output_path}",
                                repo_type="model"
                            )

                        best_model_test_path = Path(f"{output_path}/best-model.pt")
                        best_model_name = "best-model.pt"

                        if not best_model_test_path.exists():
                            # In some rare cases no best model was written (e.g. when F1-score is 0 for all epochs)
                            best_model_name = "final-model.pt"

                        api.upload_file(
                            path_or_fileobj=f"{output_path}/{best_model_name}",
                            path_in_repo="./pytorch_model.bin",
                            repo_id=f"{hf_hub_org_name}/{repo_name}",
                            repo_type="model"
                        )
                        api.upload_file(
                            path_or_fileobj=f"{output_path}/training.log",
                            path_in_repo="./training.log",
                            repo_id=f"{hf_hub_org_name}/{repo_name}",
                            repo_type="model"
                        )
