import flair
import json
import os

from experiment import ExperimentConfiguration, run_experiment

from huggingface_hub import login, HfApi

from omegaconf import OmegaConf

from pathlib import Path


# Main configuration
main_conf = OmegaConf.load('configs/main.yaml')

do_upload = main_conf.do_upload
hf_hub_org_name = main_conf.hf_hub_org_name

if do_upload:
    login()

api = HfApi()

# Experiment configuration
config_file     = os.environ.get("CONFIG")
conf = OmegaConf.load(config_file)

seeds              = conf.seeds
batch_sizes        = conf.batch_sizes
epochs             = conf.epochs
learning_rates     = conf.learning_rates
subword_poolings   = conf.subword_poolings
context_sizes      = conf.context_sizes
hf_model           = conf.hf_model
model_short_name   = conf.model_short_name
task               = conf.task
use_tensorboard    = conf.use_tensorboard

cuda = conf.cuda
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
                            task=task,
                            use_tensorboard=use_tensorboard,
                        )
                        output_path = run_experiment(experiment_configuration=experiment_configuration)

                        if not do_upload:
                            continue

                        repo_url = api.create_repo(
                            repo_id=f"{hf_hub_org_name}/{output_path}",
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
                            repo_id=f"{hf_hub_org_name}/{output_path}",
                            repo_type="model"
                        )
                        api.upload_file(
                            path_or_fileobj=f"{output_path}/training.log",
                            path_in_repo="./training.log",
                            repo_id=f"{hf_hub_org_name}/{output_path}",
                            repo_type="model"
                        )
