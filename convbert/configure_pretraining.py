# coding=utf-8

"""Config controlling hyperparameters for pre-training."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os


class PretrainingConfig(object):
  """Defines pre-training hyperparameters."""

  def __init__(self, model_name, data_dir, **kwargs):
    self.model_name = model_name
    self.debug = False  # debug mode
    self.do_train = True  # pre-train
    self.do_eval = False  # evaluate generator/discriminator on unlabeled data

    # loss functions
    self.electra_objective = True  # if False, use the BERT objective instead
    self.gen_weight = 1.0  # masked language modeling / generator loss
    self.disc_weight = 50.0  # discriminator loss
    self.mask_prob = 0.15  # percent of input tokens to mask out / replace

    # optimization
    self.learning_rate = 5e-4
    self.lr_decay_power = 1.0  # linear weight decay by default
    self.weight_decay_rate = 0.01
    self.num_warmup_steps = 10000

    # training settings
    self.iterations_per_loop = 200
    self.save_checkpoints_steps = 100000
    self.num_train_steps = 1000000
    self.num_eval_steps = 100
    self.keep_checkpoint_max = 0

    # model settings
    self.model_size = "base"  # one of "small", "medium-smal", or "base"
    # override the default transformer hparams for the provided model size; see
    # modeling.BertConfig for the possible hparams and util.training_utils for
    # the defaults
    self.model_hparam_overrides = (
        kwargs["model_hparam_overrides"]
        if "model_hparam_overrides" in kwargs else {})
    self.embedding_size = None  # bert hidden size by default
    self.vocab_size = 32000  # number of tokens in the vocabulary
    self.do_lower_case = False  # lowercase the input?
    
    # ConvBERT additional config
    self.conv_kernel_size=9
    self.linear_groups=2
    self.head_ratio=2
    self.conv_type="sdconv"
    # generator settings
    self.uniform_generator = False  # generator is uniform at random
    self.untied_generator_embeddings = False  # tie generator/discriminator
                                              # token embeddings?
    self.untied_generator = True  # tie all generator/discriminator weights?
    self.generator_layers = 1.0  # frac of discriminator layers for generator
    self.generator_hidden_size = 0.25  # frac of discrim hidden size for gen
    self.disallow_correct = False  # force the generator to sample incorrect
                                   # tokens (so 15% of tokens are always
                                   # fake)
    self.temperature = 1.0  # temperature for sampling from generator

    # batch sizes
    self.max_seq_length = 512
    self.train_batch_size = 128
    self.eval_batch_size = 128

    # TPU settings
    self.use_tpu = True
    self.tpu_job_name = None
    self.num_tpu_cores = 32
    self.tpu_name = "convbert"  # cloud TPU to use for training
    self.tpu_zone = None  # GCE zone where the Cloud TPU is located in
    self.gcp_project = None  # project name for the Cloud TPU-enabled project

    # default locations of data files
    self.pretrain_tfrecords = os.path.join(
        data_dir, "output-512/pretrain_data.tfrecord*")
    self.vocab_file = os.path.join(data_dir, "vocab.txt")
    self.model_dir = os.path.join(data_dir, "models", model_name)
    results_dir = os.path.join(self.model_dir, "results")
    self.results_txt = os.path.join(results_dir, "unsup_results.txt")
    self.results_pkl = os.path.join(results_dir, "unsup_results.pkl")

    # update defaults with passed-in hyperparameters
    self.update(kwargs)

    self.max_predictions_per_seq = int((self.mask_prob + 0.005) *
                                       self.max_seq_length)

    # debug-mode settings
    if self.debug:
      self.train_batch_size = 8
      self.num_train_steps = 20
      self.eval_batch_size = 4
      self.iterations_per_loop = 1
      self.num_eval_steps = 2

    # defaults for different-sized model
    if self.model_size in ["medium-small"]:
      self.embedding_size = 128
      self.conv_kernel_size=9
      self.linear_groups=2
      self.head_ratio=2
    elif self.model_size in ["small"]:
      self.embedding_size = 128
      self.conv_kernel_size=9
      self.linear_groups=1
      self.head_ratio=2
      self.learning_rate = 3e-4
    elif self.model_size in ["base"]:
      self.generator_hidden_size = 1/3
      self.learning_rate = 2e-4
      self.train_batch_size = 256
      self.eval_batch_size = 256
      self.conv_kernel_size=9
      self.linear_groups=1
      self.head_ratio=2

    # passed-in-arguments override (for example) debug-mode defaults
    self.update(kwargs)

  def update(self, kwargs):
    for k, v in kwargs.items():
      if k not in self.__dict__:
        raise ValueError("Unknown hparam " + k)
      self.__dict__[k] = v

