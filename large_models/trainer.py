# coding=utf-8
# Copyright 2020-present the HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
The Trainer class, to easily train a 🤗 Transformers from scratch or finetune it on a new task.
"""

import contextlib
import copy
import functools
import glob
import importlib.metadata
import inspect
import json
import math
import os
import random
import re
import shutil
import sys
import tempfile
import time
import warnings
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Type, Union


# Integrations must be imported before ML frameworks:
# isort: off
from transformers.integrations import (
    get_reporting_integration_callbacks,
    hp_params,
)

# isort: on

import huggingface_hub.utils as hf_hub_utils
import numpy as np
import torch
import torch.distributed as dist
from huggingface_hub import ModelCard, create_repo, upload_folder
from packaging import version
from torch import nn
from torch.utils.data import DataLoader, Dataset, IterableDataset, RandomSampler, SequentialSampler
from torch.optim import SGD

from transformers import __version__
from transformers import Trainer
from transformers.configuration_utils import PretrainedConfig
from transformers.data.data_collator import DataCollator, DataCollatorWithPadding, default_data_collator
from transformers.debug_utils import DebugOption, DebugUnderflowOverflow
from transformers.feature_extraction_sequence_utils import SequenceFeatureExtractor
from transformers.feature_extraction_utils import FeatureExtractionMixin
from transformers.hyperparameter_search import ALL_HYPERPARAMETER_SEARCH_BACKENDS, default_hp_search_backend
from transformers.image_processing_utils import BaseImageProcessor
from transformers.integrations.deepspeed import deepspeed_init, deepspeed_load_checkpoint, is_deepspeed_available
from transformers.integrations.tpu import tpu_spmd_dataloader
from transformers.modelcard import TrainingSummary
from transformers.modeling_utils import PreTrainedModel, load_sharded_checkpoint, unwrap_model
from transformers.models.auto.modeling_auto import (
    MODEL_FOR_CAUSAL_LM_MAPPING_NAMES,
    MODEL_MAPPING_NAMES,
)
from transformers.optimization import get_scheduler, Adafactor, AdamW
from transformers.processing_utils import ProcessorMixin
from transformers.pytorch_utils import (
    ALL_LAYERNORM_LAYERS,
    is_torch_greater_or_equal_than_1_13,
    is_torch_greater_or_equal_than_2_3,
)
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from transformers.trainer_callback import (
    CallbackHandler,
    DefaultFlowCallback,
    ExportableState,
    PrinterCallback,
    ProgressCallback,
    TrainerCallback,
    TrainerControl,
    TrainerState,
)
from transformers.trainer_pt_utils import (
    DistributedTensorGatherer,
    EvalLoopContainer,
    IterableDatasetShard,
    LabelSmoother,
    LayerWiseDummyOptimizer,
    LengthGroupedSampler,
    SequentialDistributedSampler,
    distributed_broadcast_scalars,
    distributed_concat,
    find_batch_size,
    get_model_param_count,
    get_module_class_from_name,
    get_parameter_names,
    nested_concat,
    nested_detach,
    nested_numpify,
    nested_xla_mesh_reduce,
    reissue_pt_warnings,
    remove_dummy_checkpoint,
)
from transformers.trainer_utils import (
    PREFIX_CHECKPOINT_DIR,
    BestRun,
    EvalLoopOutput,
    EvalPrediction,
    HPSearchBackend,
    HubStrategy,
    PredictionOutput,
    RemoveColumnsCollator,
    SaveStrategy,
    TrainerMemoryTracker,
    TrainOutput,
    check_target_module_exists,
    default_compute_objective,
    denumpify_detensorize,
    enable_full_determinism,
    find_executable_batch_size,
    get_last_checkpoint,
    has_length,
    neftune_post_forward_hook,
    number_of_arguments,
    seed_worker,
    set_seed,
    speed_metrics,
)
from transformers.training_args import OptimizerNames, ParallelMode, TrainingArguments
from transformers.utils import (
    ADAPTER_CONFIG_NAME,
    ADAPTER_SAFE_WEIGHTS_NAME,
    ADAPTER_WEIGHTS_NAME,
    CONFIG_NAME,
    SAFE_WEIGHTS_INDEX_NAME,
    SAFE_WEIGHTS_NAME,
    WEIGHTS_INDEX_NAME,
    WEIGHTS_NAME,
    XLA_FSDPV2_MIN_VERSION,
    PushInProgress,
    PushToHubMixin,
    can_return_loss,
    find_labels,
    is_accelerate_available,
    is_apex_available,
    is_bitsandbytes_available,
    is_datasets_available,
    is_galore_torch_available,
    is_grokadamw_available,
    is_in_notebook,
    is_ipex_available,
    is_liger_kernel_available,
    is_lomo_available,
    is_peft_available,
    is_safetensors_available,
    is_sagemaker_dp_enabled,
    is_sagemaker_mp_enabled,
    is_schedulefree_available,
    is_torch_compile_available,
    is_torch_mlu_available,
    is_torch_mps_available,
    is_torch_musa_available,
    is_torch_neuroncore_available,
    is_torch_npu_available,
    is_torch_xla_available,
    is_torch_xpu_available,
    is_torchao_available,
    logging,
    strtobool,
)
from transformers.utils.deprecation import deprecate_kwarg
from transformers.utils.quantization_config import QuantizationMethod


DEFAULT_CALLBACKS = [DefaultFlowCallback]
DEFAULT_PROGRESS_CALLBACK = ProgressCallback

if is_in_notebook():
    from .utils.notebook import NotebookProgressCallback

    DEFAULT_PROGRESS_CALLBACK = NotebookProgressCallback

if is_apex_available():
    from apex import amp

if is_datasets_available():
    import datasets

if is_torch_xla_available():
    import torch_xla.core.xla_model as xm
    import torch_xla.debug.metrics as met
    from torch_xla import __version__ as XLA_VERSION

    IS_XLA_FSDPV2_POST_2_2 = version.parse(XLA_VERSION) >= version.parse(XLA_FSDPV2_MIN_VERSION)
    if IS_XLA_FSDPV2_POST_2_2:
        import torch_xla.distributed.spmd as xs
        import torch_xla.runtime as xr
else:
    IS_XLA_FSDPV2_POST_2_2 = False


if is_sagemaker_mp_enabled():
    import smdistributed.modelparallel.torch as smp
    from smdistributed.modelparallel import __version__ as SMP_VERSION

    IS_SAGEMAKER_MP_POST_1_10 = version.parse(SMP_VERSION) >= version.parse("1.10")

    from .trainer_pt_utils import smp_forward_backward, smp_forward_only, smp_gather, smp_nested_concat
else:
    IS_SAGEMAKER_MP_POST_1_10 = False


if is_safetensors_available():
    import safetensors.torch

if is_peft_available():
    from peft import PeftModel


if is_accelerate_available():
    from accelerate import Accelerator, skip_first_batches
    from accelerate import __version__ as accelerate_version
    from accelerate.state import AcceleratorState
    from accelerate.utils import (
        DistributedDataParallelKwargs,
        DistributedType,
        load_fsdp_model,
        load_fsdp_optimizer,
        save_fsdp_model,
        save_fsdp_optimizer,
    )

    DATA_SAMPLERS = [RandomSampler]
    if version.parse(accelerate_version) > version.parse("0.23.0"):
        from accelerate.data_loader import SeedableRandomSampler

        DATA_SAMPLERS += [SeedableRandomSampler]

    if is_deepspeed_available():
        from accelerate.utils import DeepSpeedSchedulerWrapper

if is_accelerate_available("0.28.0"):
    from accelerate.utils import DataLoaderConfiguration


def _is_peft_model(model):
    if is_peft_available():
        classes_to_check = (PeftModel,) if is_peft_available() else ()
        # Here we also check if the model is an instance of `PeftMixedModel` introduced in peft>=0.7.0: https://github.com/huggingface/transformers/pull/28321
        if version.parse(importlib.metadata.version("peft")) >= version.parse("0.7.0"):
            from peft import PeftMixedModel

            classes_to_check = (*classes_to_check, PeftMixedModel)
        return isinstance(model, classes_to_check)
    return False


def _get_fsdp_ckpt_kwargs():
    # TODO: @AjayP13, @younesbelkada replace this check with version check at the next `accelerate` release
    if is_accelerate_available() and "adapter_only" in list(inspect.signature(save_fsdp_model).parameters):
        return {"adapter_only": True}
    else:
        return {}


def safe_globals():
    # Starting from version 2.4 PyTorch introduces a check for the objects loaded
    # with torch.load(weights_only=True). Starting from 2.6 weights_only=True becomes
    # a default and requires allowlisting of objects being loaded.
    # See: https://github.com/pytorch/pytorch/pull/137602
    # See: https://pytorch.org/docs/stable/notes/serialization.html#torch.serialization.add_safe_globals
    # See: https://github.com/huggingface/accelerate/pull/3036
    if version.parse(torch.__version__).release < version.parse("2.6").release:
        return contextlib.nullcontext()

    np_core = np._core if version.parse(np.__version__) >= version.parse("2.0.0") else np.core
    allowlist = [np_core.multiarray._reconstruct, np.ndarray, np.dtype]
    # numpy >1.25 defines numpy.dtypes.UInt32DType, but below works for
    # all versions of numpy
    allowlist += [type(np.dtype(np.uint32))]

    return torch.serialization.safe_globals(allowlist)


if TYPE_CHECKING:
    import optuna

    if is_datasets_available():
        import datasets

logger = logging.get_logger(__name__)


# Name of the files used for checkpointing
TRAINING_ARGS_NAME = "training_args.bin"
TRAINER_STATE_NAME = "trainer_state.json"
OPTIMIZER_NAME = "optimizer.pt"
OPTIMIZER_NAME_BIN = "optimizer.bin"
SCHEDULER_NAME = "scheduler.pt"
SCALER_NAME = "scaler.pt"
FSDP_MODEL_NAME = "pytorch_model_fsdp"

# Name of the module used for random perturbation sanity checks
SANITY_CHECK = False
SANITY_CHECK_MODULE_NAME = "model.decoder.layers.11.self_attn.k_proj.weight"

# Parameters to log v_t
PARAMETER_MAP_ZO = [
    (0, "lm_head"),
    (4, "layer0.k_proj"),
    (6, "layer0.v_proj"),
    (8, "layer0.q_proj"),
    (10, "layer0.out_proj"),
    (14, "layer0.fc1"),
    (16, "layer0.fc2"),
    (196, "layer12.k_proj"),
    (198, "layer12.v_proj"),
    (200, "layer12.q_proj"),
    (202, "layer12.out_proj"),
    (206, "layer12.fc1"),
    (208, "layer12.fc2"),
    (372, "layer23.k_proj"),
    (374, "layer23.v_proj"),
    (376, "layer23.q_proj"),
    (378, "layer23.out_proj"),
    (382, "layer23.fc1"),
    (384, "layer23.fc2"),
]

PARAMETER_MAP_FO = [
    (0, "lm_head"),
    (2, "layer0.k_proj"),
    (3, "layer0.v_proj"),
    (4, "layer0.q_proj"),
    (5, "layer0.out_proj"),
    (6, "layer0.fc1"),
    (7, "layer0.fc2"),
    (74, "layer12.k_proj"),
    (75, "layer12.v_proj"),
    (76, "layer12.q_proj"),
    (77, "layer12.out_proj"),
    (78, "layer12.fc1"),
    (79, "layer12.fc2"),
    (140, "layer23.k_proj"),
    (141, "layer23.v_proj"),
    (142, "layer23.q_proj"),
    (143, "layer23.out_proj"),
    (144, "layer23.fc1"),
    (145, "layer23.fc2"),
]

import pickle, wandb

class OurTrainer(Trainer):

    # from transformers.trainer_pt_utils import _get_learning_rate, log_metrics, metrics_format, save_metrics, save_state
    def _inner_training_loop(
        self, batch_size=None, args=None, resume_from_checkpoint=None, trial=None, ignore_keys_for_eval=None
    ):
        self.accelerator.free_memory()
        self._train_batch_size = batch_size
        if self.args.auto_find_batch_size:
            if self.state.train_batch_size != self._train_batch_size:
                from accelerate.utils import release_memory

                (self.model_wrapped,) = release_memory(self.model_wrapped)
                self.model_wrapped = self.model

                # Check for DeepSpeed *after* the intial pass and modify the config
                if self.is_deepspeed_enabled:
                    # Temporarily unset `self.args.train_batch_size`
                    original_bs = self.args.per_device_train_batch_size
                    self.args.per_device_train_batch_size = self._train_batch_size // max(1, self.args.n_gpu)
                    self.propagate_args_to_deepspeed(True)
                    self.args.per_device_train_batch_size = original_bs
            self.state.train_batch_size = self._train_batch_size
        logger.debug(f"Currently training with a batch size of: {self._train_batch_size}")
        # Data loader and number of training steps
        train_dataloader = self.get_train_dataloader()
        if self.is_fsdp_xla_v2_enabled:
            train_dataloader = tpu_spmd_dataloader(train_dataloader)

        # Setting up training control variables:
        # number of training epochs: num_train_epochs
        # number of training steps per epoch: num_update_steps_per_epoch
        # total number of training steps to execute: max_steps
        total_train_batch_size = self._train_batch_size * args.gradient_accumulation_steps * args.world_size

        len_dataloader = None
        num_train_tokens = None
        if has_length(train_dataloader):
            len_dataloader = len(train_dataloader)
            num_update_steps_per_epoch = len_dataloader // args.gradient_accumulation_steps
            num_update_steps_per_epoch = max(num_update_steps_per_epoch, 1)
            num_examples = self.num_examples(train_dataloader)
            if args.max_steps > 0:
                max_steps = args.max_steps
                num_train_epochs = args.max_steps // num_update_steps_per_epoch + int(
                    args.max_steps % num_update_steps_per_epoch > 0
                )
                # May be slightly incorrect if the last batch in the training dataloader has a smaller size but it's
                # the best we can do.
                num_train_samples = args.max_steps * total_train_batch_size
                if args.include_tokens_per_second:
                    num_train_tokens = (
                        self.num_tokens(train_dataloader, args.max_steps) * args.gradient_accumulation_steps
                    )
            else:
                max_steps = math.ceil(args.num_train_epochs * num_update_steps_per_epoch)
                num_train_epochs = math.ceil(args.num_train_epochs)
                num_train_samples = self.num_examples(train_dataloader) * args.num_train_epochs
                if args.include_tokens_per_second:
                    num_train_tokens = self.num_tokens(train_dataloader) * args.num_train_epochs
        elif args.max_steps > 0:  # Rely on max_steps when dataloader does not have a working size
            max_steps = args.max_steps
            # Setting a very large number of epochs so we go as many times as necessary over the iterator.
            num_train_epochs = sys.maxsize
            num_update_steps_per_epoch = max_steps
            num_examples = total_train_batch_size * args.max_steps
            num_train_samples = args.max_steps * total_train_batch_size
            if args.include_tokens_per_second:
                num_train_tokens = self.num_tokens(train_dataloader, args.max_steps) * args.gradient_accumulation_steps
        else:
            raise ValueError(
                "args.max_steps must be set to a positive value if dataloader does not have a length, was"
                f" {args.max_steps}"
            )

        if DebugOption.UNDERFLOW_OVERFLOW in self.args.debug:
            if self.args.n_gpu > 1:
                # nn.DataParallel(model) replicates the model, creating new variables and module
                # references registered here no longer work on other gpus, breaking the module
                raise ValueError(
                    "Currently --debug underflow_overflow is not supported under DP. Please use DDP"
                    " (torchrun or torch.distributed.launch (deprecated))."
                )
            else:
                debug_overflow = DebugUnderflowOverflow(self.model)  # noqa

        delay_optimizer_creation = is_sagemaker_mp_enabled() or self.is_fsdp_xla_enabled

        # We need to reset the scheduler, as its parameters may be different on subsequent calls
        if self._created_lr_scheduler:
            self.lr_scheduler = None
            self._created_lr_scheduler = False

        if self.is_deepspeed_enabled:
            self.optimizer, self.lr_scheduler = deepspeed_init(self, num_training_steps=max_steps)

        if not delay_optimizer_creation:
            self.create_optimizer_and_scheduler(num_training_steps=max_steps)

        self.state = TrainerState(
            stateful_callbacks=[
                cb for cb in self.callback_handler.callbacks + [self.control] if isinstance(cb, ExportableState)
            ]
        )
        self.state.is_hyper_param_search = trial is not None
        self.state.train_batch_size = self._train_batch_size

        # Compute absolute values for logging, eval, and save if given as ratio
        if args.logging_steps is not None:
            if args.logging_steps < 1:
                self.state.logging_steps = math.ceil(max_steps * args.logging_steps)
            else:
                self.state.logging_steps = args.logging_steps
        if args.eval_steps is not None:
            if args.eval_steps < 1:
                self.state.eval_steps = math.ceil(max_steps * args.eval_steps)
            else:
                self.state.eval_steps = args.eval_steps
        if args.save_steps is not None:
            if args.save_steps < 1:
                self.state.save_steps = math.ceil(max_steps * args.save_steps)
            else:
                self.state.save_steps = args.save_steps

        # Activate gradient checkpointing if needed
        if args.gradient_checkpointing:
            self.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs=args.gradient_checkpointing_kwargs)

        model = self._wrap_model(self.model_wrapped)

        # as the model is wrapped, don't use `accelerator.prepare`
        # this is for unhandled cases such as
        # FSDP-XLA, SageMaker MP/DP, DataParallel, IPEX
        use_accelerator_prepare = True if model is self.model else False

        if use_accelerator_prepare and self.is_fsdp_enabled:
            # In case of auto_find_batch_size=True
            # Remove FSDP wrapping from sub-models.
            self.model = unwrap_model(self.model, recursive=True)
            # configure fsdp plugin for qlora if any
            self._fsdp_qlora_plugin_updates()

        if delay_optimizer_creation:
            if use_accelerator_prepare:
                self.model = self.accelerator.prepare(self.model)
            self.create_optimizer_and_scheduler(num_training_steps=max_steps)

        # prepare using `accelerator` prepare
        if use_accelerator_prepare:
            self.model.train()
            if hasattr(self.lr_scheduler, "step"):
                if self.use_apex:
                    model = self.accelerator.prepare(self.model)
                else:
                    model, self.optimizer = self.accelerator.prepare(self.model, self.optimizer)
            else:
                # to handle cases wherein we pass "DummyScheduler" such as when it is specified in DeepSpeed config.
                model, self.optimizer, self.lr_scheduler = self.accelerator.prepare(
                    self.model, self.optimizer, self.lr_scheduler
                )
        elif self.args.optim in [OptimizerNames.LOMO, OptimizerNames.ADALOMO]:
            # In this case we are in DDP + LOMO, which should be supported
            self.optimizer = self.accelerator.prepare(self.optimizer)

        if self.is_fsdp_enabled:
            self.model = self.model_wrapped = model

        # for the rest of this function `model` is the outside model, whether it was wrapped or not
        if model is not self.model:
            self.model_wrapped = model

        # backward compatibility
        if self.is_deepspeed_enabled:
            self.deepspeed = self.model_wrapped

        # ckpt loading
        if resume_from_checkpoint is not None:
            if self.is_deepspeed_enabled:
                deepspeed_load_checkpoint(
                    self.model_wrapped, resume_from_checkpoint, load_module_strict=not _is_peft_model(self.model)
                )
            elif is_sagemaker_mp_enabled() or self.is_fsdp_enabled:
                self._load_from_checkpoint(resume_from_checkpoint, self.model_wrapped)

        ################# ZO added #################
        if "Adam" in args.trainer:
            self.optimizer = AdamW(self.model.parameters(), lr=args.learning_rate, betas=(args.beta1, args.beta2), eps=args.adaptivity, weight_decay=args.weight_decay)
        elif "SGD" in args.trainer:
            self.optimizer = SGD(self.model.parameters(), lr=args.learning_rate, momentum=args.momentum, weight_decay=args.weight_decay)
        else:
            logger.info(f"args.trainer {args.trainer} is not a ZO trainer. Do not define separated optimizer.")
        ############################################

        # Check if saved optimizer or scheduler states exist
        self._load_optimizer_and_scheduler(resume_from_checkpoint)

        # important: at this point:
        # self.model         is the Transformers Model
        # self.model_wrapped is DDP(Transformers Model), Deepspeed(Transformers Model),
        # FSDP(Transformers Model), Dynamo Optimized Module(Transformers Model) etc.

        # Train!
        logger.info("***** Running training *****")
        logger.info(f"  Num examples = {num_examples:,}")
        logger.info(f"  Num Epochs = {num_train_epochs:,}")
        logger.info(f"  Instantaneous batch size per device = {self.args.per_device_train_batch_size:,}")
        if self.args.per_device_train_batch_size != self._train_batch_size:
            logger.info(f"  Training with DataParallel so batch size has been adjusted to: {self._train_batch_size:,}")
        logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_train_batch_size:,}")
        logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
        logger.info(f"  Total optimization steps = {max_steps:,}")

        self.total_trainable_parameters = get_model_param_count(model, trainable_only=True)
        logger.info(f"  Number of trainable parameters = {self.total_trainable_parameters:,}")

        self.state.epoch = 0
        start_time = time.time()
        epochs_trained = 0
        steps_trained_in_current_epoch = 0
        steps_trained_progress_bar = None

        # Check if continuing training from a checkpoint
        if resume_from_checkpoint is not None and os.path.isfile(
            os.path.join(resume_from_checkpoint, TRAINER_STATE_NAME)
        ):
            self.state = TrainerState.load_from_json(os.path.join(resume_from_checkpoint, TRAINER_STATE_NAME))
            self.compare_trainer_and_checkpoint_args(self.args, self.state)
            self._load_callback_state()
            epochs_trained = int(self.state.global_step // num_update_steps_per_epoch)
            if not args.ignore_data_skip:
                steps_trained_in_current_epoch = self.state.global_step % (num_update_steps_per_epoch)
                steps_trained_in_current_epoch *= args.gradient_accumulation_steps
            else:
                steps_trained_in_current_epoch = 0

            logger.info("  Continuing training from checkpoint, will skip to saved global_step")
            logger.info(f"  Continuing training from epoch {epochs_trained}")
            logger.info(f"  Continuing training from global step {self.state.global_step}")
            if not args.ignore_data_skip:
                logger.info(
                    f"  Will skip the first {epochs_trained} epochs then the first"
                    f" {steps_trained_in_current_epoch} batches in the first epoch."
                )

        # Update the references
        self.callback_handler.model = self.model
        self.callback_handler.optimizer = self.optimizer
        self.callback_handler.lr_scheduler = self.lr_scheduler
        self.callback_handler.train_dataloader = train_dataloader
        if self.hp_name is not None and self._trial is not None:
            # use self._trial because the SigOpt/Optuna hpo only call `_hp_search_setup(trial)` instead of passing trial
            # parameter to Train when using DDP.
            self.state.trial_name = self.hp_name(self._trial)
        if trial is not None:
            assignments = trial.assignments if self.hp_search_backend == HPSearchBackend.SIGOPT else trial
            self.state.trial_params = hp_params(assignments)
        else:
            self.state.trial_params = None
        # This should be the same if the state has been saved but in case the training arguments changed, it's safer
        # to set this after the load.
        self.state.max_steps = max_steps
        self.state.num_train_epochs = num_train_epochs
        self.state.is_local_process_zero = self.is_local_process_zero()
        self.state.is_world_process_zero = self.is_world_process_zero()

        # tr_loss is a tensor to avoid synchronization of TPUs through .item()
        tr_loss = torch.tensor(0.0).to(args.device)
        # _total_loss_scalar is updated everytime .item() has to be called on tr_loss and stores the sum of all losses
        self._total_loss_scalar = 0.0
        self._globalstep_last_logged = self.state.global_step
        model.zero_grad()
        grad_norm: Optional[float] = None
        self.control = self.callback_handler.on_train_begin(args, self.state, self.control)

        if args.eval_on_start:
            self._evaluate(trial, ignore_keys_for_eval, skip_scheduler=True)

        for epoch in range(epochs_trained, num_train_epochs):
            epoch_dataloader = train_dataloader
            if hasattr(epoch_dataloader, "set_epoch"):
                epoch_dataloader.set_epoch(epoch)

            # Reset the past mems state at the beginning of each epoch if necessary.
            if args.past_index >= 0:
                self._past = None

            steps_in_epoch = (
                len(epoch_dataloader)
                if len_dataloader is not None
                else args.max_steps * args.gradient_accumulation_steps
            )
            self.control = self.callback_handler.on_epoch_begin(args, self.state, self.control)

            if epoch == epochs_trained and resume_from_checkpoint is not None and steps_trained_in_current_epoch == 0:
                self._load_rng_state(resume_from_checkpoint)

            rng_to_sync = False
            steps_skipped = 0
            if steps_trained_in_current_epoch > 0:
                epoch_dataloader = skip_first_batches(epoch_dataloader, steps_trained_in_current_epoch)
                steps_skipped = steps_trained_in_current_epoch
                steps_trained_in_current_epoch = 0
                rng_to_sync = True

            step = -1
            epoch_iterator = iter(epoch_dataloader)
            # We chunkify the epoch iterator into gradient accumulation steps `n` batches
            remainder = num_examples % args.gradient_accumulation_steps
            if remainder == 0:
                remainder = args.gradient_accumulation_steps
            update_step = -1
            total_updates = steps_in_epoch // args.gradient_accumulation_steps + 1
            for _ in range(total_updates):
                update_step += 1
                num_batches = args.gradient_accumulation_steps if update_step != (total_updates - 1) else remainder
                batch_samples, num_items_in_batch = self.get_batch_samples(epoch_iterator, num_batches)
                for i, inputs in enumerate(batch_samples):
                    step += 1
                    do_sync_step = (step + 1) % args.gradient_accumulation_steps == 0 or (step + 1) == steps_in_epoch
                    # Since we perform prefetching, we need to manually set sync_gradients
                    if not do_sync_step:
                        self.accelerator.gradient_state._set_sync_gradients(False)
                    else:
                        self.accelerator.gradient_state._set_sync_gradients(True)

                    if self.args.include_num_input_tokens_seen:
                        main_input_name = getattr(self.model, "main_input_name", "input_ids")
                        if main_input_name not in inputs:
                            logger.warning(
                                "Tried to track the number of tokens seen, however the current model is "
                                "not configured properly to know what item is the input. To fix this, add "
                                "a `main_input_name` attribute to the model class you are using."
                            )
                        else:
                            input_tokens = inputs[main_input_name].numel()
                            input_tokens = torch.tensor(input_tokens, device=self.args.device, dtype=torch.int64)
                            self.state.num_input_tokens_seen += (
                                self.accelerator.gather(input_tokens).sum().cpu().item()
                            )
                    if rng_to_sync:
                        self._load_rng_state(resume_from_checkpoint)
                        rng_to_sync = False

                    # Skip past any already trained steps if resuming training
                    if steps_trained_in_current_epoch > 0:
                        steps_trained_in_current_epoch -= 1
                        if steps_trained_progress_bar is not None:
                            steps_trained_progress_bar.update(1)
                        if steps_trained_in_current_epoch == 0:
                            self._load_rng_state(resume_from_checkpoint)
                        continue
                    elif steps_trained_progress_bar is not None:
                        steps_trained_progress_bar.close()
                        steps_trained_progress_bar = None

                    if step % args.gradient_accumulation_steps == 0:
                        self.control = self.callback_handler.on_step_begin(args, self.state, self.control)

                    # We explicitly want to avoid relying on `accelerator.accumulate` for generation training
                    context = (
                        functools.partial(self.accelerator.no_sync, model=model)
                        if i != len(batch_samples) - 1
                        else contextlib.nullcontext
                    )
                    with context():
                        ################# ZO added #################
                        if "LOZO" in args.trainer or "MeZO" in args.trainer:
                            # zo training
                            tr_loss_step = self.zo_step(model, inputs, sanity_check=SANITY_CHECK, lowrank="LOZO" in args.trainer, kfac="KFAC" in args.trainer, num_sampling=args.num_sampling)
                        else:
                            # regular training
                            tr_loss_step = self.training_step(model, inputs, num_items_in_batch)
                        ############################################

                    if (
                        args.logging_nan_inf_filter
                        and not is_torch_xla_available()
                        and (torch.isnan(tr_loss_step) or torch.isinf(tr_loss_step))
                    ):
                        # if loss is nan or inf simply add the average of previous logged losses
                        tr_loss = tr_loss + tr_loss / (1 + self.state.global_step - self._globalstep_last_logged)
                    else:
                        if tr_loss.device != tr_loss_step.device:
                            raise ValueError(
                                f"Calculated loss must be on the original device: {tr_loss.device} but device in use is {tr_loss_step.device}"
                            )
                        tr_loss = tr_loss + tr_loss_step

                    self.current_flos += float(self.floating_point_ops(inputs))

                    if do_sync_step:
                        # Since we perform prefetching, we need to manually set sync_gradients to True
                        self.accelerator.gradient_state._set_sync_gradients(True)

                        if "KFAC" in args.trainer:
                            self.control = self.callback_handler.on_pre_optimizer_step(args, self.state, self.control)
                            grad_norm = self.kfac_lowrank_zo_update()
                            self.control = self.callback_handler.on_optimizer_step(args, self.state, self.control)

                        elif "LOZO" in args.trainer:
                            self.control = self.callback_handler.on_pre_optimizer_step(args, self.state, self.control)
                            grad_norm = self.lowrank_zo_update()
                            self.control = self.callback_handler.on_optimizer_step(args, self.state, self.control)

                        elif "MeZO" in args.trainer:
                            self.control = self.callback_handler.on_pre_optimizer_step(args, self.state, self.control)
                            grad_norm = self.zo_update(num_sampling=args.num_sampling, sanity_check=SANITY_CHECK)
                            self.control = self.callback_handler.on_optimizer_step(args, self.state, self.control)
                        
                        else:
                            # Gradient clipping
                            if args.max_grad_norm is not None and args.max_grad_norm > 0:
                                # deepspeed does its own clipping

                                if is_sagemaker_mp_enabled() and args.fp16:
                                    _grad_norm = self.optimizer.clip_master_grads(args.max_grad_norm)
                                elif self.use_apex:
                                    # Revert to normal clipping otherwise, handling Apex or full precision
                                    _grad_norm = nn.utils.clip_grad_norm_(
                                        amp.master_params(self.optimizer),
                                        args.max_grad_norm,
                                    )
                                else:
                                    _grad_norm = self.accelerator.clip_grad_norm_(
                                        model.parameters(),
                                        args.max_grad_norm,
                                    )

                                if (
                                    is_accelerate_available()
                                    and self.accelerator.distributed_type == DistributedType.DEEPSPEED
                                ):
                                    grad_norm = model.get_global_grad_norm()
                                    # In some cases the grad norm may not return a float
                                    if hasattr(grad_norm, "item"):
                                        grad_norm = grad_norm.item()
                                else:
                                    grad_norm = _grad_norm

                            self.control = self.callback_handler.on_pre_optimizer_step(args, self.state, self.control)

                            self.optimizer.step()
                            # import pdb;pdb.set_trace()

                            self.control = self.callback_handler.on_optimizer_step(args, self.state, self.control)

                            optimizer_was_run = not self.accelerator.optimizer_step_was_skipped
                            if optimizer_was_run:
                                # Delay optimizer scheduling until metrics are generated
                                if not isinstance(self.lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                                    self.lr_scheduler.step()

                        # import pdb; pdb.set_trace()
                        model.zero_grad()
                        self.state.global_step += 1
                        self.state.epoch = epoch + (step + 1 + steps_skipped) / steps_in_epoch
                        self.control = self.callback_handler.on_step_end(args, self.state, self.control)
                        self._maybe_log_save_evaluate(
                            tr_loss, grad_norm, model, trial, epoch, ignore_keys_for_eval, start_time
                        )

                        if self.args.v_t_logging_steps > 0 and self.state.global_step % self.args.v_t_logging_steps == 0:
                            parameter_map = PARAMETER_MAP_FO if self.args.trainer == "regular" else PARAMETER_MAP_ZO
                            for param_map in parameter_map:
                                # m_t = self.optimizer.state[list(self.optimizer.state.keys())[param_map[0]]]['exp_avg'].clone().detach().cpu()
                                v_t = self.optimizer.state[list(self.optimizer.state.keys())[param_map[0]]]['exp_avg_sq'].clone().detach().cpu()
                                mean_v_t, std_v_t = v_t.mean().item(), v_t.std().item()
                                wandb.log({f"mean(v_t)/{param_map[1]}": mean_v_t})
                                wandb.log({f"std(v_t)/{param_map[1]}": std_v_t})
                                wandb.log({f"CoV(v_t)/{param_map[1]}": std_v_t/mean_v_t})
                                
                                # try:
                                #     with open(os.path.join(self.args.output_dir, f"{param_map[1]}/v_t_steps_{self.state.global_step}.pkl"), mode="wb") as f:
                                #         pickle.dump(v_t, f)
                                # except FileNotFoundError:
                                #     os.makedirs(os.path.join(self.args.output_dir, f"{param_map[1]}"))
                                # finally:
                                #     with open(os.path.join(self.args.output_dir, f"{param_map[1]}/v_t_steps_{self.state.global_step}.pkl"), mode="wb") as f:
                                #         pickle.dump(v_t, f)

                    else:
                        self.control = self.callback_handler.on_substep_end(args, self.state, self.control)

                    # PyTorch/XLA relies on the data loader to insert the mark_step for
                    # each step. Since we are breaking the loop early, we need to manually
                    # insert the mark_step here.
                    if self.control.should_epoch_stop or self.control.should_training_stop:
                        if is_torch_xla_available():
                            xm.mark_step()
                        break
                # We also need to break out of the nested loop
                if self.control.should_epoch_stop or self.control.should_training_stop:
                    if is_torch_xla_available():
                        xm.mark_step()
                    break
            if step < 0:
                logger.warning(
                    "There seems not to be a single sample in your epoch_iterator, stopping training at step"
                    f" {self.state.global_step}! This is expected if you're using an IterableDataset and set"
                    f" num_steps ({max_steps}) higher than the number of available samples."
                )
                self.control.should_training_stop = True

            self.control = self.callback_handler.on_epoch_end(args, self.state, self.control)
            self._maybe_log_save_evaluate(tr_loss, grad_norm, model, trial, epoch, ignore_keys_for_eval, start_time)

            if DebugOption.TPU_METRICS_DEBUG in self.args.debug:
                if is_torch_xla_available():
                    # tpu-comment: Logging debug metrics for PyTorch/XLA (compile, execute times, ops, etc.)
                    xm.master_print(met.metrics_report())
                else:
                    logger.warning(
                        "You enabled PyTorch/XLA debug metrics but you don't have a TPU "
                        "configured. Check your training configuration if this is unexpected."
                    )
            if self.control.should_training_stop:
                break

        if args.past_index and hasattr(self, "_past"):
            # Clean the state at the end of training
            delattr(self, "_past")

        logger.info("\n\nTraining completed. Do not forget to share your model on huggingface.co/models =)\n\n")
        if args.load_best_model_at_end and self.state.best_model_checkpoint is not None:
            # Wait for everyone to get here so we are sure the model has been saved by process 0.
            if is_torch_xla_available():
                xm.rendezvous("load_best_model_at_end")
            elif args.parallel_mode == ParallelMode.DISTRIBUTED:
                dist.barrier()
            elif is_sagemaker_mp_enabled():
                smp.barrier()

            self._load_best_model()

        # add remaining tr_loss
        self._total_loss_scalar += tr_loss.item()
        effective_global_step = max(self.state.global_step, 0.001)  # Avoid ZeroDivisionError
        train_loss = self._total_loss_scalar / effective_global_step

        metrics = speed_metrics(
            "train",
            start_time,
            num_samples=num_train_samples,
            num_steps=self.state.max_steps,
            num_tokens=num_train_tokens,
        )
        self.store_flos()
        metrics["total_flos"] = self.state.total_flos
        metrics["train_loss"] = train_loss

        self.is_in_train = False

        self._memory_tracker.stop_and_update_metrics(metrics)

        self.log(metrics)

        run_dir = self._get_output_dir(trial)
        checkpoints_sorted = self._sorted_checkpoints(use_mtime=False, output_dir=run_dir)

        # Delete the last checkpoint when save_total_limit=1 if it's different from the best checkpoint and process allowed to save.
        if self.args.should_save and self.state.best_model_checkpoint is not None and self.args.save_total_limit == 1:
            for checkpoint in checkpoints_sorted:
                if not os.path.samefile(checkpoint, self.state.best_model_checkpoint):
                    logger.info(f"Deleting older checkpoint [{checkpoint}] due to args.save_total_limit")
                    shutil.rmtree(checkpoint, ignore_errors=True)

        self.control = self.callback_handler.on_train_end(args, self.state, self.control)

        # Wait for the checkpoint to be uploaded.
        self._finish_current_push()

        # After training we make sure to retrieve back the original forward pass method
        # for the embedding layer by removing the forward post hook.
        if self.neftune_noise_alpha is not None:
            self._deactivate_neftune(self.model)

        return TrainOutput(self.state.global_step, train_loss, metrics)


    # =========================================== LOZO Functions ==============================================================

    def zo_perturb_parameters(self, random_seed=None, scaling_factor=1, order=0, sampling_order=0, sanity_check=False):
        """
        Perturb the parameters with random vector z.
        Input: 
        - random_seed: random seed for MeZO in-place perturbation (if it's None, we will use self.zo_random_seed)
        - scaling_factor: theta = theta + scaling_factor * z * eps
        """

        # Set the random seed to ensure that we sample the same z for perturbation/update
        torch.manual_seed(random_seed if random_seed is not None else self.zo_random_seed)
        
        for name, param in self.named_parameters_to_optim:
            # z = torch.randn(param.data.size(), device=param.data.device, dtype=param.data.dtype)
            if self.args.std_scaling:
                std=1/np.sqrt(param.numel())
            else:
                std=1.0
            
            z = torch.normal(mean=0, std=std, size=param.data.size(), device=param.data.device, dtype=param.data.dtype)
            param.data = param.data + scaling_factor * z * self.args.zo_eps

            # Sanity Check
            if sanity_check and name == SANITY_CHECK_MODULE_NAME:
                self.sanity_check[sampling_order][order][name] = z.clone()
        
    def lowrank_zo_perturb_parameters(self, random_seed=None, scaling_factor=1, order=0, sampling_order=0, sanity_check=False):
        """
        Perturb the parameters with low-rank perturbation.
        """
        args = self.args
        step = self.step

        torch.manual_seed(random_seed if random_seed is not None else self.zo_random_seed)
        
        for name, param in self.named_parameters_to_optim:
            if param.data.ndim >= 2:
                if step % args.step_interval == 0:
                    v = torch.randn(param.data.size(1), args.rank_r, device=param.data.device, dtype=param.data.dtype)
                    self.v[name] = v
                else:
                    v = self.v[name]
                u = self.random_gaussian_matrix(m=param.data.size(0), n=args.rank_r, device=param.data.device, dtype=param.data.dtype)
                param.data = param.data + scaling_factor * u@v.t() * self.args.zo_eps

                # Sanity Check
                if sanity_check and name == SANITY_CHECK_MODULE_NAME:
                    self.sanity_check[sampling_order][order][name] = z.clone()

            else:
                # for bias
                z = torch.randn(param.data.size(), device=param.data.device, dtype=param.data.dtype)
                param.data = param.data + scaling_factor * z * self.args.zo_eps


    def kfac_lowrank_zo_perturb_parameters(self, random_seed=None, scaling_factor=1, order=0, sampling_order=0, sanity_check=False):
        """
        Perturb the parameters with Kronecker factored low-rank perturbation.
        """
        args = self.args
        step = self.step

        torch.manual_seed(random_seed if random_seed is not None else self.zo_random_seed)
        
        for name, param in self.named_parameters_to_optim:
            if param.data.ndim >= 2:
                if step % args.step_interval == 0:
                    u = torch.randn(param.data.size(0), args.rank_r, device=param.data.device, dtype=param.data.dtype)
                    v = torch.randn(param.data.size(1), args.rank_r, device=param.data.device, dtype=param.data.dtype)
                    u, v = u / torch.linalg.norm(u, dim=0), v / torch.linalg.norm(v, dim=0)
                    self.u[name] = u
                    self.v[name] = v

                else:
                    u = self.u[name]
                    v = self.v[name]
                
                z = torch.randn(param.data.size(), device=param.data.device, dtype=param.data.dtype)
                
                # Sanity Check
                if sanity_check and name == SANITY_CHECK_MODULE_NAME:
                    self.sanity_check[sampling_order][order][name] = z.clone()
            
                param.data = param.data + scaling_factor * ((u@(u.t()@z))@v)@v.t() * self.args.zo_eps

            else:
                # for bias
                z = torch.randn(param.data.size(), device=param.data.device, dtype=param.data.dtype)
                param.data = param.data + scaling_factor * z * self.args.zo_eps

    def zo_forward(self, model, inputs):
        """
        Get (no gradient) loss from the model. Dropout is turned off too.
        """
        model.eval()
        if self.args.non_diff:
            # Non-differentiable objective (may require autoregressive generation)
            return self.zo_forward_nondiff(model, inputs)

        with torch.inference_mode():
            inputs = self._prepare_inputs(inputs)
            with self.compute_loss_context_manager():
                loss = self.compute_loss(model, inputs)
            if self.args.n_gpu > 1:
                # Warning: this is copied from the original Huggingface Trainer. Untested.
                loss = loss.mean()  # mean() to average on multi-gpu parallel training
        return loss.detach()


    def zo_forward_nondiff(self, model, inputs):
        """
        Get (no gradient) non-diffiable loss from the model.
        """
        model.eval()
        assert self.args.task_name == "SQuAD", "Non differentiable objective only supports SQuAD for now."

        with torch.inference_mode():
            inputs = self._prepare_inputs(inputs)
            args = self.args
            outputs = self.model.generate(
                inputs["input_ids"], do_sample=args.sampling, temperature=args.temperature, 
                num_beams=args.num_beams, top_p=args.top_p, top_k=args.top_k, max_new_tokens=min(args.max_new_tokens, args.max_length - inputs["input_ids"].size(1)), 
                num_return_sequences=1, eos_token_id=[self.tokenizer.encode(args.eos_token, add_special_tokens=False)[-1], self.tokenizer.eos_token_id],
            )
            output_text = []
            for i in range(len(outputs)):
                output_text.append(self.tokenizer.decode(outputs[i][inputs["input_ids"].size(1):], skip_special_tokens=True).strip())
            f1s = [f1(output_text[i], inputs['gold'][i]) for i in range(len(output_text))]
        
        return -torch.tensor(np.mean(f1s), dtype=torch.float32)


    def zo_step(self, model, inputs, lowrank=False, kfac=False, sanity_check=False, num_sampling=1):
        """
        Estimate gradient by Lowrank-zo. Return the loss from f(theta + uv^t)
        """
        args = self.args
        if hasattr(self, 'step'):
            self.step += 1
        else:
            self.step = 0
            self.v = {}
            self.u = {}
            if sanity_check:
                self.sanity_check = [[{}, {}, {}] for _ in range(num_sampling)]

        loss = self.zo_forward(model, inputs)

        self.named_parameters_to_optim = []
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.named_parameters_to_optim.append((name, param))

        if kfac:
            perturb_parameters_func = self.kfac_lowrank_zo_perturb_parameters
        elif lowrank:
            perturb_parameters_func = self.lowrank_zo_perturb_parameters
        else:
            perturb_parameters_func = self.zo_perturb_parameters
        
        if num_sampling > 1:
            assert not (lowrank or kfac), "`num_sampling` is only supported for MeZO"
            assert self.args.gradient_accumulation_steps == 1
            
            self.zo_random_seeds = []
            self.projected_grads = []
            
            for i in range(num_sampling):
                # Sample the random seed for sampling 
                self.zo_random_seeds.append(np.random.randint(1000000000))

                # First function evaluation
                perturb_parameters_func(scaling_factor=1, sanity_check=sanity_check, order=0, sampling_order=i, random_seed=self.zo_random_seeds[-1])
                loss1 = self.zo_forward(model, inputs)

                # Second function evaluation
                perturb_parameters_func(scaling_factor=-2, sanity_check=sanity_check, order=1, sampling_order=i, random_seed=self.zo_random_seeds[-1])
                loss2 = self.zo_forward(model, inputs)

                self.projected_grads.append(((loss1 - loss2) / (2 * self.args.zo_eps)).item())

                # Reset model back to its parameters at start of step
                perturb_parameters_func(scaling_factor=1, sanity_check=sanity_check, order=2, sampling_order=i, random_seed=self.zo_random_seeds[-1])

        else:
            # Sample the random seed for sampling 
            self.zo_random_seed = np.random.randint(1000000000)

            # First function evaluation
            perturb_parameters_func(scaling_factor=1, sanity_check=sanity_check, order=0)
            loss1 = self.zo_forward(model, inputs)

            # Second function evaluation
            perturb_parameters_func(scaling_factor=-2, sanity_check=sanity_check, order=1)
            loss2 = self.zo_forward(model, inputs)

            self.projected_grad = ((loss1 - loss2) / (2 * self.args.zo_eps)).item()

            # No gradient accumulation support
            assert self.args.gradient_accumulation_steps == 1

            # Reset model back to its parameters at start of step
            perturb_parameters_func(scaling_factor=1, sanity_check=sanity_check, order=2)
        
        return loss

    def zo_update(self, sanity_check=False, num_sampling=1):
        """
        Update the parameters with the estimated gradients.
        """
        args = self.args
        # import pdb; pdb.set_trace()

        if num_sampling > 1:
            assert num_sampling == len(self.zo_random_seeds) and num_sampling == len(self.projected_grads)

            for i in range(num_sampling):
                torch.manual_seed(self.zo_random_seeds[i])

                for name, param in self.named_parameters_to_optim:
                    # z = torch.randn(param.data.size(), device=param.data.device, dtype=param.data.dtype)
                    if self.args.std_scaling:
                        std=1/np.sqrt(param.numel())
                    else:
                        std=1.0
                    
                    z = torch.normal(mean=0, std=std, size=param.data.size(), device=param.data.device, dtype=param.data.dtype)

                    # sanity check
                    if sanity_check and name == SANITY_CHECK_MODULE_NAME:
                        pert_pert1 = torch.allclose(z, self.sanity_check[i][0][name], atol=1e-5)
                        pert1_pert2 = torch.allclose(self.sanity_check[i][0][name], self.sanity_check[i][1][name], atol=1e-5)
                        pert2_pert3 = torch.allclose(self.sanity_check[i][1][name], self.sanity_check[i][2][name], atol=1e-5)
                        if not (pert_pert1 and pert1_pert2 and pert2_pert3):
                            import pdb; pdb.set_trace()

                    if i == 0:
                        if "bias" not in name and "layer_norm" not in name and "layernorm" not in name:
                            param.grad = (self.projected_grads[i] * z + args.weight_decay * param.data) / num_sampling
                        else:
                            param.grad = (self.projected_grads[i] * z) / num_sampling
                    else:
                        if "bias" not in name and "layer_norm" not in name and "layernorm" not in name:
                            param.grad += (self.projected_grads[i] * z + args.weight_decay * param.data) / num_sampling
                        else:
                            param.grad += (self.projected_grads[i] * z) / num_sampling
            
            self.optimizer.step()
            self.lr_scheduler.step()
            self.optimizer.zero_grad()

            return 0
        else:
            # Reset the random seed for sampling zs
            torch.manual_seed(self.zo_random_seed)

            grad_norm_list = []
            for name, param in self.named_parameters_to_optim:
                # Resample z
                z = torch.randn(param.data.size(), device=param.data.device, dtype=param.data.dtype)

                # sanity check
                if sanity_check and name == SANITY_CHECK_MODULE_NAME:
                    pert_pert1 = torch.allclose(z, self.sanity_check[0][0][name], atol=1e-5)
                    pert1_pert2 = torch.allclose(self.sanity_check[0][0][name], self.sanity_check[0][1][name], atol=1e-5)
                    pert2_pert3 = torch.allclose(self.sanity_check[0][1][name], self.sanity_check[0][2][name], atol=1e-5)
                    if not (pert_pert1 and pert1_pert2 and pert2_pert3):
                        import pdb; pdb.set_trace()

                if "bias" not in name and "layer_norm" not in name and "layernorm" not in name:
                    param.grad = self.projected_grad * z + args.weight_decay * param.data
                else:
                    param.grad = self.projected_grad * z

                # Gradient clipping
                if args.max_grad_norm is not None and args.max_grad_norm > 0:
                    parameter_ratio = np.sqrt(param.numel() / self.total_trainable_parameters)
                    grad_norm_list.append(self.accelerator.clip_grad_norm_(
                        param,
                        args.max_grad_norm * parameter_ratio,
                    ).item())
                    
                # import pdb;pdb.set_trace()
                # more mem-efficient:
                # run optimizer.step here to avoid caching all grad.
                self.optimizer.step()
                param.grad = None

            self.lr_scheduler.step()
            self.optimizer.zero_grad()
            
            return np.sqrt(sum(grad_norm ** 2 for grad_norm in grad_norm_list))

    def lowrank_zo_update(self, sanity_check=False):
        args = self.args
        step = self.step

        # Reset the random seed for sampling
        torch.manual_seed(self.zo_random_seed)     

        grad_norm_list = []
        for name, param in self.named_parameters_to_optim:
            if param.data.ndim >= 2:
                v = self.v[name]
                
                if step % args.step_interval == 0:
                    # Dummy sampling for the reproducibility of u
                    v_ = torch.randn(param.data.size(1), args.rank_r, device=param.data.device, dtype=param.data.dtype)
                    del v_
                
                u = self.random_gaussian_matrix(m=param.data.size(0), n=args.rank_r, device=param.data.device, dtype=param.data.dtype)
                
                # sanity check
                if sanity_check and name == SANITY_CHECK_MODULE_NAME:
                    pert_pert1 = torch.allclose(z, self.sanity_check[0][0][name], atol=1e-5)
                    pert1_pert2 = torch.allclose(self.sanity_check[0][0][name], self.sanity_check[0][1][name], atol=1e-5)
                    pert2_pert3 = torch.allclose(self.sanity_check[0][1][name], self.sanity_check[0][2][name], atol=1e-5)
                    if not (pert_pert1 and pert1_pert2 and pert2_pert3):
                        import pdb; pdb.set_trace()
                
                param.grad = self.projected_grad * u@v.t()
            else:
                # Resample z for bias
                z = torch.randn(param.data.size(), device=param.data.device, dtype=param.data.dtype)
                param.grad = self.projected_grad * z

            # Gradient clipping
            if args.max_grad_norm is not None and args.max_grad_norm > 0:
                parameter_ratio = np.sqrt(param.numel() / self.total_trainable_parameters)
                grad_norm_list.append(self.accelerator.clip_grad_norm_(
                    param,
                    args.max_grad_norm * parameter_ratio,
                ).item())
            
            # import pdb;pdb.set_trace()
            # more mem-efficient:
            # run optimizer.step here to avoid caching all grad.
            self.optimizer.step()
            param.grad = None # avoid further update.
        
        self.lr_scheduler.step()
        self.optimizer.zero_grad()

        return np.sqrt(sum(grad_norm ** 2 for grad_norm in grad_norm_list))

    def kfac_lowrank_zo_update(self, sanity_check=False):
        args = self.args
        step = self.step

        # Reset the random seed for sampling
        torch.manual_seed(self.zo_random_seed)     

        grad_norm_list = []
        for name, param in self.named_parameters_to_optim:
            if param.data.ndim >= 2:
                u = self.u[name]
                v = self.v[name]
                
                if step % args.step_interval == 0:
                    # Dummy sampling for the reproducibility of the z
                    u_ = torch.randn(param.data.size(0), args.rank_r, device=param.data.device, dtype=param.data.dtype)
                    v_ = torch.randn(param.data.size(1), args.rank_r, device=param.data.device, dtype=param.data.dtype)
                    del u_, v_

                z = torch.randn(param.data.size(), device=param.data.device, dtype=param.data.dtype)

                # sanity check
                if sanity_check and name == SANITY_CHECK_MODULE_NAME:
                    pert_pert1 = torch.allclose(z, self.sanity_check[0][0][name], atol=1e-5)
                    pert1_pert2 = torch.allclose(self.sanity_check[0][0][name], self.sanity_check[0][1][name], atol=1e-5)
                    pert2_pert3 = torch.allclose(self.sanity_check[0][1][name], self.sanity_check[0][2][name], atol=1e-5)
                    if not (pert_pert1 and pert1_pert2 and pert2_pert3):
                        import pdb; pdb.set_trace()
                
                param.grad = self.projected_grad * ((u @ (u.t() @ z)) @ v) @ v.t()
            else:
                # Resample z for bias
                z = torch.randn(param.data.size(), device=param.data.device, dtype=param.data.dtype)
                param.grad = self.projected_grad * z
            
            # Gradient clipping
            if args.max_grad_norm is not None and args.max_grad_norm > 0:
                parameter_ratio = np.sqrt(param.numel() / self.total_trainable_parameters)
                grad_norm_list.append(self.accelerator.clip_grad_norm_(
                    param,
                    args.max_grad_norm * parameter_ratio,
                ).item())

            # import pdb;pdb.set_trace()
            # more mem-efficient:
            # run optimizer.step here to avoid caching all grad.
            self.optimizer.step()
            param.grad = None # avoid further update.
        
        self.lr_scheduler.step()
        self.optimizer.zero_grad()

        return np.sqrt(sum(grad_norm ** 2 for grad_norm in grad_norm_list))

    def random_gaussian_matrix(self, m, n, device, dtype, random_seed=None):
        if random_seed is not None:
            torch.manual_seed(random_seed)

        random_matrix = torch.randn(m, n, device=device, dtype=dtype)
        return random_matrix

    ############## Misc overload functions ##############


    def _set_signature_columns_if_needed(self):
        """
        We overload this function for non-differentiable objective training to pass "gold" -- the gold text for the task
        """
        if self._signature_columns is None:
            # Inspect model forward signature to keep only the arguments it accepts.
            model_to_inspect = self.model
            if _is_peft_model(self.model):
                if hasattr(self.model, "get_base_model"):
                    model_to_inspect = self.model.get_base_model()
                else:
                    # PeftMixedModel do not provide a `get_base_model` method
                    model_to_inspect = self.model.base_model.model
            signature = inspect.signature(model_to_inspect.forward)
            self._signature_columns = list(signature.parameters.keys())
            # Labels may be named label or label_ids, the default data collator handles that.
            self._signature_columns += list(set(["label", "label_ids"] + self.label_names))
            self._signature_columns += ["gold"]