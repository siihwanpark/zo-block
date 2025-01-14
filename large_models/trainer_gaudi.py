# coding=utf-8
# Copyright 2022 The HuggingFace Team. All rights reserved.
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
The Gaudi Trainer class, to easily train a 🤗 Transformers from scratch or finetune it on a new task.
"""

import contextlib
import copy
import functools
import inspect
import json
import math
import os
import random
import shutil
import sys
import time
import warnings
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union

import huggingface_hub.utils as hf_hub_utils
import numpy as np
import torch
from accelerate import skip_first_batches
from accelerate.data_loader import SeedableRandomSampler
from accelerate.utils import (
    DistributedDataParallelKwargs,
    GradientAccumulationPlugin,
    load_fsdp_model,
    load_fsdp_optimizer,
    save_fsdp_model,
    save_fsdp_optimizer,
)
from huggingface_hub import upload_folder
from torch.utils.data import DataLoader, Dataset, IterableDataset, RandomSampler
from transformers import Trainer
from transformers.data.data_collator import DataCollator
from transformers.debug_utils import DebugOption, DebugUnderflowOverflow
from transformers.integrations import hp_params
from transformers.integrations.deepspeed import (
    deepspeed_load_checkpoint,
    is_deepspeed_available,
    is_deepspeed_zero3_enabled,
)
from transformers.modeling_utils import PreTrainedModel, load_sharded_checkpoint
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from transformers.trainer import _get_fsdp_ckpt_kwargs, _is_peft_model
from transformers.trainer_callback import ExportableState, TrainerCallback, TrainerState
from transformers.trainer_pt_utils import (
    DistributedTensorGatherer,
    EvalLoopContainer,
    IterableDatasetShard,
    LengthGroupedSampler,
    SequentialDistributedSampler,
    find_batch_size,
    get_model_param_count,
    nested_concat,
    nested_detach,
    reissue_pt_warnings,
    remove_dummy_checkpoint,
)
from transformers.trainer_utils import (
    PREFIX_CHECKPOINT_DIR,
    EvalLoopOutput,
    EvalPrediction,
    HPSearchBackend,
    HubStrategy,
    IntervalStrategy,
    PredictionOutput,
    TrainOutput,
    denumpify_detensorize,
    enable_full_determinism,
    find_executable_batch_size,
    get_last_checkpoint,
    has_length,
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
    PushInProgress,
    is_datasets_available,
    is_peft_available,
    is_safetensors_available,
)

from optimum.utils import logging

from optimum.habana.accelerate import GaudiAccelerator
from optimum.habana.accelerate.utils import FP8ContextWrapper, GaudiDistributedType
from optimum.habana.utils import (
    HabanaProfile,
    get_hpu_memory_stats,
    set_seed,
    speed_metrics,
    to_device_dtype,
)
from optimum.habana.transformers.gaudi_configuration import GAUDI_CONFIG_NAME, GaudiConfig
from optimum.habana.transformers.integrations.deepspeed import deepspeed_init
from optimum.habana.transformers.trainer_utils import convert_into_dtypes, get_dtype
from optimum.habana.transformers.training_args import GaudiTrainingArguments

if is_datasets_available():
    import datasets

if is_safetensors_available():
    import safetensors.torch

if is_peft_available():
    from peft import PeftModel
    from peft.utils import PeftType

if is_deepspeed_available():
    from accelerate.utils import DeepSpeedSchedulerWrapper

from accelerate.utils import DataLoaderConfiguration

### ZO added ###
from optimum.habana.transformers import GaudiTrainer
from torch.optim import AdamW, SGD

from pruning_utils import (
    fast_random_mask_like,
    estimate_pretrained_model_magnitude_pruning_threshold,
    compute_named_parameters_to_sparsity,
    estimate_pretrained_model_magnitude_pruning_layerwise_thresholds,
    get_random_mask,
    get_threshold_mask
)
from utils import encode_prompt, Prediction
from metrics import calculate_metric
from tqdm import tqdm
import re
import torch.nn.functional as F
from collections import defaultdict
################

def _get_input_update_settings(model, lazy_mode: Optional[bool] = None) -> Tuple[bool, Dict]:
    """
    Determines whether the input settings need to be updated.

    Currently (attn_softmax_bf16, use_flash_attention, flash_attention_recompute,
    flash_attention_causal_mask) are enabled only for llama, qwen2, starcoder2, gemma, baichuan
    and chatglm

    lazy_mode for llama, qwen2, starcoder2 and mistral

    Args:
        model: The model instance for which the input update settings are being evaluated
        lazy_mode[Optional[bool]]: Whether to use lazy mode for the model (defaults to `None`)

    Returns:
        Tuple[bool, Dict]: A flag indicating whether the input settings should be updated.
        A dictionary containing the specific input settings that need to be updated, if any
    """
    inputs_update: Dict = {}

    should_update_inputs = (getattr(model, "generation_config", None) is not None) and (
        model.config.model_type in ("llama", "qwen2", "starcoder2", "gemma", "baichuan", "chatglm")
    )
    if should_update_inputs:
        if model.generation_config.attn_softmax_bf16:
            inputs_update["attn_softmax_bf16"] = True
        if model.generation_config.use_flash_attention:
            inputs_update["use_flash_attention"] = True
        if model.generation_config.flash_attention_recompute:
            inputs_update["flash_attention_recompute"] = True
        if model.generation_config.flash_attention_causal_mask:
            inputs_update["flash_attention_causal_mask"] = True

    should_update_inputs = (
        (getattr(model, "generation_config", None) is not None)
        and (model.config.model_type in ("llama", "qwen2", "starcoder2", "mistral"))
        and (lazy_mode is not None)
    )
    if should_update_inputs:
        if _is_peft_model(model):
            forward_method = getattr(model.get_base_model(), "forward")
        else:
            forward_method = getattr(model, "forward")
        signature = inspect.signature(forward_method)
        if "lazy_mode" in signature.parameters:
            inputs_update["lazy_mode"] = lazy_mode

    should_update_inputs: bool = len(inputs_update) > 0

    return should_update_inputs, inputs_update

if TYPE_CHECKING:
    import optuna

DATA_SAMPLERS = [RandomSampler, SeedableRandomSampler]

logger = logging.get_logger(__name__)

# Name of the files used for checkpointing
TRAINING_ARGS_NAME = "training_args.bin"
TRAINER_STATE_NAME = "trainer_state.json"
OPTIMIZER_NAME = "optimizer.pt"
OPTIMIZER_NAME_BIN = "optimizer.bin"
SCHEDULER_NAME = "scheduler.pt"
SCALER_NAME = "scaler.pt"

SANITY_CHECK = False

class OurGaudiTrainer(GaudiTrainer):
    def _inner_training_loop(
        self,
        batch_size=None,
        args=None,
        resume_from_checkpoint=None,
        trial=None,
        ignore_keys_for_eval=None,
    ):
        self.accelerator.free_memory()
        self._train_batch_size = batch_size
        if self.args.auto_find_batch_size:
            if self.state.train_batch_size != self._train_batch_size:
                from accelerate.utils import release_memory

                (self.model_wrapped,) = release_memory(self.model_wrapped)
                self.model_wrapped = self.model

                # Check for DeepSpeed *after* the initial pass and modify the config
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

        # Setting up training control variables:
        # number of training epochs: num_train_epochs
        # number of training steps per epoch: num_update_steps_per_epoch
        # total number of training steps to execute: max_steps
        total_train_batch_size = self._train_batch_size * args.gradient_accumulation_steps * args.world_size
        if (
            self.accelerator.mpu.sequence_parallel_is_initialized()
            and self.accelerator.mpu.get_sequence_parallel_world_size() > 1
        ):
            total_train_batch_size = total_train_batch_size / self.accelerator.mpu.get_sequence_parallel_world_size()

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
            debug_overflow = DebugUnderflowOverflow(self.model)  # noqa

        delay_optimizer_creation = self.is_fsdp_enabled

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
            import transformers.modeling_utils

            if args.deepspeed:
                from deepspeed.runtime.activation_checkpointing.checkpointing import (
                    CheckpointFunction,
                    non_reentrant_checkpoint,
                )

                # HACK because outputs should always be tuples
                def hpu_deepspeed_checkpointing(function, *checkpoint_args, use_reentrant: Optional[bool] = None):
                    """DeepSpeed activation checkpointing."""
                    if use_reentrant is None:
                        use_reentrant = True
                    if use_reentrant:
                        all_outputs = []
                        CheckpointFunction.apply(function, all_outputs, *checkpoint_args)
                    else:
                        logger.info("DeepSpeed activation checkpointing=non_reentrant_checkpoint")
                        all_outputs = non_reentrant_checkpoint(function, *checkpoint_args)

                    # Always return a tuple
                    # When all_outputs contains only one element, DeepSpeed returns this element instead of a tuple
                    # which is not consistent with some models. See https://github.com/microsoft/DeepSpeed/issues/1057.
                    return tuple(all_outputs)

                torch.utils.checkpoint.checkpoint = hpu_deepspeed_checkpointing
                transformers.modeling_utils.checkpoint = hpu_deepspeed_checkpointing
            elif args.use_lazy_mode:
                from .gradient_checkpointing import checkpoint as lazy_mode_checkpointing

                torch.utils.checkpoint.checkpoint = lazy_mode_checkpointing
                transformers.modeling_utils.checkpoint = lazy_mode_checkpointing

            self.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs=args.gradient_checkpointing_kwargs)

            # Wrap `_gradient_checkpointing_func` in the model with `transformer_engine` `activation_checkpointing` context.
            if self.accelerator.state.is_fp8_enabled:
                FP8ContextWrapper.gradient_checkpointing_wrap(self.model)
        else:
            # Hack because `RegressionModel` in test_trainer.py doesn't have `gradient_checkpointing_disable`
            if hasattr(self.model, "gradient_checkpointing_disable"):
                self.model.gradient_checkpointing_disable()

        model = self._wrap_model(self.model_wrapped)

        # as the model is wrapped, don't use `accelerator.prepare`
        # this is for unhandled cases such as
        # FSDP-XLA, SageMaker MP/DP, DataParallel, IPEX
        use_accelerator_prepare = True if model is self.model else False

        if delay_optimizer_creation:
            if use_accelerator_prepare:
                self._fsdp_qlora_plugin_updates()
                self.model = self.accelerator.prepare(self.model)
            self.create_optimizer_and_scheduler(num_training_steps=max_steps)

        # prepare using `accelerator` prepare
        if use_accelerator_prepare:
            self.model.train()
            if hasattr(self.lr_scheduler, "step"):
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
            elif self.is_fsdp_enabled:
                self._load_from_checkpoint(resume_from_checkpoint, self.model_wrapped)

        ################# ZO added #################
        if "Adam" in args.trainer:
            self.optimizer = AdamW(model.parameters(), lr=args.learning_rate, betas=(args.beta1, args.beta2), eps=args.adaptivity, weight_decay=args.weight_decay)
        elif "SGD" in args.trainer:
            self.optimizer = SGD(model.parameters(), lr=args.learning_rate, momentum=args.momentum, weight_decay=args.weight_decay)
        else:
            raise ValueError(f"args.trainer {args.trainer} is not a ZO trainer. Do not define separated optimizer.")

        if args.bcd:
            self.init_block_coordinate_descent(model=model, base_optimizer=self.optimizer, block_ordering=args.bcd_ordering,include_embedding=args.include_embedding, include_lm_head=args.include_lm_head)
        ############################################

        # Check if saved optimizer or scheduler states exist
        self._load_optimizer_and_scheduler(resume_from_checkpoint)
        
        if self.gaudi_config.use_fused_clip_norm:
            try:
                from habana_frameworks.torch.hpex.normalization import FusedClipNorm
            except ImportError as error:
                error.msg = (
                    f"Could not import 'FusedClipNorm' from 'habana_frameworks.torch.hpex.normalization'. {error.msg}."
                )
                raise error
            self.FusedNorm = FusedClipNorm(
                model.parameters(),
                args.max_grad_norm,
            )

        # important: at this point:
        # self.model         is the Transformers Model
        # self.model_wrapped is DDP(Transformers Model), Deepspeed(Transformers Model), etc.
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
        logger.info(f"  Number of trainable parameters = {get_model_param_count(model, trainable_only=True):,}")

        self.state.epoch = 0
        start_time = time.time()
        start_time_after_warmup = None
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

        # In multi-worker training: broadcast model parameters from worker:0 to all the others.
        # This must be done manually unless DistributedDataParallel is used.
        if self.args.parallel_mode == ParallelMode.DISTRIBUTED and self.args.distribution_strategy == "fast_ddp":
            from ..distributed import all_reduce_gradients

            logger.debug(
                f"Broadcasting the model parameters to assure that each of {self.args.world_size} workers start the training from the same point."
            )
            for param in model.parameters():
                torch.distributed.broadcast(param.data, src=0)

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
        # _total_loss_scalar is updated every time .item() has to be called on tr_loss and stores the sum of all losses
        self._total_loss_scalar = 0.0
        self._globalstep_last_logged = self.state.global_step
        self._zero_model_grad(model)
        _grad_norm: Optional[float] = None
        _should_compute_grad_norm: bool = not self.accelerator.distributed_type == GaudiDistributedType.DEEPSPEED and (
            # Gradient clipping
            args.max_grad_norm is not None and args.max_grad_norm > 0
        )

        # attn_softmax_bf16 and use_flash_attention are enabled only for llama, qwen2, starcoder2, gemma and baichuan
        # lazy_mode for llama, qwen2, starcoder2 and mistral
        _should_update_inputs, _inputs_update = _get_input_update_settings(self.model, lazy_mode=args.use_lazy_mode)

        self.control = self.callback_handler.on_train_begin(args, self.state, self.control)

        ################## ZO added ###################
        if args.sparse_perturbation:
            # For Gaudi, we cannot use RNG since `torch.Generator()` does not support HPU devices (CPU is extremely slow)
            # Instead, we manually sample the sparse masks and hold them in the HPU memory
            self.gradient_sparsity = args.gradient_sparsity
            
            # Precompute layerwise thresholds corresponding to the given target sparsity
            if args.sparse_perturbation_type == "scale":
                self.named_parameters_to_threshold = estimate_pretrained_model_magnitude_pruning_layerwise_thresholds(model, self.gradient_sparsity)
        ###############################################

        if args.eval_on_start:
            self._evaluate(trial, ignore_keys_for_eval, skip_scheduler=True)

        if self.args.adjust_throughput:
            self.log_evaluate_save_time = 0
        else:
            self.log_evaluate_save_time = None

        hb_profiler = HabanaProfile(
            warmup=self.args.profiling_warmup_steps,
            active=self.args.profiling_steps,
            record_shapes=self.args.profiling_record_shapes,
            with_stack=self.args.profiling_with_stack,
        )
        hb_profiler.start()

        total_batched_samples = 0
        if _is_peft_model(self.model) and self.model.peft_type == PeftType.ADALORA:
            self.model.base_model.peft_config[self.model.trainable_adapter_name].total_step = max_steps
            if max_steps < self.model.base_model.peft_config[self.model.trainable_adapter_name].tfinal:
                self.model.base_model.peft_config[self.model.trainable_adapter_name].tfinal = 0
        for epoch in range(epochs_trained, num_train_epochs):
            epoch_iterator = train_dataloader
            if hasattr(epoch_iterator, "set_epoch"):
                epoch_iterator.set_epoch(epoch)

            # Reset the past mems state at the beginning of each epoch if necessary.
            if args.past_index >= 0:
                self._past = None

            steps_in_epoch = (
                len(epoch_iterator)
                if len_dataloader is not None
                else args.max_steps * args.gradient_accumulation_steps
            )
            self.control = self.callback_handler.on_epoch_begin(args, self.state, self.control)

            if epoch == epochs_trained and resume_from_checkpoint is not None and steps_trained_in_current_epoch == 0:
                self._load_rng_state(resume_from_checkpoint)

            rng_to_sync = False
            steps_skipped = 0
            if steps_trained_in_current_epoch > 0:
                epoch_iterator = skip_first_batches(epoch_iterator, steps_trained_in_current_epoch)
                steps_skipped = steps_trained_in_current_epoch
                steps_trained_in_current_epoch = 0
                rng_to_sync = True

            step = -1
            for step, inputs in enumerate(epoch_iterator):
                if (
                    args.throughput_warmup_steps > 0
                    and (args.throughput_warmup_steps * args.gradient_accumulation_steps)
                    == epoch * steps_in_epoch + step
                ):
                    start_time_after_warmup = time.time()

                total_batched_samples += 1

                if self.args.include_num_input_tokens_seen:
                    main_input_name = getattr(self.model, "main_input_name", "input_ids")
                    if main_input_name not in inputs:
                        logger.warning(
                            "Tried to track the number of tokens seen, however the current model is "
                            "not configured properly to know what item is the input. To fix this, add "
                            "a `main_input_name` attribute to the model class you are using."
                        )
                    else:
                        self.state.num_input_tokens_seen += (
                            torch.sum(
                                self.accelerator.gather(
                                    torch.tensor(
                                        inputs[main_input_name].numel(), device=self.args.device, dtype=torch.int64
                                    )
                                )
                            )
                            .cpu()
                            .item()
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

                ################# ZO added #################
                # update sparse mask
                if args.sparse_perturbation and args.sparse_perturbation_type == "random":
                    self.named_parameters_to_sparse_mask = get_random_mask(model, self.gradient_sparsity)

                # update active block for block coordinate descent
                if args.bcd and (self.state.global_step % args.bcd_interval == 0):
                    self.update_active_blocks(model, block_ordering=args.bcd_ordering)
                ############################################

                # attn_softmax_bf16 and use_flash_attention is enabled only for llama, qwen2, starcoder2, gemma, baichuan and chatglm
                # lazy_mode for llama, qwen2, starcoder2 and mistral
                if _should_update_inputs:
                    inputs.update(_inputs_update)

                # TODO: keep syncs for fast DDP?
                with self.accelerator.accumulate(model):
                    ################# ZO added #################
                    if "MeZO" in args.trainer:
                        if args.lozo_perturbation:
                            tr_loss_step = self.lowrank_zo_step(model, inputs, sanity_check=SANITY_CHECK)
                        else:
                            # zo training
                            tr_loss_step = self.zo_step(model, inputs, sanity_check=SANITY_CHECK)
                    else:
                        # regular training
                        tr_loss_step = self.training_step(model, inputs)
                    ############################################
                    
                is_last_step_and_steps_less_than_grad_acc = (
                    steps_in_epoch <= args.gradient_accumulation_steps and (step + 1) == steps_in_epoch
                )

                is_optimization_step = (
                    total_batched_samples % args.gradient_accumulation_steps == 0
                    or
                    # last step in epoch but step is always smaller than gradient_accumulation_steps
                    is_last_step_and_steps_less_than_grad_acc
                )

                if (
                    args.parallel_mode == ParallelMode.DISTRIBUTED
                    and args.distribution_strategy == "fast_ddp"
                    and is_optimization_step
                ):
                    all_reduce_gradients(
                        model, use_hpu_graphs=True
                    )  # use HPU graphs for gradient fusion regardless of args.use_hpu_graphs_for_training setting

                if args.logging_nan_inf_filter and (torch.isnan(tr_loss_step) or torch.isinf(tr_loss_step)):
                    # if loss is nan or inf simply add the average of previous logged losses
                    tr_loss += tr_loss / (1 + self.state.global_step - self._globalstep_last_logged)
                else:
                    if tr_loss.device != tr_loss_step.device:
                        raise ValueError(
                            f"Calculated loss must be on the original device: {tr_loss.device} but device in use is {tr_loss_step.device}"
                        )
                    tr_loss += tr_loss_step

                self.current_flos += float(self.floating_point_ops(inputs))
                if args.use_lazy_mode:
                    self.htcore.mark_step()

                if is_optimization_step:
                    # the `or` condition of `is_last_step_and_steps_less_than_grad_acc` is not covered
                    # in accelerate. So, explicitly enable sync gradients to True in that case.
                    if is_last_step_and_steps_less_than_grad_acc:
                        self.accelerator.gradient_state._set_sync_gradients(True)

                    if "MeZO" in args.trainer:
                        self.control = self.callback_handler.on_pre_optimizer_step(args, self.state, self.control)
                        if self.args.lozo_perturbation:
                            grad_norm = self.lowrank_zo_update(sanity_check=SANITY_CHECK)
                        else:
                            grad_norm = self.zo_update(sanity_check=SANITY_CHECK)
                        self.control = self.callback_handler.on_optimizer_step(args, self.state, self.control)
                    else:
                        # If the condition is true, we need to compute _grad_norm
                        if _should_compute_grad_norm:
                            # deepspeed does its own clipping
                            if self.gaudi_config.use_fused_clip_norm and args.use_habana:
                                # TODO: to merge self.accelerator.clip_grad_norm_ when HMP is removed
                                _grad_norm = self.FusedNorm.clip_norm(model.parameters())
                            else:
                                # Revert to normal clipping otherwise
                                _grad_norm = self.accelerator.clip_grad_norm_(
                                    model.parameters(),
                                    args.max_grad_norm,
                                )

                        self.control = self.callback_handler.on_pre_optimizer_step(args, self.state, self.control)

                        self.optimizer.step()

                        self.control = self.callback_handler.on_optimizer_step(args, self.state, self.control)

                        optimizer_was_run = not self.accelerator.optimizer_step_was_skipped
                        if optimizer_was_run:
                            # Delay optimizer scheduling until metrics are generated
                            if not isinstance(self.lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                                self.lr_scheduler.step()

                    self._zero_model_grad(model)
                    self.state.global_step += 1
                    self.state.epoch = epoch + (step + 1 + steps_skipped) / steps_in_epoch
                    if args.use_lazy_mode:
                        self.htcore.mark_step()
                    self.control = self.callback_handler.on_step_end(args, self.state, self.control)

                    self._maybe_log_save_evaluate(tr_loss, _grad_norm, model, trial, epoch, ignore_keys_for_eval)
                else:
                    self.control = self.callback_handler.on_substep_end(args, self.state, self.control)

                hb_profiler.step()
                if self.control.should_epoch_stop or self.control.should_training_stop:
                    break
            if step < 0:
                logger.warning(
                    "There seems not to be a single sample in your epoch_iterator, stopping training at step"
                    f" {self.state.global_step}! This is expected if you're using an IterableDataset and set"
                    f" num_steps ({max_steps}) higher than the number of available samples."
                )
                self.control.should_training_stop = True

            self.control = self.callback_handler.on_epoch_end(args, self.state, self.control)
            self._maybe_log_save_evaluate(tr_loss, _grad_norm, model, trial, epoch, ignore_keys_for_eval)

            if self.control.should_training_stop:
                break

        hb_profiler.stop()

        if args.past_index and hasattr(self, "_past"):
            # Clean the state at the end of training
            delattr(self, "_past")

        logger.info("\n\nTraining completed. Do not forget to share your model on huggingface.co/models =)\n\n")
        if args.load_best_model_at_end and self.state.best_model_checkpoint is not None:
            # Wait for everyone to get here so we are sure the model has been saved by process 0.
            if args.parallel_mode == ParallelMode.DISTRIBUTED:
                torch.distributed.barrier()

            self._load_best_model()

        # add remaining tr_loss
        self._total_loss_scalar += tr_loss.item()
        effective_global_step = max(self.state.global_step, 0.001)  # Avoid ZeroDivisionError
        train_loss = self._total_loss_scalar / effective_global_step

        # Warmup steps are removed from the calculation of speed metrics
        num_samples_for_speed_metrics = num_train_samples - args.throughput_warmup_steps * total_train_batch_size
        num_steps_for_speed_metrics = self.state.max_steps - args.throughput_warmup_steps
        metrics = speed_metrics(
            "train",
            start_time,
            num_samples=num_samples_for_speed_metrics,
            num_steps=num_steps_for_speed_metrics,
            num_tokens=num_train_tokens,
            start_time_after_warmup=start_time_after_warmup,
            log_evaluate_save_time=self.log_evaluate_save_time,
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

    # =========================================== ZO Functions ==============================================================
    def get_grad_sparsity_by_name(self, name):
        if self.gradient_sparsity is None:
            return None
        elif isinstance(self.gradient_sparsity, float):
            return self.gradient_sparsity
        elif isinstance(self.gradient_sparsity, dict):
            return self.gradient_sparsity[name]
    
    def zo_perturb_parameters(self, random_seed=None, scaling_factor=1, order=0, sampling_order=0, sanity_check=False):
        """
        Perturb the parameters with random vector z.
        Input: 
        - random_seed: random seed for MeZO in-place perturbation (if it's None, we will use self.zo_random_seed)
        - scaling_factor: theta = theta + scaling_factor * z * eps
        """
        args = self.args
        if args.sparse_perturbation and args.sparse_perturbation_type == "random":
            name_to_mask = self.named_parameters_to_sparse_mask
        
        # Set the random seed to ensure that we sample the same z for perturbation/update
        torch.manual_seed(random_seed if random_seed is not None else self.zo_random_seed)
        
        for name, param in self.named_parameters_to_optim:
            if not args.save_perturbations:
                z = torch.normal(mean=0, std=1.0, size=param.data.size(), device=param.data.device, dtype=param.data.dtype)
            else:
                z = self.z[name]
            
            if args.sparse_perturbation:
                if args.sparse_perturbation_type == "random":
                    z = name_to_mask[name] * z
                elif args.sparse_perturbation_type == "scale":
                    mask = get_threshold_mask(name, param.data, self.named_parameters_to_threshold[name]).to(param.device)
                    z = mask * z

            param.data = param.data + scaling_factor * z * self.args.zo_eps

            # Sanity Check
            if sanity_check and name == SANITY_CHECK_MODULE_NAME:
                self.sanity_check[sampling_order][order][name] = z.clone()

    def zo_forward(self, model, inputs):
        """
        Get (no gradient) loss from the model. Dropout is turned off too.
        """
        model.eval()
        if self.args.non_diff:
            # Non-differentiable objective (may require autoregressive generation)
            return self.zo_forward_nondiff(model, inputs)

        # with torch.inference_mode():
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


    def zo_step(self, model, inputs, sanity_check=False):
        """
        Estimate gradient by Lowrank-zo. Return the loss from f(theta + uv^t)
        """
        args = self.args
        if hasattr(self, 'step'):
            self.step += 1
        else:
            self.step = 0
            self.s_u = {}
            if sanity_check:
                self.sanity_check = [{}, {}, {}]
        
        if args.save_perturbations:
            self.z = {}

        self.named_parameters_to_optim = []
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.named_parameters_to_optim.append((name, param))
        
        # Sample the random seed for sampling 
        self.zo_random_seed = np.random.randint(1000000000)
        if args.save_perturbations:
            torch.manual_seed(self.zo_random_seed)
            for name, param in self.named_parameters_to_optim:
                self.z[name] = torch.normal(mean=0, std=1, size=param.size(), device=param.device, dtype=param.dtype)

        # First function evaluation
        self.zo_perturb_parameters(scaling_factor=1, sanity_check=sanity_check, order=0)
        loss1 = self.zo_forward(model, inputs)

        # Second function evaluation
        self.zo_perturb_parameters(scaling_factor=-2, sanity_check=sanity_check, order=1)
        loss2 = self.zo_forward(model, inputs)

        self.projected_grad = ((loss1 - loss2) / (2 * self.args.zo_eps)).item()

        # No gradient accumulation support
        assert args.gradient_accumulation_steps == 1

        # Reset model back to its parameters at start of step
        self.zo_perturb_parameters(scaling_factor=1, sanity_check=sanity_check, order=2)
        
        return loss1

    def zo_update(self, sanity_check=False):
        """
        Update the parameters with the estimated gradients.
        """
        args = self.args
        if args.sparse_perturbation and args.sparse_perturbation_type == "random":
            name_to_mask = self.named_parameters_to_sparse_mask

        # Reset the random seed for sampling zs
        torch.manual_seed(self.zo_random_seed)

        grad_norm_list = []
        for name, param in self.named_parameters_to_optim:
            if not args.save_perturbations:
                z = torch.normal(mean=0, std=1.0, size=param.data.size(), device=param.data.device, dtype=param.data.dtype)
            else:
                z = self.z[name]
            
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

            # sparse random perturbations
            if args.sparse_perturbation:
                if args.sparse_perturbation_type == "random":
                    param.grad = name_to_mask[name] * param.grad
                elif args.sparse_perturbation_type == "scale":
                    mask = get_threshold_mask(name, param.data, self.named_parameters_to_threshold[name]).to(param.device)
                    param.grad = mask * param.grad

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
    
    ############## Low-rank Random Perturbation Functions ##############
    def lowrank_zo_perturb_parameters(self, random_seed=None, scaling_factor=1, order=-1, sanity_check=False):
        args = self.args
        step = self.step

        torch.manual_seed(random_seed if random_seed is not None else self.zo_random_seed)

        for name, param in self.named_parameters_to_optim:
            if param.data.ndim >= 2:
                if step % args.lowrank_step_interval == 0:
                    v = torch.normal(mean=0, std=1, size=(param.data.size(1), args.rank_r), device=param.data.device, dtype=param.data.dtype)
                    self.v[name] = v
                else:
                    v = self.v[name]
                
                if not args.save_perturbations:
                    u = torch.normal(mean=0, std=1, size=(param.data.size(0), args.rank_r), device=param.data.device, dtype=param.data.dtype)               
                else:
                    u = self.u[name]
                param.data = param.data + scaling_factor * u@v.t() * args.zo_eps
            
            else:
                if not args.save_perturbations:
                    z = torch.normal(mean=0, std=1, size=param.data.size(), device=param.data.device, dtype=param.data.dtype)
                else:
                    z = self.u[name]
                param.data = param.data + scaling_factor * z * args.zo_eps

    def lowrank_zo_step(self, model, inputs, sanity_check=False):
        """
        Estimate gradient by Lowrank-zo. Return the loss from f(theta + uv^t)
        """
        args = self.args
        
        if hasattr(self, 'step'):
            self.step += 1
        else:
            self.step = 0
            self.v = {}
            if sanity_check:
                self.sanity_check = [{}, {}, {}]

        if args.save_perturbations:
            self.u = {}

        loss = self.zo_forward(model, inputs)

        self.named_parameters_to_optim = []
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.named_parameters_to_optim.append((name, param))
        
        # Sample the random seed for sampling 
        self.zo_random_seed = np.random.randint(1000000000)

        if args.save_perturbations:
            torch.manual_seed(self.zo_random_seed)
            for name, param in self.named_parameters_to_optim:
                self.u[name] = torch.normal(mean=0, std=1, size=(param.data.size(0), args.rank_r), device=param.data.device, dtype=param.data.dtype)               

        # First function evaluation
        self.lowrank_zo_perturb_parameters(scaling_factor=1, sanity_check=sanity_check, order=0)
        loss1 = self.zo_forward(model, inputs)

        # Second function evaluation
        self.lowrank_zo_perturb_parameters(scaling_factor=-2, sanity_check=sanity_check, order=1)
        loss2 = self.zo_forward(model, inputs)

        self.projected_grad = ((loss1 - loss2) / (2 * self.args.zo_eps)).item()

        # No gradient accumulation support
        assert args.gradient_accumulation_steps == 1

        # Reset model back to its parameters at start of step
        self.lowrank_zo_perturb_parameters(scaling_factor=1, sanity_check=sanity_check, order=2)
        
        return loss

    def lowrank_zo_update(self, sanity_check=False):
        """
        Update the parameters with the estimated gradients.
        """
        args = self.args
        step = self.step

        # Reset the random seed for sampling zs
        torch.manual_seed(self.zo_random_seed)

        grad_norm_list = []
        for name, param in self.named_parameters_to_optim:
            if param.data.ndim >= 2:
                if step % args.lowrank_step_interval == 0:
                    v = self.v[name]    
                    # # dummy sampling for the reproducibility of u
                    # v_ = torch.normal(mean=0, std=1, size=(param.data.size(1), args.rank_r), device=param.data.device, dtype=param.data.dtype)
                    # del v_
                else:
                    v = self.v[name]
                
                if not args.save_perturbations:
                    u = torch.normal(mean=0, std=1, size=(param.data.size(0), args.rank_r), device=param.data.device, dtype=param.data.dtype)
                else:
                    u = self.u[name]
                grad = self.projected_grad * u@v.t()

            else:
                if not args.save_perturbations:
                    z = torch.normal(mean=0, std=1, size=param.data.size(), device=param.data.device, dtype=param.data.dtype)
                else:
                    z = self.u[name]
                grad = self.projected_grad * z

            if "bias" not in name and "layer_norm" not in name and "layernorm" not in name:
                param.grad = grad + args.weight_decay * param.data
            else:
                param.grad = grad

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

    ############## Block Coordinate Descent Functions ##############
    def infer_param_groups(self, model, include_embedding=False, include_lm_head=False):
        """automatic inference of the parameter groups based on the parameter names.
        divide groups into:
            * embedding
            * transformer layers
            * lm_head and others

        Reference : https://github.com/Ledzy/BAdam/blob/12511504e53face3d2612f5bb4bac3a02afa817e/src/badam/block_optim.py#L143
        """

        block_prefix_list = []
        lm_head_and_other_params = []
        embed_pattern = r'.*embed[^.]*\.'
        layer_pattern = r'.*layers.[^.]*\.'

        for name, _ in model.named_parameters():
            if any(prefix[0] in name for prefix in block_prefix_list):
                continue
            
            if re.findall(layer_pattern, name):
                block_prefix_list.append(re.findall(layer_pattern, name))
            elif re.findall(embed_pattern, name) and include_embedding:
                block_prefix_list.append(re.findall(embed_pattern, name))
            else:
                lm_head_and_other_params.append(name)
        
        if include_lm_head:
            block_prefix_list.append(lm_head_and_other_params)

        return block_prefix_list
    
    def init_block_coordinate_descent(self, model, base_optimizer, block_ordering="random", active_modules=[], include_embedding=False, include_lm_head=False):
        
        assert base_optimizer is not None, "base_optimizer should be initialized before init_block_coordinate_descent."
        self.active_modules = active_modules
        self.block_prefix_list = self.infer_param_groups(model, include_embedding=include_embedding, include_lm_head=include_lm_head)
        self.block_num = len(self.block_prefix_list)

        if block_ordering == "random":
            self.block_order = torch.randperm(self.block_num).tolist()
        
        self.block_optimizer_defaults = base_optimizer.defaults

    def update_active_blocks(self, model, block_ordering="random"):
        """
        Update the active blocks for block coordinate descent and re-initialize the optimizer to flush the optimizer states.
        """
        assert hasattr(self, 'block_prefix_list') and hasattr(self, 'block_num'), "Block prefix list should be initialized properly."
        if block_ordering == "random":
            if len(self.block_order) == 0:
                self.block_order = torch.randperm(self.block_num).tolist()
                logger.info("Next block epoch's order has been updated")
            self.current_block_idx = self.block_order.pop()
        elif block_ordering == "ascending":
            if hasattr(self, 'current_block_idx'):
                self.current_block_idx = (self.current_block_idx + 1) % self.block_num
            else:
                self.current_block_idx = 0
        elif block_ordering == "descending":
            if hasattr(self, 'current_block_idx'):
                self.current_block_idx = (self.current_block_idx - 1) % self.block_num
            else:
                self.current_block_idx = self.block_num - 1
        else:
            raise ValueError(f"{block_ordering} is not a valid block ordering")

        trainable_param_groups = [
            {
                'params': [],
                'weight_decay': self.optimizer.param_groups[0]['weight_decay'],
                **self.block_optimizer_defaults
            },
            {
                'params': [],
                "weight_decay": 0.0,
                **self.block_optimizer_defaults
            }
        ]

        # Set param.requires_grad = False to every inactivated block
        self.active_param_prefixs = self.block_prefix_list[self.current_block_idx] + self.active_modules
        for name, param in model.named_parameters():
            if not any(p in name for p in self.active_param_prefixs):
                param.requires_grad_(False)
                param.grad = None
            else:
                param.requires_grad_(True)
                
                if "bias" not in name and "layer_norm" not in name and "layernorm" not in name:
                    trainable_param_groups[0]['params'].append(param)
                else:
                    trainable_param_groups[1]['params'].append(param)

        # remove the empty param groups
        trainable_param_groups[:] = [pg for pg in trainable_param_groups if len(pg["params"]) != 0]
        self.optimizer.param_groups = trainable_param_groups
        if self.args.state_flush:
            self.optimizer.state = defaultdict(dict) # flush the optimizer state
    
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

    def forward(self, input_ids, option_len=None, generation=False):
        """
        Given input_ids and the length of the option, return the log-likelihood of each token in the option.
        For generation tasks, return the generated text.
        This function is only for inference
        """
        input_ids = torch.tensor([input_ids]).to(self.model.device)

        if generation:
            args = self.args
            # Autoregressive generation
            outputs = self.model.generate(
                input_ids, do_sample=args.sampling, temperature=args.temperature, 
                num_beams=args.num_beams, top_p=args.top_p, top_k=args.top_k, max_new_tokens=min(args.max_new_tokens, args.max_length - input_ids.size(1)), 
                num_return_sequences=1, eos_token_id=[self.tokenizer.encode(args.eos_token, add_special_tokens=False)[-1], self.tokenizer.eos_token_id],
            )
            # For generation, directly return the text output
            output_text = self.tokenizer.decode(outputs[0][input_ids.size(1):], skip_special_tokens=True).strip()
            return output_text
        else:
            with torch.inference_mode():
                self.model.eval()
                logits = self.model(input_ids=input_ids).logits
            labels = input_ids[0, 1:]
            logits = logits[0, :-1] 
            log_probs = F.log_softmax(logits, dim=-1)

            selected_log_probs = log_probs[torch.arange(len(labels)).to(labels.device), labels]
            selected_log_probs = selected_log_probs.cpu().detach()
            # Only return the option (candidate) part
            return selected_log_probs[-option_len:]

    def one_step_pred(self, train_samples, eval_sample, verbose=False):
        """
        Return the prediction on the eval sample. In ICL, use train_samples as demonstrations
        """
        verbose = verbose or self.args.verbose
        if verbose:
            logger.info("========= Example =========")
            logger.info(f"Candidate: {eval_sample.candidates}")
            logger.info(f"Correct candidate: {eval_sample.correct_candidate}")

        # Encode (add prompt and tokenize) the sample; if multiple-choice/classification, encode all candidates (options)
        encoded_candidates, option_lens = encode_prompt(
            self.task, self.task.get_template(), train_samples, eval_sample, self.tokenizer, max_length=self.args.max_length, 
            generation=self.task.generation, max_new_tokens=self.args.max_new_tokens
        )

        # Calibration
        if self.args.sfc or self.args.icl_sfc:
            sfc_encoded_candidates, sfc_option_lens = encode_prompt(self.task, self.task.get_template(), 
                train_samples, eval_sample, self.tokenizer, max_length=self.args.max_length,
                sfc=self.args.sfc, icl_sfc=self.args.icl_sfc, generation=self.task.generation, 
                max_new_tokens=self.args.max_new_tokens
            )

        outputs = []
        if self.task.generation:
            # For generation tasks, return the autoregressively-generated text
            output_text = self.forward(encoded_candidates[0], generation=True)
            if verbose:
                logger.info("=== Prompt ===")
                logger.info(self.tokenizer.decode(encoded_candidates[0]))
                logger.info(f"Output: {output_text}") 
            return Prediction(correct_candidate=eval_sample.correct_candidate, predicted_candidate=output_text)
        else:
            # For classification/multiple-choice, calculate the probabilities of all candidates
            for candidate_id, encoded_candidate in enumerate(encoded_candidates):
                selected_log_probs = self.forward(encoded_candidate, option_len=option_lens[candidate_id])
                if verbose:
                    if candidate_id == 0:
                        logger.info("=== Candidate %d ===" % candidate_id)
                        logger.info(self.tokenizer.decode(encoded_candidate))
                    else:
                        logger.info("=== Candidate %d (without context)===" % candidate_id)
                        logger.info(self.tokenizer.decode(encoded_candidate).split(self.task.train_sep)[-1])
                    logger.info(f"Log probabilities of the option tokens: {selected_log_probs}")

                if self.args.sfc or self.args.icl_sfc:
                    sfc_selected_log_probs = self.forward(sfc_encoded_candidates[candidate_id], option_len=sfc_option_lens[candidate_id])
                    if verbose:
                        logger.info("=== Candidate %d (without context) SFC ===" % candidate_id)
                        logger.info(self.tokenizer.decode(sfc_encoded_candidates[candidate_id]).split(self.task.train_sep)[-1])
                        logger.info(f"Log probabilities of the option tokens: {sfc_selected_log_probs}")

                outputs.append({"log_probs": selected_log_probs, "sfc_log_probs": sfc_selected_log_probs if self.args.sfc or self.args.icl_sfc else None})

            if self.args.sfc or self.args.icl_sfc:
                # Calibrated probabilities (surface form competition; https://arxiv.org/pdf/2104.08315.pdf)
                # log p(candidate | input) = log p_lm(candidate | input) - log p_lm(candidate | sfc prompt)
                scores = [x['log_probs'].sum().item() - x['sfc_log_probs'].sum().item() for x in outputs]
            else:
                # (Default) length-normalized log probabilities
                # log p(candidate | input) = log p_lm(candidate | input) / |candidate #tokens|
                scores = [x['log_probs'].mean().item() for x in outputs]

            if verbose:
                logger.info(f"Prediction scores: {scores}")

            if isinstance(eval_sample.correct_candidate, list):
                # For some datasets there are multiple correct answers
                correct_candidate_id = [eval_sample.candidates.index(c) for c in eval_sample.correct_candidate]
            else:
                correct_candidate_id = eval_sample.candidates.index(eval_sample.correct_candidate)

            return Prediction(correct_candidate=correct_candidate_id, predicted_candidate=int(np.argmax(scores)))
    
    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix="eval"):
        super().evaluate(eval_dataset, ignore_keys, metric_key_prefix)

        metrics = {}
        # Prediction loop
        predictions = []
        for eval_id, eval_sample in enumerate(tqdm(self.eval_samples)):
            predictions.append(
                self.one_step_pred([], eval_sample, verbose=(eval_id < 3))
            )

        # Calculate metrics
        metric_name = getattr(self.task, "metric_name", "accuracy")
        metrics[f"eval_{metric_name}"]=calculate_metric(predictions, metric_name)

        # Prediction loop
        predictions = []
        for dev_id, dev_sample in enumerate(tqdm(self.dev_samples)):
            predictions.append(
                self.one_step_pred([], dev_sample, verbose=(dev_id < 3))
            )

        # Calculate metrics 
        metrics[f"dev_{metric_name}"]=calculate_metric(predictions, metric_name)
        logger.info(metrics)
        self.log(metrics)

        if hasattr(self, 'best_eval_metrics'):
            if self.best_eval_metrics[f"best_eval_{metric_name}"] < metrics[f"eval_{metric_name}"]:
                self.best_eval_metrics[f"best_eval_{metric_name}"] = metrics[f"eval_{metric_name}"]
            if self.best_eval_metrics[f"best_dev_{metric_name}"] < metrics[f"dev_{metric_name}"]:
                if self.args.early_stop:
                    self.patience = 0
                self.best_eval_metrics[f"best_dev_{metric_name}"] = metrics[f"dev_{metric_name}"]
            else:
                if self.args.early_stop:
                    self.patience += 1
                    if self.patience >= self.args.patience:
                        self.control.should_training_stop = True
        else:
            if self.args.early_stop:
                self.patience = 0
            self.best_eval_metrics = {f"best_eval_{metric_name}": metrics[f"eval_{metric_name}"], f"best_dev_{metric_name}": metrics[f"dev_{metric_name}"]}
        self.log(self.best_eval_metrics)
    

    ############## Misc overload functions ##############
    def save_model(self, output_dir: Optional[str] = None, _internal_call: bool = False):
        """
        We overload this function to fix an FSDP saving bug (before fix, it will likely cause OOM) 
        """

        if output_dir is None:
            output_dir = self.args.output_dir

        if is_torch_tpu_available():
            self._save_tpu(output_dir)
        elif is_sagemaker_mp_enabled():
            # Calling the state_dict needs to be done on the wrapped model and on all processes.
            os.makedirs(output_dir, exist_ok=True)
            state_dict = self.model_wrapped.state_dict()
            if self.args.should_save:
                self._save(output_dir, state_dict=state_dict)
            if IS_SAGEMAKER_MP_POST_1_10:
                # 'user_content.pt' indicates model state_dict saved with smp >= 1.10
                Path(os.path.join(output_dir, "user_content.pt")).touch()
        elif (
            ShardedDDPOption.ZERO_DP_2 in self.args.sharded_ddp
            or ShardedDDPOption.ZERO_DP_3 in self.args.sharded_ddp
            or self.fsdp is not None
        ):
            from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, StateDictType, FullStateDictConfig
            full_state_dict_config = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)

            # Fix the FSDP loading bug
            with FSDP.state_dict_type(self.model, StateDictType.FULL_STATE_DICT, full_state_dict_config):
                state_dict = self.model.state_dict()
            # state_dict = self.model.state_dict()

            if self.args.should_save:
                self._save(output_dir, state_dict=state_dict)
        elif self.deepspeed:
            # this takes care of everything as long as we aren't under zero3
            if self.args.should_save:
                self._save(output_dir)

            if is_deepspeed_zero3_enabled():
                # It's too complicated to try to override different places where the weights dump gets
                # saved, so since under zero3 the file is bogus, simply delete it. The user should
                # either user deepspeed checkpoint to resume or to recover full weights use
                # zero_to_fp32.py stored in the checkpoint.
                if self.args.should_save:
                    file = os.path.join(output_dir, WEIGHTS_NAME)
                    if os.path.isfile(file):
                        # logger.info(f"deepspeed zero3: removing {file}, see zero_to_fp32.py to recover weights")
                        os.remove(file)

                # now save the real model if stage3_gather_16bit_weights_on_model_save=True
                # if false it will not be saved.
                # This must be called on all ranks
                if not self.deepspeed.save_16bit_model(output_dir, WEIGHTS_NAME):
                    logger.warning(
                        "deepspeed.save_16bit_model didn't save the model, since"
                        " stage3_gather_16bit_weights_on_model_save=false. Saving the full checkpoint instead, use"
                        " zero_to_fp32.py to recover weights"
                    )
                    self.deepspeed.save_checkpoint(output_dir)

        elif self.args.should_save:
            self._save(output_dir)

        # Push to the Hub when `save_model` is called by the user.
        if self.args.push_to_hub and not _internal_call:
            self.push_to_hub(commit_message="Model save")
