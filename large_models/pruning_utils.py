# -*- coding: utf-8 -*-
# @Author: pingzhili
# @Time: 2024/1/18
# reference : https://github.com/ZO-Bench/ZO-LLM/blob/main/zo-bench/gradient_pruning/pruning_utils.py

from typing import Dict

import torch
from transformers import PreTrainedModel


@torch.no_grad()
def random_mask_like(tensor, nonzero_ratio, generator=None):
    """
    Return a random mask with the same shape as the input tensor, where the fraction of True is equal to the sparsity.

    Examples
    --------
    >>> random_mask_like(torch.randn(10, 10), 0.1).count_nonzero()
    tensor(10)
    """
    mask = torch.zeros_like(tensor)
    mask.view(-1)[torch.randperm(mask.numel(), generator=generator)[:int(nonzero_ratio * mask.numel())]] = 1
    return mask.bool()


@torch.no_grad()
def fast_random_mask_like(tensor, nonzero_ratio, generator=None):
    """
    A much faster version of random_zero_mask_like, but the sparsity is not guaranteed.

    Examples
    --------
    >>> fast_random_mask_like(torch.randn(10, 10), 0.1).count_nonzero() < 20
    tensor(True)
    """
    mask = torch.empty_like(tensor).normal_(generator=generator) < nonzero_ratio
    return mask.bool()


@torch.no_grad()
def estimate_pretrained_model_magnitude_pruning_threshold(
        model: PreTrainedModel,
        global_sparsity: float,
) -> float:
    """
    Compute the magnitude threshold for pruning based on the global sparsity requirement.
    """
    all_weights = []
    for param in model.parameters():
        all_weights.append(
            param.view(-1).abs().clone().detach().cpu()
        )
    all_weights = torch.cat(all_weights)
    # subsample 102400 elements to estimate the threshold
    sample_size = int(min(1e7, all_weights.numel()))
    print(f"[Sparse gradient] Subsampling {sample_size} elements to estimate the threshold.")
    sub_weights = all_weights[torch.randperm(all_weights.numel())[:sample_size]]
    return torch.quantile(sub_weights.float(), global_sparsity).item()


@torch.no_grad()
def compute_named_parameters_to_sparsity(
        model: PreTrainedModel,
        threshold: float,
) -> Dict[str, float]:
    """
    Compute the sparsity of each named parameter in the model.
    """
    named_parameters_to_sparsity = {}
    for name, param in model.named_parameters():
        named_parameters_to_sparsity[name] = param.abs().le(threshold).float().mean().item()
    return named_parameters_to_sparsity

@torch.no_grad()
def estimate_pretrained_model_magnitude_pruning_layerwise_thresholds(
    model: PreTrainedModel,
    sparsity: float,
) -> Dict[str, float]:
    named_parameters_to_threshold = {}
    for name, param in model.named_parameters():
        param_ = param.view(-1).abs().clone().detach().cpu()
        sample_size = int(min(1e6, param_.numel()))
        param_ = param_[:sample_size]
        threshold = torch.quantile(param_.abs().float(), 1-sparsity).item()
        named_parameters_to_threshold[name] = threshold
    
    return named_parameters_to_threshold

@torch.no_grad()
def get_threshold_mask(model, thresholds):
    named_parameters_to_sparse_mask = {}
    for name, param in model.named_parameters():
        if not name in thresholds.keys():
            continue
        
        mask = param.abs().le(thresholds[name]).bool()
        named_parameters_to_sparse_mask[name] = mask

    return named_parameters_to_sparse_mask

@torch.no_grad()
def get_random_mask(model, sparsity):
    named_parameters_to_sparse_mask = {}
    for name, param in model.named_parameters():
        mask = (torch.normal(mean=0, std=1, size=param.size(), device=param.device, dtype=param.dtype) > sparsity).bool()
        named_parameters_to_sparse_mask[name] = mask
    
    return named_parameters_to_sparse_mask


# Block structured sparsity
@torch.no_grad()
def structured_random_mask_like(tensor, name, n_heads, nonzero_ratio, generator=None):
    """
    Generate a random structured mask with the same shape as the input tensor.
    The fraction of `True` values is approximately equal to `nonzero_ratio`.

    Parameters
    ----------
    tensor : torch.Tensor
        Input tensor to determine the mask's shape.
    name : str
        Name of the layer, used to determine structured masking logic.
    n_heads : int
        Number of attention heads, used for head-wise masking.
    nonzero_ratio : float
        Fraction of `True` values in the mask.
    generator : torch.Generator, optional
        Random generator for reproducibility.

    Returns
    -------
    torch.Tensor
        Boolean tensor mask with the same shape as the input tensor.

    Examples
    --------
    >>> structured_random_mask_like(torch.randn(16, 16), "q_proj", 8, 0.25).float().mean()
    tensor(0.2500)
    """
    if tensor.dim() == 1:
        mask = torch.zeros(tensor.numel())
        mask.view(-1)[torch.randperm(tensor.numel(), generator=generator)[:int(nonzero_ratio * tensor.numel())]] = 1
        return mask.bool()
    
    num_rows, num_cols = tensor.size(0), tensor.size(1)
    mask = torch.zeros(num_rows, 1)
    if "q_proj" in name or "k_proj" in name:
        mask.view(n_heads, -1)[torch.randperm(n_heads, generator=generator)[:int(nonzero_ratio * n_heads)]] = 1
        mask = mask.repeat_interleave(num_rows // n_heads)
        import pdb;pdb.set_trace()
        
    else:
        mask.view(-1)[torch.randperm(num_rows, generator=generator)[:int(nonzero_ratio * num_rows)]] = 1
    mask = mask.expand(-1, num_cols)

    return mask.bool()


@torch.no_grad()
def fast_structured_random_mask_like(tensor, name, n_heads, nonzero_ratio, generator=None):
    """
    A faster version of structured_random_mask_like. Sparsity is approximated but not guaranteed.

    Examples
    --------
    >>> fast_structured_random_mask_like(torch.randn(10, 10), "q_proj", 8, 0.25).float().mean()
    tensor(0.2500)
    """

    if tensor.dim() == 1:
        mask = torch.empty_like(tensor).normal_(generator=generator) < nonzero_ratio
        return mask.bool()
        
    num_rows, num_cols = tensor.size(0), tensor.size(1)
    if "q_proj" in name or "k_proj" in name:
        # Head-wise partitioning
        mask = torch.empty(n_heads, device=tensor.device).normal_(generator=generator) < nonzero_ratio
        mask = mask.repeat_interleave(num_rows // n_heads).view(num_rows, 1)
    else:
        # General embedding or lm_head case
        mask = torch.empty(num_rows, 1, device=tensor.device).normal_(generator=generator) < nonzero_ratio
    mask = mask.expand(-1, num_cols)
    
    return mask.bool()


if __name__ == "__main__":
    gradient_sparsity = 0.8
