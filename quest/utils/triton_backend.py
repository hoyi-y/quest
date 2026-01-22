import importlib.util
from pathlib import Path
from typing import Tuple

import torch

_TRITON_MODULE = None


def _load_triton_module():
    global _TRITON_MODULE
    if _TRITON_MODULE is not None:
        return _TRITON_MODULE

    triton_path = Path(__file__).resolve().parents[4] / "my_triton" / "quest_qwen3" / "triton_kernels.py"
    spec = importlib.util.spec_from_file_location("quest_triton_kernels", triton_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Failed to load Triton kernels from {triton_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    _TRITON_MODULE = module
    return module


def _materialize_kv(iController, layer_idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
    kv_data = iController.kv_cache.buf_layer(layer_idx)
    indices = iController.kv_cache.indicies
    if len(indices) == 0:
        raise ValueError("KV cache is empty.")

    page_indices = torch.tensor(indices, device=kv_data.device, dtype=torch.long)
    pages = kv_data.index_select(0, page_indices)
    k = pages[:, 0].reshape(-1, iController.num_heads, iController.head_dim)
    v = pages[:, 1].reshape(-1, iController.num_heads, iController.head_dim)

    kv_len = iController.kv_cache.seqlen
    return k[:kv_len], v[:kv_len]


def triton_prefill_forward(q: torch.Tensor, iController, layer_idx: int) -> torch.Tensor:
    kernels = _load_triton_module()
    k, v = _materialize_kv(iController, layer_idx)
    return kernels.triton_attention(q, k, v)


def triton_decode_estimate(q: torch.Tensor, iController, layer_idx: int) -> torch.Tensor:
    kernels = _load_triton_module()
    k, _ = _materialize_kv(iController, layer_idx)
    return kernels.triton_page_estimate(q, k, iController.page_size)


def triton_decode_topk(estimated_attn_score: torch.Tensor, iController):
    page_budget = iController.inference_page_budget - 1
    if page_budget <= 0:
        return

    values, indices = torch.topk(estimated_attn_score, k=page_budget, dim=-1)
    iController.topk_dout_buffer[: values.size(0), : values.size(1)] = values
    iController.topk_dindices_buffer[: indices.size(0), : indices.size(1)] = indices.to(
        dtype=iController.topk_dindices_buffer.dtype
    )


def triton_decode_sparse_attn(
    q: torch.Tensor,
    iController,
    layer_idx: int,
    topk_indices: torch.Tensor,
) -> torch.Tensor:
    kernels = _load_triton_module()
    k, v = _materialize_kv(iController, layer_idx)

    if topk_indices is None or topk_indices is iController.kv_indices_without_last:
        return kernels.triton_attention(q, k, v)

    return kernels.triton_sparse_attention(q, k, v, topk_indices, iController.page_size)
