import math
import numpy as np
from typing import Optional, Tuple, Union

import torch
from torch import nn
import torch.utils.checkpoint
import torch.nn.functional as F
from torch.cuda.amp import autocast


import types

from transformers.models.llama.modeling_llama import (
    LlamaAttention,
    apply_rotary_pos_emb,
    repeat_kv,
)

from transformers.cache_utils import DynamicCache,Cache

from transformers.models.mistral.modeling_mistral import MistralAttention
# Qwen3 模型会通过 trust_remote_code=True 动态加载，无需显式 import
from transformers.models.qwen3.modeling_qwen3 import Qwen3Attention

def optimized_heavy_hitter_mask(attn_weights, token_budget, chunk_size):
    # attn_weights: [bsz, n_heads, 1, seq_len] (Decode 阶段 q_len=1)
    bsz, n_heads, q_len, seq_len = attn_weights.shape
    
    # 1. 计算需要的 padding (使用 F.pad 代替 cat，速度更快)
    remainder = seq_len % chunk_size
    if remainder != 0:
        pad_len = chunk_size - remainder
        # Pad with min value
        padded_weights = F.pad(attn_weights, (0, pad_len), value=torch.finfo(attn_weights.dtype).min)
    else:
        padded_weights = attn_weights
        
    num_chunks = padded_weights.shape[-1] // chunk_size

    # 2. Chunk Max Pooling
    # [bsz, n_heads, 1, num_chunks, chunk_size] -> Max -> [bsz, n_heads, 1, num_chunks]
    # 这里直接 view + max，显存开销极小
    chunk_max = padded_weights.view(bsz, n_heads, q_len, num_chunks, chunk_size).amax(dim=-1)
    
    # 3. 选出 Top-K 个 Chunk
    # 保证至少选 3 个 chunk，或者由 budget 决定
    k = min(max(3, token_budget // chunk_size), num_chunks)
    _, topk_chunk_idx = chunk_max.topk(k=k, dim=-1) # [bsz, n_heads, q_len, k]

    # 4. 还原为 Token 级别的 Mask
    # 利用广播将 chunk 索引扩展为 token 索引
    # topk_chunk_idx: [..., k] -> [..., k, 1]
    chunk_start_idx = topk_chunk_idx.unsqueeze(-1) * chunk_size 
    # offset: [chunk_size] (0, 1, 2, ... chunk_size-1)
    offset = torch.arange(chunk_size, device=attn_weights.device)
    
    # 得到所有被选中 Token 的索引 [..., k * chunk_size]
    token_indices = (chunk_start_idx + offset).reshape(bsz, n_heads, q_len, -1)
    
    # 5. 边界处理：防止 padding 出来的索引越界
    token_indices = token_indices.clamp(max=seq_len - 1)
    
    # 6. 生成布尔 Mask
    mask = torch.zeros((bsz, n_heads, q_len, seq_len), dtype=torch.bool, device=attn_weights.device)
    mask.scatter_(-1, token_indices, True)
    
    return mask
def local_heavy_hitter_mask(attn_weights, token_budget, chunk_size):
    # attn_weights (BS, head, query, keys)

    # expend attn_weights to be divisible by chunk_size
    seq_length = attn_weights.shape[-1]
    padding_length = chunk_size - ((seq_length - 1) % chunk_size + 1)
    attn_weights = torch.cat(
        [
            attn_weights,
            torch.ones(
                (
                    attn_weights.shape[0],
                    attn_weights.shape[1],
                    attn_weights.shape[2],
                    padding_length,
                ),
                device=attn_weights.device,
            )
            * torch.tensor(torch.finfo(attn_weights.dtype).min),
        ],
        dim=-1,
    )

    # chunk attn_weights into chunk_size tokens
    chunk_attn_weights = attn_weights.reshape(
        attn_weights.shape[0],
        attn_weights.shape[1],
        attn_weights.shape[2],
        attn_weights.shape[3] // chunk_size,
        chunk_size,
    ).amax(dim=-1)

    _, topk = chunk_attn_weights.topk(
        k=min(max(3, token_budget // chunk_size), chunk_attn_weights.size(-1)), dim=-1
    )
    # repeat topk chunk_size times and recover the original indexes (* chunk_size + arange(chunk_size))
    topk = topk.unsqueeze(-1).repeat(
        1, 1, 1, 1, chunk_size
    ) * chunk_size + torch.arange(chunk_size, device=topk.device)
    topk = topk.reshape(topk.shape[0], topk.shape[1], topk.shape[2], -1)
    mask_bottom = torch.zeros_like(attn_weights, dtype=torch.bool)
    mask_bottom.scatter_(-1, topk, True)

    # remove the padding
    mask_bottom = mask_bottom[:, :, :, :seq_length]

    return mask_bottom

def forward(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_value: Optional[Tuple[torch.Tensor]] = None,
    output_attentions: bool = False,
    use_cache: bool = False,
    **kwargs,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
    bsz, q_len, _ = hidden_states.size()

    if q_len > 1 or self.layer_id < 2:
        return self.flash_forward(
            hidden_states,
            attention_mask,
            position_ids,
            past_key_value,
            output_attentions,
            use_cache,
            **kwargs,
        )

    query_states = (
        self.q_proj(hidden_states)
        .view(bsz, q_len, self.num_attention_heads, self.head_dim)
        .transpose(1, 2)
    )
    key_states = (
        self.k_proj(hidden_states)
        .view(bsz, q_len, self.num_key_value_heads, self.head_dim)
        .transpose(1, 2)
    )
    value_states = (
        self.v_proj(hidden_states)
        .view(bsz, q_len, self.num_key_value_heads, self.head_dim)
        .transpose(1, 2)
    )
    
    # New cache format
    if isinstance(past_key_value, DynamicCache):
        kv_seq_len = past_key_value.get_seq_length()
    # Legacy cache format
    else:
        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None:
            assert isinstance(past_key_value, tuple)
            kv_seq_len += past_key_value[0].shape[-2]
    
    cos, sin = self.rotary_emb(value_states, position_ids.to(value_states.device))
    query_states, key_states = apply_rotary_pos_emb(
        query_states, key_states, cos, sin, position_ids
    )
    # [bsz, nh, t, hd]

    # New cache format
    if isinstance(past_key_value, DynamicCache):
        if use_cache:
            key_states, value_states = past_key_value.update(key_states, value_states, layer_idx=self.layer_idx)
    # Legacy cache format
    else:
        if past_key_value is not None:
            # reuse k, v, self_attention
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)
        past_key_value = (key_states, value_states) if use_cache else None

    key_states = repeat_kv(key_states, self.num_key_value_groups)
    value_states = repeat_kv(value_states, self.num_key_value_groups)

    attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(
        self.head_dim
    )

    sign = (query_states > 0) + (~(query_states > 0)) * -1
    max_key = key_states * sign
    postive_query = query_states * sign

    # expend max_key to be divisible by chunk_size
    seq_length = max_key.shape[-2]
    padding_length = self.chunk_size - ((seq_length - 1) % self.chunk_size + 1)
    max_key = torch.cat(
        [
            max_key,
            torch.ones(
                (max_key.shape[0], max_key.shape[1], padding_length, max_key.shape[3]),
                device=max_key.device,
            )
            * torch.tensor(torch.finfo(max_key.dtype).min),
        ],
        dim=-2,
    )

    # chunk max_key into chunk_size tokens
    chunk_max_key = max_key.reshape(
        max_key.shape[0],
        max_key.shape[1],
        max_key.shape[2] // self.chunk_size,
        self.chunk_size,
        max_key.shape[3],
    ).amax(dim=-2)

    # duplicate chunk_max_key chunk_size times
    chunk_max_key = chunk_max_key.unsqueeze(-2).repeat(1, 1, 1, self.chunk_size, 1)
    # reshape chunk_max_key to the original shape
    chunk_max_key = chunk_max_key.reshape(
        chunk_max_key.shape[0], chunk_max_key.shape[1], -1, chunk_max_key.shape[-1]
    )[:, :, :seq_length, :]

    quantized_weight = torch.matmul(
        postive_query.float(),
        chunk_max_key.transpose(2, 3),
    )

    if attn_weights.size() != (bsz, self.num_attention_heads, q_len, kv_seq_len):
        raise ValueError(
            f"Attention weights should be of size {(bsz, self.num_attention_heads, q_len, kv_seq_len)}, but is"
            f" {attn_weights.size()}"
        )

    if attention_mask is not None:
        if attention_mask.size() != (bsz, 1, q_len, kv_seq_len):
            raise ValueError(
                f"Attention mask should be of size {(bsz, 1, q_len, kv_seq_len)}, but is {attention_mask.size()}"
            )
        attn_weights = attn_weights + attention_mask
        attn_weights = torch.max(
            attn_weights, torch.tensor(torch.finfo(attn_weights.dtype).min)
        )
        quantized_weight = quantized_weight + attention_mask
        quantized_weight = torch.max(
            quantized_weight, torch.tensor(torch.finfo(quantized_weight.dtype).min)
        )

    token_budget = min(kv_seq_len, self.token_budget)

    attn_weights_for_selection = quantized_weight

    if token_budget > 0:
        mask_bottom = local_heavy_hitter_mask(
            attn_weights_for_selection, token_budget, self.chunk_size
        )  # Default: No padding applied to input
    else:
        mask_bottom = torch.zeros_like(attn_weights_for_selection, dtype=torch.bool)

    mask_bottom = torch.tril(mask_bottom, diagonal=position_ids[0][0].item())
    attn_weights[~mask_bottom] = torch.tensor(torch.finfo(attn_weights.dtype).min)

    # upcast attention to fp32
    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(
        query_states.dtype
    )
    attn_output = torch.matmul(attn_weights, value_states)

    if attn_output.size() != (bsz, self.num_attention_heads, q_len, self.head_dim):
        raise ValueError(
            f"`attn_output` should be of size {(bsz, self.num_attention_heads, q_len, self.head_dim)}, but is"
            f" {attn_output.size()}"
        )

    attn_output = attn_output.transpose(1, 2)
    attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

    attn_output = self.o_proj(attn_output)

    if not output_attentions:
        attn_weights = None

    return attn_output, attn_weights, past_key_value



def qwen3_quest_forward_cache(
    self,
    hidden_states: torch.Tensor,
    position_embeddings: tuple[torch.Tensor, torch.Tensor],  # 必须参数，与官方一致
    attention_mask: Optional[torch.Tensor] = None,
    past_key_values: Optional[Cache] = None,
    cache_position: Optional[torch.LongTensor] = None,
    **kwargs 
) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
    bsz, q_len, _ = hidden_states.size()

    if q_len > 1 or self.layer_id < 2:
        return self.flash_forward(
            hidden_states=hidden_states,
            position_embeddings=position_embeddings,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            cache_position=cache_position,
            **kwargs,
        )

    input_shape = hidden_states.shape[:-1]
    hidden_shape = (*input_shape, -1, self.head_dim)

    query_states = self.q_norm(self.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
    key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
    value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
    
    # New cache format
    if isinstance(past_key_values, DynamicCache):
        kv_seq_len = past_key_values.get_seq_length()
    # Legacy cache format
    else:
        kv_seq_len = key_states.shape[-2]
        if past_key_values is not None:
            assert isinstance(past_key_values, tuple)
            kv_seq_len += past_key_values[0].shape[-2]
    
    cos, sin = position_embeddings
    query_states, key_states = apply_rotary_pos_emb(
        query_states, key_states, cos, sin
    )
    # [bsz, nh, t, hd]

    # New cache format
    if isinstance(past_key_values, DynamicCache):
        cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
        key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx, cache_kwargs)
    # Legacy cache format
    # else:
    #     if past_key_value is not None:
    #         # reuse k, v, self_attention
    #         key_states = torch.cat([past_key_value[0], key_states], dim=2)
    #         value_states = torch.cat([past_key_value[1], value_states], dim=2)
    #     past_key_value = (key_states, value_states) if use_cache else None

    key_states = repeat_kv(key_states, self.num_key_value_groups)
    value_states = repeat_kv(value_states, self.num_key_value_groups)

    attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(
        self.head_dim
    )

    sign = (query_states > 0) + (~(query_states > 0)) * -1
    max_key = key_states * sign
    postive_query = query_states * sign

    # expend max_key to be divisible by chunk_size
    seq_length = max_key.shape[-2]
    padding_length = self.chunk_size - ((seq_length - 1) % self.chunk_size + 1)
    max_key = torch.cat(
        [
            max_key,
            torch.ones(
                (max_key.shape[0], max_key.shape[1], padding_length, max_key.shape[3]),
                device=max_key.device,
            )
            * torch.tensor(torch.finfo(max_key.dtype).min),
        ],
        dim=-2,
    )

    # chunk max_key into chunk_size tokens
    chunk_max_key = max_key.reshape(
        max_key.shape[0],
        max_key.shape[1],
        max_key.shape[2] // self.chunk_size,
        self.chunk_size,
        max_key.shape[3],
    ).amax(dim=-2)

    # duplicate chunk_max_key chunk_size times
    chunk_max_key = chunk_max_key.unsqueeze(-2).repeat(1, 1, 1, self.chunk_size, 1)
    # reshape chunk_max_key to the original shape
    chunk_max_key = chunk_max_key.reshape(
        chunk_max_key.shape[0], chunk_max_key.shape[1], -1, chunk_max_key.shape[-1]
    )[:, :, :seq_length, :]

    quantized_weight = torch.matmul(
        postive_query.float(),
        chunk_max_key.transpose(2, 3),
    )

    if attn_weights.size() != (bsz, self.config.num_attention_heads, q_len, kv_seq_len):
        raise ValueError(
            f"Attention weights should be of size {(bsz, self.config.num_attention_heads, q_len, kv_seq_len)}, but is"
            f" {attn_weights.size()}"
        )

    if attention_mask is not None:
        if attention_mask.size() != (bsz, 1, q_len, kv_seq_len):
            raise ValueError(
                f"Attention mask should be of size {(bsz, 1, q_len, kv_seq_len)}, but is {attention_mask.size()}"
            )
        attn_weights = attn_weights + attention_mask
        attn_weights = torch.max(
            attn_weights, torch.tensor(torch.finfo(attn_weights.dtype).min)
        )
        quantized_weight = quantized_weight + attention_mask
        quantized_weight = torch.max(
            quantized_weight, torch.tensor(torch.finfo(quantized_weight.dtype).min)
        )

    token_budget = min(kv_seq_len, self.token_budget)

    attn_weights_for_selection = quantized_weight

    if token_budget > 0:
        mask_bottom = local_heavy_hitter_mask(
            attn_weights_for_selection, token_budget, self.chunk_size
        )  # Default: No padding applied to input
    else:
        mask_bottom = torch.zeros_like(attn_weights_for_selection, dtype=torch.bool)


    current_pos = cache_position[0].item() if cache_position is not None else 0
    mask_bottom = torch.tril(mask_bottom, diagonal=current_pos)
    attn_weights[~mask_bottom] = torch.tensor(torch.finfo(attn_weights.dtype).min)

    # upcast attention to fp32
    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(
        query_states.dtype
    )
    attn_output = torch.matmul(attn_weights, value_states)

    if attn_output.size() != (bsz, self.config.num_attention_heads, q_len, self.head_dim):
        raise ValueError(
            f"`attn_output` should be of size {(bsz, self.config.num_attention_heads, q_len, self.head_dim)}, but is"
            f" {attn_output.size()}"
        )

    attn_output = attn_output.transpose(1, 2)
    attn_output = attn_output.reshape(*input_shape, -1)

    attn_output = self.o_proj(attn_output)


    return attn_output, attn_weights

# Qwen3 Quest Attention Forward - 支持 tuple 格式的 past_key_value
def qwen3_quest_forward_tuple(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_value: Optional[Tuple[torch.Tensor]] = None,
    output_attentions: bool = False,
    use_cache: bool = False,
    padding_mask: Optional[torch.LongTensor] = None,
    position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
    """
    Quest Attention for Qwen3 with tuple-style KV cache

    适配要点：
    1. 使用 tuple 格式的 past_key_value (key, value)
    2. 正确应用 q_norm 和 k_norm（Qwen3 特有）
    3. 在 decode 阶段使用 Quest 稀疏化
    4. 返回值包含 past_key_value tuple
    """
    bsz, q_len, _ = hidden_states.size()

    # 1. Prefill 阶段或前两层：使用原始实现
    if q_len > 1 or getattr(self, "layer_id", 0) < 2:
        return self.flash_forward(
            hidden_states,
            attention_mask,
            position_ids,
            past_key_value,
            output_attentions,
            use_cache,
            padding_mask=padding_mask,
            position_embeddings=position_embeddings,
        )

    # 2. QKV 投影（与 Qwen3 官方实现一致）
    input_shape = hidden_states.shape[:-1]
    hidden_shape = (*input_shape, -1, self.head_dim)

    # 关键：必须使用 q_norm 和 k_norm（Qwen3 特有）
    query_states = self.q_norm(self.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
    key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
    value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

    # 3. 计算 kv_seq_len（包含过去的 cache）
    kv_seq_len = key_states.shape[-2]
    if past_key_value is not None:
        kv_seq_len += past_key_value[0].shape[-2]

    # 4. 应用 RoPE
    if position_embeddings is None:
        cos, sin = self.rotary_emb(value_states, position_ids)
    else:
        cos, sin = position_embeddings

    from transformers.models.qwen3.modeling_qwen3 import apply_rotary_pos_emb, repeat_kv
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

    # 5. 拼接 KV Cache（tuple 模式）
    if past_key_value is not None:
        key_states = torch.cat([past_key_value[0], key_states], dim=2)
        value_states = torch.cat([past_key_value[1], value_states], dim=2)

    past_key_value = (key_states, value_states) if use_cache else None

    # 6. GQA 扩展
    key_states_full = repeat_kv(key_states, self.num_key_value_groups)
    value_states_full = repeat_kv(value_states, self.num_key_value_groups)

    # 7. Quest 稀疏化核心 - 计算初始 attention weights
    attn_weights = torch.matmul(query_states, key_states_full.transpose(2, 3)) * self.scaling

    # 8. 计算 quantized weight 用于 token 选择
    sign = (query_states > 0) + (~(query_states > 0)) * -1
    max_key = key_states_full * sign
    positive_query = query_states * sign

    # Expand max_key to be divisible by chunk_size
    seq_length = max_key.shape[-2]
    padding_length = self.chunk_size - ((seq_length - 1) % self.chunk_size + 1)
    max_key = torch.cat(
        [
            max_key,
            torch.ones(
                (max_key.shape[0], max_key.shape[1], padding_length, max_key.shape[3]),
                device=max_key.device,
            )
            * torch.tensor(torch.finfo(max_key.dtype).min),
        ],
        dim=-2,
    )

    # Chunk max_key into chunk_size tokens
    chunk_max_key = max_key.reshape(
        max_key.shape[0],
        max_key.shape[1],
        max_key.shape[2] // self.chunk_size,
        self.chunk_size,
        max_key.shape[3],
    ).amax(dim=-2)

    # Duplicate chunk_max_key chunk_size times
    chunk_max_key = chunk_max_key.unsqueeze(-2).repeat(1, 1, 1, self.chunk_size, 1)
    chunk_max_key = chunk_max_key.reshape(
        chunk_max_key.shape[0], chunk_max_key.shape[1], -1, chunk_max_key.shape[-1]
    )[:, :, :seq_length, :]

    quantized_weight = torch.matmul(
        positive_query.float(),
        chunk_max_key.transpose(2, 3),
    )

    # 9. 应用 attention_mask
    if attention_mask is not None:
        attn_weights = attn_weights + attention_mask
        attn_weights = torch.max(
            attn_weights, torch.tensor(torch.finfo(attn_weights.dtype).min)
        )
        quantized_weight = quantized_weight + attention_mask
        quantized_weight = torch.max(
            quantized_weight, torch.tensor(torch.finfo(quantized_weight.dtype).min)
        )

    # 10. 生成稀疏 Mask
    token_budget = min(kv_seq_len, self.token_budget)
    if token_budget > 0:
        mask_bottom = local_heavy_hitter_mask(quantized_weight, token_budget, self.chunk_size)
    else:
        mask_bottom = torch.zeros_like(attn_weights, dtype=torch.bool)

    # 11. 应用因果掩码（对于 decode 阶段）
    if position_ids is not None:
        mask_bottom = torch.tril(mask_bottom, diagonal=position_ids[0][0].item())

    attn_weights[~mask_bottom] = torch.tensor(torch.finfo(attn_weights.dtype).min)

    # 12. 计算最终输出
    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(
        query_states.dtype
    )
    attn_output = torch.matmul(attn_weights, value_states_full)

    attn_output = attn_output.transpose(1, 2).contiguous()
    attn_output = attn_output.reshape(*input_shape, -1)
    attn_output = self.o_proj(attn_output)

    if not output_attentions:
        attn_weights = None

    return attn_output, attn_weights, past_key_value


# 假设这些辅助函数已经在你的 quest_attention.py 中定义
# from .utils import apply_rotary_pos_emb, repeat_kv, local_heavy_hitter_mask
#qwen3 版本的 Quest Attention 前向函数
# # n能跑但全季冒号::::::::::::::::::::::::::::::::::
def qwen3_quest_forward(
    self,
    hidden_states: torch.Tensor,
    position_embeddings: tuple[torch.Tensor, torch.Tensor],  # 必须参数，与官方一致
    attention_mask: Optional[torch.Tensor] = None,
    past_key_values: Optional[Cache] = None,
    cache_position: Optional[torch.LongTensor] = None,
    **kwargs 
) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
    """
    Quest Attention for Qwen3 - 完全兼容官方实现

    关键修复：
    1. position_embeddings 作为必需参数（与官方一致）
    2. 正确应用 q_norm 和 k_norm
    3. KV Cache 更新在 RoPE 之后
    4. 返回值匹配 DecoderLayer 期望
    """
    bsz, q_len, _ = hidden_states.size()

    # 1. Prefill 或前两层：使用原始实现
    if q_len > 1 or getattr(self, "layer_id", 0) < 2:
        print("full^^^^questno")
        return self._original_forward(
            hidden_states=hidden_states,
            position_embeddings=position_embeddings,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            cache_position=cache_position,
            **kwargs,
        )

    # 2. QKV 投影（完全复制官方逻辑）
    input_shape = hidden_states.shape[:-1]
    hidden_shape = (*input_shape, -1, self.head_dim)

    # 关键：必须使用 q_norm 和 k_norm（Qwen3 特有）
    query_states = self.q_norm(self.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
    key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
    value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

    # 3. 应用 RoPE（使用外部传入的 position_embeddings）
    cos, sin = position_embeddings

    # 动态导入以避免循环依赖
    from transformers.models.qwen3.modeling_qwen3 import apply_rotary_pos_emb, repeat_kv
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

    # 4. 更新 KV Cache（在 RoPE 之后，与官方一致）
    if past_key_values is not None:
        cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
        key_states, value_states = past_key_values.update(
            key_states, value_states, self.layer_idx, cache_kwargs
        )

    kv_seq_len = key_states.shape[-2]

    # 5. GQA 扩展（用于后续 attention 计算）
    key_states_full = repeat_kv(key_states, self.num_key_value_groups)
    value_states_full = repeat_kv(value_states, self.num_key_value_groups)

    # 6. Quest 稀疏化核心 - 计算初始 attention weights
    # 使用官方的 scaling factor
    attn_weights = torch.matmul(query_states, key_states_full.transpose(2, 3)) * self.scaling

    #

    sign = (query_states > 0) + (~(query_states > 0)) * -1
    max_key = key_states_full * sign
    positive_query = query_states * sign

    seq_length = max_key.shape[-2]
    padding_length = self.chunk_size - ((seq_length - 1) % self.chunk_size + 1)
    max_key = torch.cat(
        [
            max_key,
            torch.ones(
                (max_key.shape[0], max_key.shape[1], padding_length, max_key.shape[3]),
                device=max_key.device,
            )
            * torch.tensor(torch.finfo(max_key.dtype).min),
        ],
        dim=-2,
    )

    # chunk max_key into chunk_size tokens
    chunk_max_key = max_key.reshape(
        max_key.shape[0],
        max_key.shape[1],
        max_key.shape[2] // self.chunk_size,
        self.chunk_size,
        max_key.shape[3],
    ).amax(dim=-2)

    # duplicate chunk_max_key chunk_size times
    chunk_max_key = chunk_max_key.unsqueeze(-2).repeat(1, 1, 1, self.chunk_size, 1)
    # reshape chunk_max_key to the original shape
    chunk_max_key = chunk_max_key.reshape(
        chunk_max_key.shape[0], chunk_max_key.shape[1], -1, chunk_max_key.shape[-1]
    )[:, :, :seq_length, :]

    quantized_weight = torch.matmul(
        positive_query.float(),
        chunk_max_key.transpose(2, 3),
    )
    # 计算近似的 attention 分数（用于 chunk 选择）
    # quantized_weight = torch.matmul(positive_query, max_key.transpose(2, 3)) * self.scaling

    # 7. 应用 attention_mask（如果存在）
    if attention_mask is not None:
        # 修复：支持广播，Qwen3 的 mask 可能是不同形状
        attn_weights = attn_weights + attention_mask
        quantized_weight = quantized_weight + attention_mask

    # 8. 生成稀疏 Mask（使用 local_heavy_hitter_mask）
    token_budget = min(kv_seq_len, self.token_budget)
    if token_budget > 0:
        mask_bottom = local_heavy_hitter_mask(quantized_weight, token_budget, self.chunk_size)
    else:
        mask_bottom = torch.zeros_like(attn_weights, dtype=torch.bool)

    # 9. 应用稀疏掩码到 attention weights
    # mask_bottom 中 True 表示保留，False 表示丢弃
    # 注意：在 decode 阶段 (q_len=1)，不需要额外的因果掩码
    # 因为 query 只有 1 个 token，它可以看到所有 KV cache
    attn_weights = attn_weights.masked_fill(~mask_bottom, torch.finfo(attn_weights.dtype).min)

    # 10. 计算 softmax 和最终输出
    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
    attn_output = torch.matmul(attn_weights, value_states_full)

    # 11. 重塑输出（与官方一致）
    attn_output = attn_output.reshape(*input_shape, -1).contiguous()
    attn_output = self.o_proj(attn_output)
    # print("使用 Quest Attention 生成 token。")
    # 12. 返回值：(attn_output, attn_weights)，与官方签名一致
    return attn_output, attn_weights  # 不返回 attn_weights 以节省内存


def qwen3_quest_forward_cache_gemini(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[Cache] = None, # 强制要求 Cache 格式
    output_attentions: bool = False,
    use_cache: bool = False,
    cache_position: Optional[torch.LongTensor] = None,
    position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None, # 适配新版 Transformers
    **kwargs,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]: # 返回值不含 past_key_values
    
    bsz, q_len, _ = hidden_states.size()

    # --- 1. Prefill 阶段 (q_len > 1) ---
    # Quest 在 Prefill 阶段无收益，直接调用原生 Flash Attention
    # 注意：这里我们假设原生的 forward 已经被你存为 flash_forward
    if q_len > 1:
        # 原生 forward 通常返回 (output, weights, past_kv)，我们需要截断第三个返回值
        out = self.flash_forward(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            **kwargs
        )
        return out[0], out[1] # 丢弃 past_key_values

    # --- 2. Qwen 特有的 Proj + Norm 流程 ---
    input_shape = hidden_states.shape[:-1]
    hidden_shape = (*input_shape, -1, self.head_dim)

    # 投影
    query_states = self.q_proj(hidden_states).view(hidden_shape)
    key_states = self.k_proj(hidden_states).view(hidden_shape)
    value_states = self.v_proj(hidden_states).view(hidden_shape)

    # Q-Norm 和 K-Norm (Qwen2/3 核心差异)
    query_states = self.q_norm(query_states).transpose(1, 2)
    key_states = self.k_norm(key_states).transpose(1, 2)
    value_states = value_states.transpose(1, 2)

    # --- 3. RoPE ---
    if position_embeddings is None:
        cos, sin = self.rotary_emb(value_states, position_ids)
    else:
        cos, sin = position_embeddings
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

    # --- 4. Cache 更新 (必须使用 Cache 对象) ---
    if past_key_values is not None:
        # cache_position 在 Transformers 4.36+ 中是必须的
        cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
        key_states, value_states = past_key_values.update(
            key_states, value_states, self.layer_idx, cache_kwargs
        )

    # --- 5. GQA 展开 ---
    # 为了计算 Attention Score，我们需要展开 KV
    key_states_expanded = repeat_kv(key_states, self.num_key_value_groups)
    value_states_expanded = repeat_kv(value_states, self.num_key_value_groups)

    # --- 6. Quest 逻辑 (Accuracy Fix) ---
    
    # 只有当 seq_len 超过 budget 时才进行稀疏化，否则退化为标准 Attention
    kv_seq_len = key_states.shape[-2]
    token_budget = min(kv_seq_len, self.token_budget)
    
    # 计算 Scaling
    scaling =  self.head_dim ** -0.5
    
    # [关键优化]：直接计算全量 Attention Scores
    # 原因：Q-Norm 导致 Quest 的 "Max(K)" 近似失效。
    # 在 Decode (q=1) 时，这一步只是 Vector-Matrix 乘法，速度非常快。
    attn_weights = torch.matmul(query_states, key_states_expanded.transpose(2, 3)) * scaling

    # 如果需要稀疏化 (Sparse Attention)
    if 0 < token_budget < kv_seq_len:
        # 调用优化后的 Mask 函数，基于真实分数筛选
        mask_bottom = optimized_heavy_hitter_mask(attn_weights, token_budget, self.chunk_size)
        
        # 应用 Mask (保留 True 的部分，其他设为 -inf)
        min_value = torch.finfo(attn_weights.dtype).min
        attn_weights = attn_weights.masked_fill(~mask_bottom, min_value)

    # 加上 Attention Mask (如果有，比如 padding mask)
    if attention_mask is not None:
        attn_weights = attn_weights + attention_mask

    # --- 7. Softmax & Output ---
    # Upcast to fp32 for stability
    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
    
    # 乘 Value
    attn_output = torch.matmul(attn_weights, value_states_expanded)

    # Reshape & Output Proj
    attn_output = attn_output.transpose(1, 2).contiguous().reshape(*input_shape, -1)
    attn_output = self.o_proj(attn_output)

    if not output_attentions:
        attn_weights = None

    # --- 8. 返回 ---
    # 根据你的要求，这里只返回 2 个元素，不要 past_key_values
    return attn_output, attn_weights

global layer_id
layer_id = 28


def enable_quest_attention_eval(model, args):

    for name, module in reversed(model._modules.items()):
        if len(list(module.children())) > 0:
            enable_quest_attention_eval(
                module,
                args,
            )

        global layer_id
        # 动态检查 Attention 类型（支持 Qwen3Attention 通过类名）
        if isinstance(module, (LlamaAttention, MistralAttention)):
            # For Llama and Mistral models
            layer_id -= 1
            model._modules[name].layer_id = layer_id
            # 保存原始 forward（重要！）
            model._modules[name].flash_forward = model._modules[name].forward
            model._modules[name].forward = types.MethodType(
                forward, model._modules[name]
            )

            model._modules[name].token_budget = args.token_budget
            model._modules[name].chunk_size = args.chunk_size

        elif isinstance(module, Qwen3Attention):
            # For Qwen3 model with tuple KV cache
            layer_id -= 1
            model._modules[name].layer_id = layer_id
            # 保存原始 forward（重要！）
            model._modules[name].flash_forward = model._modules[name].forward
            model._modules[name].forward = types.MethodType(
                qwen3_quest_forward_cache_gemini, model._modules[name]
            )

            model._modules[name].token_budget = args.token_budget
            model._modules[name].chunk_size = args.chunk_size

