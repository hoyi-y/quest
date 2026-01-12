"""
Qwen3 模型的 Tuple KV Cache 支持 (优化适配版)
"""

import torch
import math
import torch.nn as nn
from typing import Optional, Tuple, Union
import transformers

# ============================================
# 1. 核心工具函数：处理 GQA
# ============================================
def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    Qwen3 使用 Grouped Query Attention (GQA)，需要将 KV 扩展到和 Q 的头数一致
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)

# ============================================
# 2. FlashAttention 前向传播 (修复版)
# ============================================
def old_qwen3_flash_attention_forward(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_value: Optional[Tuple[torch.Tensor]] = None,
    output_attentions: bool = False,
    use_cache: bool = False,
    cache_position: Optional[torch.LongTensor] = None,
    position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None, 
    **kwargs,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
    
    bsz, q_len, _ = hidden_states.size()

    # 修复 AttributeError: 使用 config 确保兼容性
    num_heads = self.config.num_attention_heads
    num_key_value_heads = self.config.num_key_value_heads
    num_key_value_groups = num_heads // num_key_value_heads
    head_dim = self.config.hidden_size // num_heads

    # 投影
    query_states = self.q_proj(hidden_states).view(bsz, q_len, num_heads, head_dim).transpose(1, 2)
    key_states = self.k_proj(hidden_states).view(bsz, q_len, num_key_value_heads, head_dim).transpose(1, 2)
    value_states = self.v_proj(hidden_states).view(bsz, q_len, num_key_value_heads, head_dim).transpose(1, 2)

    # 处理 RoPE (适配 v4.45+)
    # 注意：新版 Qwen 会自动在 Model 层生成 cos/sin
    kv_seq_len = key_states.shape[-2]
    if past_key_value is not None:
        kv_seq_len += past_key_value[0].shape[-2]
    
    # 动态获取 RoPE
    cos, sin = self.rotary_emb(value_states, position_ids)
    from transformers.models.qwen2.modeling_qwen2 import apply_rotary_pos_emb
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

    # 拼接 KV Cache (Tuple 模式)
    if past_key_value is not None:
        key_states = torch.cat([past_key_value[0], key_states], dim=2)
        value_states = torch.cat([past_key_value[1], value_states], dim=2)

    present_key_value = (key_states, value_states) if use_cache else None

    # GQA 处理：手动计算模式下必须 repeat
    key_states = repeat_kv(key_states, num_key_value_groups)
    value_states = repeat_kv(value_states, num_key_value_groups)

    # 注意力计算 (Eager 模式)
    attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(head_dim)

    if attention_mask is not None:
        # 注意：Qwen3 的 mask 形状可能与 Llama 不同，这里进行切片适配
        attn_weights = attn_weights + attention_mask

    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
    attn_output = torch.matmul(attn_weights, value_states)

    attn_output = attn_output.transpose(1, 2).contiguous()
    attn_output = attn_output.reshape(bsz, q_len, self.config.hidden_size)
    attn_output = self.o_proj(attn_output)

    return attn_output, None, present_key_value

# ============================================
# 3. Model 层前向传播 (修复 Mask 和位置逻辑)
# ============================================
def old_qwen3_model_forward(
    self,
    input_ids: torch.LongTensor = None,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[Tuple[torch.Tensor]] = None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    use_cache: Optional[bool] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    return_dict: Optional[bool] = None,
) -> Union[Tuple, transformers.modeling_outputs.BaseModelOutputWithPast]:
    
    use_cache = use_cache if use_cache is not None else self.config.use_cache
    
    if input_ids is not None:
        batch_size, seq_length = input_ids.shape
    else:
        batch_size, seq_length, _ = inputs_embeds.shape

    past_key_values_length = 0
    if past_key_values is not None:
        # 修改：Tuple 模式下获取长度的正确姿势
        past_key_values_length = past_key_values[0].shape[2]

    if position_ids is None:
        device = input_ids.device if input_ids is not None else inputs_embeds.device
        position_ids = torch.arange(
            past_key_values_length, seq_length + past_key_values_length, dtype=torch.long, device=device
        )
        position_ids = position_ids.unsqueeze(0)

    if inputs_embeds is None:
        inputs_embeds = self.embed_tokens(input_ids)

    # 构建简单的 Causal Mask
    # Qwen3 在补丁模式下不再调用复杂的 _prepare_decoder_attention_mask
    causal_mask = torch.full(
        (seq_length, seq_length + past_key_values_length), 
        float("-inf"), 
        device=inputs_embeds.device
    )
    causal_mask = torch.triu(causal_mask, diagonal=past_key_values_length + 1)
    causal_mask = causal_mask[None, None, :, :]

    hidden_states = inputs_embeds
    next_decoder_cache = () if use_cache else None

    for idx, decoder_layer in enumerate(self.layers):
        past_key_value = past_key_values[idx] if past_key_values is not None else None
        
        layer_outputs = decoder_layer(
            hidden_states,
            attention_mask=causal_mask,
            position_ids=position_ids,
            past_key_value=past_key_value, # 注意：Qwen 层内部通常用单数
            output_attentions=output_attentions,
            use_cache=use_cache,
        )
        hidden_states = layer_outputs[0]
        if use_cache:
            next_decoder_cache += (layer_outputs[1],)

    hidden_states = self.norm(hidden_states)

    return transformers.modeling_outputs.BaseModelOutputWithPast(
        last_hidden_state=hidden_states,
        past_key_values=next_decoder_cache,
        hidden_states=None,
        attentions=None,
    )

def old_qwen3_decoder_layer_forward(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_value: Optional[Tuple[torch.Tensor]] = None,
    output_attentions: Optional[bool] = False,
    use_cache: Optional[bool] = False,
    **kwargs,
) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
    # 1. 正常的残差逻辑：LayerNorm -> Attention
    residual = hidden_states
    hidden_states = self.input_layernorm(hidden_states)

    # 核心调用：这里会进入我们之前改写的 old_qwen3_flash_attention_forward
    hidden_states, self_attn_weights, present_key_value = self.self_attn(
        hidden_states=hidden_states,
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_value=past_key_value, # 注意确保这里是单数，与 Attention 对应
        output_attentions=output_attentions,
        use_cache=use_cache,
        **kwargs,
    )
    hidden_states = residual + hidden_states

    # 2. MLP 部分：保持原样即可
    residual = hidden_states
    hidden_states = self.post_attention_layernorm(hidden_states)
    hidden_states = self.mlp(hidden_states)
    hidden_states = residual + hidden_states

    outputs = (hidden_states,)
    if output_attentions:
        outputs += (self_attn_weights,)
    if use_cache:
        outputs += (present_key_value,)

    return outputs


def old_qwen3_for_causal_lm_forward(
    self,
    input_ids: torch.LongTensor = None,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[Tuple[torch.Tensor]] = None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    labels: Optional[torch.LongTensor] = None,
    use_cache: Optional[bool] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    return_dict: Optional[bool] = None,
    **kwargs,
) -> Union[Tuple, transformers.modeling_outputs.CausalLMOutputWithPast]:
    
    return_dict = return_dict if return_dict is not None else self.config.use_return_dict

    # 调用之前改写的 Model 层
    outputs = self.model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_values=past_key_values,
        inputs_embeds=inputs_embeds,
        use_cache=use_cache,
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
        return_dict=return_dict,
    )

    hidden_states = outputs[0]
    
    # 【优化点】推理模式下只拿最后一个 token 的 hidden_state 算 logits
    if not self.training:
        logits = self.lm_head(hidden_states[:, -1:, :])
    else:
        logits = self.lm_head(hidden_states)

    loss = None
    if labels is not None:
        # 如果是训练模式，计算交叉熵损失
        loss_fct = torch.nn.CrossEntropyLoss()
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        loss = loss_fct(shift_logits.view(-1, self.config.vocab_size), shift_labels.view(-1))

    if not return_dict:
        output = (logits,) + outputs[1:]
        return (loss,) + output if loss is not None else output

    return transformers.modeling_outputs.CausalLMOutputWithPast(
        loss=loss,
        logits=logits,
        past_key_values=outputs.past_key_values,
        hidden_states=outputs.hidden_states,
        attentions=outputs.attentions,
    )

# ============================================
# 4. 启用补丁
# ============================================
def enable_tuple_kv_cache_for_qwen3():
    print("🚀 Full Patching Qwen3 for Tuple KV Cache...")
    
    # 动态获取 Qwen2/3 模块逻辑
    import transformers.models.qwen3.modeling_qwen3 as mod 
    
    # 替换整个调用链
    mod.Qwen3ForCausalLM.forward = old_qwen3_for_causal_lm_forward
    mod.Qwen3Model.forward = old_qwen3_model_forward
    mod.Qwen3DecoderLayer.forward = old_qwen3_decoder_layer_forward
    
    # 针对 Attention，覆盖所有可能的实现类
    mod.Qwen3Attention.forward = old_qwen3_flash_attention_forward
    if hasattr(mod, "Qwen3FlashAttention2"):
        mod.Qwen3FlashAttention2.forward = old_qwen3_flash_attention_forward
    if hasattr(mod, "Qwen3SdpaAttention"):
        mod.Qwen3SdpaAttention.forward = old_qwen3_flash_attention_forward

    print("✅ Full Chain Patched Successfully.")