"""
Qwen3 模型的 Tuple KV Cache 支持
注意: Qwen3 有独立的模型类，不同于 Qwen2
"""
 
import torch
import math
import torch.nn.functional as F
from typing import Optional, Tuple, Union, List
 
# ============================================
# 导入 Qwen3 模型类
# ============================================
try:
    from transformers.models.qwen3 import modeling_qwen3
    from transformers.models.qwen3.modeling_qwen3 import (
        Qwen3Attention,
        Qwen3ForCausalLM,
        Qwen3Model,
        Qwen3DecoderLayer,
    )
    from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
    from torch.nn import CrossEntropyLoss
    print("✓ 成功导入 transformers.models.qwen3")
except ImportError as e:
    print(f"✗ 导入失败: {e}")
    print("请确保 transformers 版本 >= 4.48.0 (支持 Qwen3)")
    raise
 
 
# ============================================
# FlashAttention 前向传播 (Tuple KV Cache 版本)
# ============================================
 
def old_flash_attention_forward(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[Tuple[torch.Tensor]] = None,  # 注意: Qwen3 用 past_key_values (复数)
    output_attentions: bool = False,
    use_cache: bool = False,
    **kwargs,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
    """
    Qwen3 FlashAttention 前向传播 - Tuple KV Cache 版本
    """
    bsz, q_len, _ = hidden_states.size()
 
    # QKV 投影
    query_states = self.q_proj(hidden_states)
    key_states = self.k_proj(hidden_states)
    value_states = self.v_proj(hidden_states)
 
    # Reshape 和转置
    query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
    key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
    value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
 
    # KV 序列长度
    kv_seq_len = key_states.shape[-2]
    if past_key_values is not None:
        kv_seq_len += past_key_values[0].shape[-2]
    
    # RoPE
    cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
    query_states, key_states = self.rotary_emb.apply_rotary_pos_emb(
        query_states, key_states, cos, sin, position_ids
    )
    
    # GQA 处理
    if self.num_key_value_groups != 1:
        key_states = key_states.repeat_interleave(self.num_key_value_groups, dim=2)
        value_states = value_states.repeat_interleave(self.num_key_value_groups, dim=2)
    
    # 合并 KV Cache (Tuple 格式)
    if past_key_values is not None:
        key_states = torch.cat([past_key_values[0], key_states], dim=2)
        value_states = torch.cat([past_key_values[1], value_states], dim=2)
    
    past_key_values = (key_states, value_states) if use_cache else None
    
    # 注意力计算
    attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)
    
    if attention_mask is not None:
        causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
        attn_weights = attn_weights + causal_mask
    
    # Softmax
    attn_weights = torch.max(
        attn_weights, torch.tensor(torch.finfo(attn_weights.dtype).min, device=attn_weights.device)
    )
    attn_weights = torch.nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
    
    # 输出
    attn_output = torch.matmul(attn_weights, value_states)
    attn_output = attn_output.transpose(1, 2).contiguous()
    attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)
    attn_output = self.o_proj(attn_output)
    
    return attn_output, None, past_key_values
 
 
# ============================================
# Decoder Layer 前向传播 (Tuple KV Cache 版本)
# ============================================
 
def old_qwen3_decoder_layer_forward(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[Tuple[torch.Tensor]] = None,
    output_attentions: bool = False,
    use_cache: bool = False,
    **kwargs,
) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
    """
    Qwen3 Decoder Layer 前向传播 - Tuple KV Cache 版本
    """
    residual = hidden_states
    
    # Self Attention
    hidden_states = self.input_layernorm(hidden_states)
    hidden_states, self_attn_weights, present_key_value = self.self_attn(
        hidden_states=hidden_states,
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_values=past_key_values,  # 注意: 用 past_key_values (复数)
        output_attentions=output_attentions,
        use_cache=use_cache,
    )
    hidden_states = residual + hidden_states
    
    # MLP
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
 
 
# ============================================
# Model 前向传播 (Tuple KV Cache 版本)
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
) -> Union[Tuple, BaseModelOutputWithPast]:
    """
    Qwen3 Model 前向传播 - Tuple KV Cache 版本
    """
    output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
    output_hidden_states = (
        output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
    )
    use_cache = use_cache if use_cache is not None else self.config.use_cache
    return_dict = return_dict if return_dict is not None else self.config.use_return_dict
    
    # 获取输入
    if input_ids is not None and inputs_embeds is not None:
        raise ValueError("You cannot specify both input_ids and inputs_embeds")
    elif input_ids is not None:
        batch_size, seq_length = input_ids.shape
    elif inputs_embeds is not None:
        batch_size, seq_length, _ = inputs_embeds.shape
    else:
        raise ValueError("You have to specify either input_ids or inputs_embeds")
    
    past_key_values_length = 0
    if past_key_values is not None:
        past_key_values_length = past_key_values[0][0].shape[2]
    
    # Position IDs
    if position_ids is None:
        device = input_ids.device if input_ids is not None else inputs_embeds.device
        position_ids = torch.arange(
            past_key_values_length, seq_length + past_key_values_length, dtype=torch.long, device=device
        )
        position_ids = position_ids.unsqueeze(0).view(-1, seq_length)
    
    # Embeddings
    if inputs_embeds is None:
        inputs_embeds = self.embed_tokens(input_ids)
    
    hidden_states = inputs_embeds
    
    # Causal Mask
    if attention_mask is None:
        attention_mask = torch.ones(
            (batch_size, seq_length + past_key_values_length), dtype=torch.bool, device=hidden_states.device
        )
    
    causal_mask = torch.full(
        (seq_length, seq_length), 
        torch.finfo(hidden_states.dtype).min, 
        device=hidden_states.device
    )
    causal_mask = torch.triu(causal_mask, diagonal=1)
    causal_mask = causal_mask[None, None, :, :].expand(batch_size, 1, -1, -1)
    
    if past_key_values_length > 0:
        causal_mask = torch.cat(
            [torch.zeros(batch_size, 1, seq_length, past_key_values_length, device=hidden_states.device), causal_mask],
            dim=-1
        )
    
    # Output containers
    all_hidden_states = () if output_hidden_states else None
    all_self_attentions = () if output_attentions else None
    next_decoder_cache = () if use_cache else None
    
    # Decoder layers
    for idx, decoder_layer in enumerate(self.layers):
        if output_hidden_states:
            all_hidden_states += (hidden_states,)
        
        past_key_value = past_key_values[idx] if past_key_values is not None else None
        
        layer_outputs = decoder_layer(
            hidden_states,
            attention_mask=causal_mask,
            position_ids=position_ids,
            past_key_values=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
        )
        
        hidden_states = layer_outputs[0]
        
        if use_cache:
            next_decoder_cache += (layer_outputs[2 if output_attentions else 1],)
        
        if output_attentions:
            all_self_attentions += (layer_outputs[1],)
    
    hidden_states = self.norm(hidden_states)
    
    if output_hidden_states:
        all_hidden_states += (hidden_states,)
    
    next_cache = next_decoder_cache if use_cache else None
    
    if not return_dict:
        return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attentions] if v is not None)
    
    return BaseModelOutputWithPast(
        last_hidden_state=hidden_states,
        past_key_values=next_cache,
        hidden_states=all_hidden_states,
        attentions=all_self_attentions,
    )
 
 
# ============================================
# ForCausalLM 前向传播 (Tuple KV Cache 版本)
# ============================================
 
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
) -> Union[Tuple, CausalLMOutputWithPast]:
    """
    Qwen3 ForCausalLM 前向传播 - Tuple KV Cache 版本
    """
    output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
    output_hidden_states = (
        output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
    )
    return_dict = return_dict if return_dict is not None else self.config.use_return_dict
    
    # Decoder forward
    outputs = self.model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_values=past_key_values,  # 注意: 用 past_key_values (复数)
        inputs_embeds=inputs_embeds,
        use_cache=use_cache,
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
        return_dict=return_dict,
    )
    
    hidden_states = outputs[0]
    logits = self.lm_head(hidden_states)
    
    # Loss
    loss = None
    if labels is not None:
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        loss_fct = CrossEntropyLoss()
        shift_logits = shift_logits.view(-1, self.config.vocab_size)
        shift_labels = shift_labels.view(-1)
        loss = loss_fct(shift_logits, shift_labels)
    
    if not return_dict:
        output = (logits,) + outputs[1:]
        return (loss,) + output if loss is not None else output
    
    return CausalLMOutputWithPast(
        loss=loss,
        logits=logits,
        past_key_values=outputs.past_key_values,
        hidden_states=outputs.hidden_states,
        attentions=outputs.attentions,
    )
 
 
# ============================================
# 启用 Tuple KV Cache
# ============================================
 
def enable_tuple_kv_cache_for_qwen():
    """
    为 Qwen3 启用 Tuple 格式 KV Cache
    """
    print("Enabling tuple KV cache for Qwen3")
    
    # 禁用 mask 准备
    modeling_qwen3.Qwen3Model._prepare_decoder_attention_mask = (
        lambda *args, **kwargs: None
    )
    
    # 替换前向传播方法
    modeling_qwen3.Qwen3Model.forward = old_qwen3_model_forward
    modeling_qwen3.Qwen3DecoderLayer.forward = old_qwen3_decoder_layer_forward
    modeling_qwen3.Qwen3Attention.forward = old_flash_attention_forward
    modeling_qwen3.Qwen3ForCausalLM.forward = old_qwen3_for_causal_lm_forward
    
    print("✓ Qwen3 Tuple KV Cache enabled successfully")