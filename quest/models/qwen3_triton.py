"""Qwen3 Quest model using Triton attention backend for prefill/decode."""
import torch
import torch.nn.functional as F
from torch import nn

import quest.utils
from quest.utils import triton_backend
from quest.models import qwen3 as qwen3_base


class QuestQwen3AttentionTriton(qwen3_base.QuestQwen3Attention):
    """Quest attention for Qwen3 using Triton kernels for prefill/decode."""

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask=None,
        position_ids=None,
        past_key_value=None,
        output_attentions: bool = False,
        use_cache: bool = False,
        iController: quest.utils.InferenceController = None,
    ):
        bsz, q_len, _ = hidden_states.size()

        assert bsz == 1, "QuestAttention only supports batch size 1."
        assert hasattr(self, "layer_idx"), "QuestAttention requires layer_idx to inference."

        if self.pretraining_tp > 1:
            key_value_slicing = (self.num_key_value_heads * self.head_dim) // self.pretraining_tp
            query_slices = self.q_proj.weight.split((self.num_heads * self.head_dim) // self.pretraining_tp, dim=0)
            key_slices = self.k_proj.weight.split(key_value_slicing, dim=0)
            value_slices = self.v_proj.weight.split(key_value_slicing, dim=0)

            query_states = [F.linear(hidden_states, query_slices[i]) for i in range(self.pretraining_tp)]
            query_states = torch.cat(query_states, dim=-1)

            key_states = [F.linear(hidden_states, key_slices[i]) for i in range(self.pretraining_tp)]
            key_states = torch.cat(key_states, dim=-1)

            value_states = [F.linear(hidden_states, value_slices[i]) for i in range(self.pretraining_tp)]
            value_states = torch.cat(value_states, dim=-1)
        else:
            torch.cuda.nvtx.range_push("qkv_proj")
            query_states = self.q_proj(hidden_states)
            key_states = self.k_proj(hidden_states)
            value_states = self.v_proj(hidden_states)
            torch.cuda.nvtx.range_pop()

        query_states = query_states.view(q_len, self.num_heads, self.head_dim)
        key_states = key_states.view(q_len, self.num_key_value_heads, self.head_dim)
        value_states = value_states.view(q_len, self.num_key_value_heads, self.head_dim)

        query_states = self._rms_norm_fp32(query_states, self.q_norm.weight, self.q_norm.variance_epsilon)
        key_states = self._rms_norm_fp32(key_states, self.k_norm.weight, self.k_norm.variance_epsilon)

        key_states = self._repeat_kv(key_states)

        torch.cuda.nvtx.range_push("RoPE")
        quest.utils.apply_rope_in_place(
            query_states,
            key_states,
            iController.kv_cache.seqlen - q_len,
            rope_scale=self.rope_scale,
            rope_theta=self.rope_theta,
        )
        torch.cuda.nvtx.range_pop()

        value_states = self._repeat_kv(value_states)

        torch.cuda.nvtx.range_push("append_kv")
        quest.utils.append_kv(
            key_states,
            value_states,
            iController,
            self.layer_idx,
        )
        torch.cuda.nvtx.range_pop()

        if q_len > 1:
            torch.cuda.nvtx.range_push("prefill_attn_triton")
            attn_output = triton_backend.triton_prefill_forward(
                query_states,
                iController,
                self.layer_idx,
            )
            torch.cuda.nvtx.range_pop()
        else:
            if iController.need_estimate() is False:
                torch.cuda.nvtx.range_push("full_attn_triton")
                attn_output = triton_backend.triton_decode_sparse_attn(
                    query_states,
                    iController,
                    self.layer_idx,
                    iController.kv_indices_without_last,
                )
                torch.cuda.nvtx.range_pop()
            else:
                torch.cuda.nvtx.range_push("estimate_triton")
                estimated_attn_score = triton_backend.triton_decode_estimate(
                    query_states,
                    iController,
                    self.layer_idx,
                )
                torch.cuda.nvtx.range_pop()

                torch.cuda.nvtx.range_push("topk_triton")
                triton_backend.triton_decode_topk(
                    estimated_attn_score,
                    iController,
                )
                torch.cuda.nvtx.range_pop()

                torch.cuda.nvtx.range_push("approx_attn_triton")
                attn_output = triton_backend.triton_decode_sparse_attn(
                    query_states,
                    iController,
                    self.layer_idx,
                    iController.topk_dindices_buffer,
                )
                torch.cuda.nvtx.range_pop()

        attn_output = attn_output.unsqueeze(0)
        if attn_output.size() != (bsz, q_len, self.num_heads, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

        torch.cuda.nvtx.range_push("o_proj")
        if self.pretraining_tp > 1:
            attn_output = attn_output.split(self.hidden_size // self.pretraining_tp, dim=2)
            o_proj_slices = self.o_proj.weight.split(self.hidden_size // self.pretraining_tp, dim=1)
            attn_output = sum([F.linear(attn_output[i], o_proj_slices[i]) for i in range(self.pretraining_tp)])
        else:
            attn_output = self.o_proj(attn_output)
        torch.cuda.nvtx.range_pop()

        if not output_attentions:
            attn_weights = None

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value


class Qwen3DecoderLayerTriton(nn.Module):
    def __init__(self, config: qwen3_base.Qwen3Config, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = QuestQwen3AttentionTriton(config=config, layer_idx=layer_idx)
        self.mlp = qwen3_base.Qwen3MLP(config)
        self.input_layernorm = qwen3_base.Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = qwen3_base.Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask=None,
        position_ids=None,
        past_key_value=None,
        output_attentions: bool = False,
        use_cache: bool = False,
        iController: quest.utils.InferenceController = None,
    ):
        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            iController=iController,
        )
        hidden_states = residual + hidden_states

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


class Qwen3ModelTriton(qwen3_base.Qwen3Model):
    def __init__(self, config: qwen3_base.Qwen3Config):
        qwen3_base.Qwen3PreTrainedModel.__init__(self, config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList([Qwen3DecoderLayerTriton(config, i) for i in range(config.num_hidden_layers)])
        self.norm = qwen3_base.Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        self.gradient_checkpointing = False

        self.iController = None

        self.post_init()


class Qwen3ForCausalLMTriton(qwen3_base.Qwen3ForCausalLM):
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config):
        qwen3_base.Qwen3PreTrainedModel.__init__(self, config)
        self.model = Qwen3ModelTriton(config)
        self.pretraining_tp = getattr(config, "pretraining_tp", 1)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self._config = config
        self.post_init()
