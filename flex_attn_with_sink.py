from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.attention.flex_attention import flex_attention

flex_attention = torch.compile(flex_attention, dynamic=True)


# copy from https://github.com/huggingface/transformers/blob/main/src/transformers/models/gpt_oss/modeling_gpt_oss.py#L210-L219
def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


# copy and modify from https://github.com/huggingface/transformers/blob/main/src/transformers/models/gpt_oss/modeling_gpt_oss.py#L241-L270
def eager_attention_forward(
        query: torch.Tensor,  # [b, n, sq, d]
        key: torch.Tensor,  # [b, n, skv, d]
        value: torch.Tensor,  # [b, n, skv, d]
        sinks: torch.Tensor,  # [n]
        attention_mask: Optional[torch.Tensor],  # [b, n, sq, skv]
        scaling: Optional[float] = None,
        num_key_value_groups: int = 8,
        dropout: float = 0.0,
        training: bool = False,
        **kwargs,
):
    b, n, sq, d = query.shape
    if scaling is None:
        scaling = d ** -0.5
    key_states = repeat_kv(key, num_key_value_groups)
    value_states = repeat_kv(value, num_key_value_groups)
    attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling
    if attention_mask is not None:
        causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
        attn_weights = attn_weights + causal_mask

    sinks = sinks.reshape(1, -1, 1, 1).expand(b, -1, sq, -1)
    combined_logits = torch.cat([attn_weights, sinks], dim=-1)

    # This was not in the original implementation and slightly affect results; it prevents overflow in BF16/FP16
    # when training with bsz>1 we clamp max values.

    combined_logits = combined_logits - combined_logits.amax(dim=-1, keepdim=True)
    probs = F.softmax(combined_logits, dim=-1, dtype=combined_logits.dtype)
    scores = probs[..., :-1]  # we drop the sink here
    attn_weights = nn.functional.dropout(scores, p=dropout, training=training)
    attn_output = torch.matmul(attn_weights, value_states)
    attn_output = attn_output.transpose(1, 2).contiguous()
    return attn_output, attn_weights


def flex_attn_with_sink_func(
        query: torch.Tensor,  # [b, n, sq, d]
        key: torch.Tensor,  # [b, n, skv, d]
        value: torch.Tensor,  # [b, n, skv, d]
        sinks: torch.Tensor,  # [n]
        attention_mask: Optional[torch.Tensor] = None,  # [b, n, sq, skv]
        scaling: Optional[float] = None,
        num_key_value_groups: int = 8,
        is_causal: bool = False,
        **kwargs,
):
    assert (dropout := kwargs.get('dropout', 0.0) == 0.0), \
        f'{dropout=} is not supported!'
    assert attention_mask is None or not is_causal, \
        'attention_mask is not None, so is_causal must be False!'
    b, n, sq, d = query.shape

    key = repeat_kv(key, num_key_value_groups)
    value = repeat_kv(value, num_key_value_groups)

    sk = key.size(2)
    sv = value.size(2)
    assert sk == sv, f'{sk=} is not equal to {sv=}!'

    if scaling is None:
        scaling = d ** -0.5

    dummy_key = F.pad(
        key,
        (0, 0, 0, 1, 0, 0, 0, 0),
        mode='constant',
        value=0.0,
    )
    dummy_value = F.pad(
        value,
        (0, 0, 0, 1, 0, 0, 0, 0),
        value=0.0,
    )

    if is_causal:
        def score_mod(
                score: torch.Tensor,
                b_idx: torch.Tensor,
                n_idx: torch.Tensor,
                q_idx: torch.Tensor,
                kv_idx: torch.Tensor,
        ):
            score = torch.where(
                kv_idx == sk,
                sinks[n_idx],
                torch.where(
                    q_idx >= kv_idx,
                    score,
                    torch.finfo(score.dtype).min,
                )
            )
            return score
    else:
        def score_mod(
                score: torch.Tensor,
                b_idx: torch.Tensor,
                n_idx: torch.Tensor,
                q_idx: torch.Tensor,
                kv_idx: torch.Tensor,
        ) -> torch.Tensor:
            score = torch.where(kv_idx == sk, sinks[n_idx], score)
            return score

    attn_output = flex_attention(
        query,
        dummy_key,
        dummy_value,
        score_mod=score_mod,
        block_mask=None,
        scale=scaling,
        enable_gqa=False,
        return_lse=False,
        kernel_options=None,
    )
    return attn_output


@torch.inference_mode()
def test():
    dtype = torch.bfloat16
    device = torch.device('cuda:0')

    num_key_value_groups = 8
    b, n, sq, skv, d = 1, 40, 4096, 4096, 128
    query = torch.randn((b, n, sq, d), device=device, dtype=dtype)
    key = torch.randn((b, n // num_key_value_groups, skv, d), device=device, dtype=dtype)
    value = torch.randn((b, n // num_key_value_groups, skv, d), device=device, dtype=dtype)
    sinks = torch.randn((n,), device=device, dtype=dtype)

    attention_causal_mask = torch.triu(
        torch.full((sq, skv), torch.finfo(dtype).min), diagonal=1,
    )[None, None].to(device=device, dtype=dtype)

    def run_full_attention():
        gt, _ = eager_attention_forward(
            query=query,
            key=key,
            value=value,
            sinks=sinks,
            attention_mask=None,
            scaling=None,
            num_key_value_groups=num_key_value_groups,
            dropout=0.0,
            training=False,
        )

        pred = flex_attn_with_sink_func(
            query=query,
            key=key,
            value=value,
            sinks=sinks,
            attention_mask=None,
            scaling=None,
            num_key_value_groups=num_key_value_groups,
            dropout=0.0,
            is_causal=False,
            training=False,
        ).permute(0, 2, 1, 3)

        return gt, pred

    def run_casual_attention():
        gt, _ = eager_attention_forward(
            query=query,
            key=key,
            value=value,
            sinks=sinks,
            attention_mask=attention_causal_mask,
            scaling=None,
            num_key_value_groups=num_key_value_groups,
            dropout=0.0,
            training=False,
        )

        pred = flex_attn_with_sink_func(
            query=query,
            key=key,
            value=value,
            sinks=sinks,
            attention_mask=None,
            scaling=None,
            num_key_value_groups=num_key_value_groups,
            dropout=0.0,
            is_causal=True,
            training=False,
        ).permute(0, 2, 1, 3)

        return gt, pred

    for fn in [run_full_attention, run_casual_attention]:
        gt, pred = fn()
        diff = gt - pred

        print('*' * 100)
        if not torch.allclose(gt, pred, rtol=1e-3, atol=1e-2):
            print(f'{fn.__name__} is not allclose!')
            print(f'mean: {diff.mean().item():.7f}, max: {diff.max().item():.7f}, min: {diff.min().item():.7f}')
        else:
            print(f'{fn.__name__} is allclose!')


if __name__ == '__main__':
    test()
