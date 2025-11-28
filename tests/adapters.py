from __future__ import annotations
from errno import EBADARCH
import os
from collections.abc import Iterable
from typing import IO, Any, BinaryIO

from numpy import min_scalar_type
import numpy.typing as npt
import torch
from jaxtyping import Bool, Float, Int
from torch import Tensor, rms_norm

from cs336_basics.train_bpe import train_bpe
from cs336_basics.tokenizer import Tokenizer

from cs336_basics.linear_module import Linear
from cs336_basics.embedding import Embedding
from cs336_basics.RMSNorm import RMSNorm

from cs336_basics.SwiGLU_FFN import SwiGLU_FFN
from cs336_basics.Rope import Rope
from cs336_basics.scaled_dot_product_attention import softmax,scaled_dot_product_attention 
from cs336_basics.CausalMultiheadSelfAttention import CausalMultiheadSelfAttention


from cs336_basics.Transformer_block import TransformerBlock
from cs336_basics.TransformerLM import TransformerLM 

from cs336_basics.Cross_entropy import Cross_entropy
from cs336_basics.adamW import AdamW

from cs336_basics.lr_gc import lr_cosine_schedule,gradient_clipping
from cs336_basics.get_batch import get_batch
from cs336_basics.checkpointing import save_checkpoint,load_checkpoint

def run_linear(
    d_in: int,
    d_out: int,
    weights: Float[Tensor, " d_out d_in"],
    in_features: Float[Tensor, " ... d_in"],
) -> Float[Tensor, " ... d_out"]:

    model = Linear(d_in, d_out)
    state = model.state_dict()
    state["W"] = weights          
    model.load_state_dict(state)
    return model(in_features)

def run_embedding(
    vocab_size: int,
    d_model: int,
    weights: Float[Tensor, " vocab_size d_model"],
    token_ids: Int[Tensor, " ..."],
) -> Float[Tensor, " ... d_model"]:

    model = Embedding(vocab_size,d_model)
    state = model.state_dict()
    state["embeddings"] = weights
    model.load_state_dict(state)
    return model(token_ids)

def run_swiglu(
    d_model: int,
    d_ff: int,
    w1_weight: Float[Tensor, " d_ff d_model"],
    w2_weight: Float[Tensor, " d_model d_ff"],
    w3_weight: Float[Tensor, " d_ff d_model"],
    in_features: Float[Tensor, " ... d_model"],
) -> Float[Tensor, " ... d_model"]:

    device = in_features.device
    dtype = in_features.dtype
    swiglu = SwiGLU_FFN(d_model=d_model, d_ff=d_ff, device=device, dtype=dtype)
    state = swiglu.state_dict()
    state["W1.W"] = w1_weight.to(device=device, dtype=dtype)
    state["W2.W"] = w2_weight.to(device=device, dtype=dtype)
    state["W3.W"] = w3_weight.to(device=device, dtype=dtype)
    swiglu.load_state_dict(state)
    return swiglu(in_features)



def run_scaled_dot_product_attention(
    Q: Float[Tensor, " ... queries d_k"],
    K: Float[Tensor, " ... keys d_k"],
    V: Float[Tensor, " ... values d_v"],
    mask: Bool[Tensor, " ... queries keys"] | None = None,
) -> Float[Tensor, " ... queries d_v"]:
    return scaled_dot_product_attention(Q,K,V,mask)


def run_multihead_self_attention(
    d_model: int,
    num_heads: int,
    q_proj_weight: Float[Tensor, " d_k d_in"],
    k_proj_weight: Float[Tensor, " d_k d_in"],
    v_proj_weight: Float[Tensor, " d_v d_in"],
    o_proj_weight: Float[Tensor, " d_model d_v"],
    in_features: Float[Tensor, " ... sequence_length d_in"],
) -> Float[Tensor, " ... sequence_length d_out"]:

    attn = CausalMultiheadSelfAttention(
        d_model=d_model,
        num_heads=num_heads,
        max_seq_len=None,
        device=in_features.device,
        dtype=in_features.dtype,
    )

    attn.W_q.load_state_dict({"W": q_proj_weight})
    attn.W_k.load_state_dict({"W": k_proj_weight})
    attn.W_v.load_state_dict({"W": v_proj_weight})
    attn.W_o.load_state_dict({"W": o_proj_weight})

    return attn(in_features)

def run_multihead_self_attention_with_rope(
    d_model: int,
    num_heads: int,
    max_seq_len: int,
    theta: float,
    q_proj_weight: Float[Tensor, " d_k d_in"],
    k_proj_weight: Float[Tensor, " d_k d_in"],
    v_proj_weight: Float[Tensor, " d_v d_in"],
    o_proj_weight: Float[Tensor, " d_model d_v"],
    in_features: Float[Tensor, " ... sequence_length d_in"],
    token_positions: Int[Tensor, " ... sequence_length"] | None = None,
) -> Float[Tensor, " ... sequence_length d_out"]:
    attn = CausalMultiheadSelfAttention(
        d_model=d_model,
        num_heads=num_heads,
        max_seq_len=max_seq_len,
        theta=theta,
        device=in_features.device,
        dtype=in_features.dtype,
    )

    attn.W_q.load_state_dict({"W": q_proj_weight})
    attn.W_k.load_state_dict({"W": k_proj_weight})
    attn.W_v.load_state_dict({"W": v_proj_weight})
    attn.W_o.load_state_dict({"W": o_proj_weight})

    return attn(in_features, token_positions)


def run_rope(
    d_k: int,
    theta: float,
    max_seq_len: int,
    in_query_or_key: Float[Tensor, " ... sequence_length d_k"],
    token_positions: Int[Tensor, " ... sequence_length"],
) -> Float[Tensor, " ... sequence_length d_k"]:

    model = Rope(theta,d_k,max_seq_len)
    return model(in_query_or_key,token_positions)


def run_transformer_block(
    d_model: int,
    num_heads: int,
    d_ff: int,
    max_seq_len: int,
    theta: float,
    weights: dict[str, Tensor],
    in_features: Float[Tensor, " batch sequence_length d_model"],
) -> Float[Tensor, " batch sequence_length d_model"]:

    model = TransformerBlock(
        d_model=d_model,
        num_heads=num_heads,
        d_ff=d_ff,
        max_seq_len=max_seq_len,
        theta=theta,
        device=in_features.device,
    )
    state = {
        "norm1.g":        weights["ln1.weight"],
        "norm2.g":        weights["ln2.weight"],
        "attn.W_q.W":     weights["attn.q_proj.weight"],
        "attn.W_k.W":     weights["attn.k_proj.weight"],
        "attn.W_v.W":     weights["attn.v_proj.weight"],
        "attn.W_o.W":     weights["attn.output_proj.weight"],
        "ffn.W1.W":       weights["ffn.w1.weight"],
        "ffn.W2.W":       weights["ffn.w2.weight"],
        "ffn.W3.W":       weights["ffn.w3.weight"],
    }

    model.load_state_dict(state, strict=False)
    return model(in_features)

def run_transformer_lm(
    vocab_size: int,
    context_length: int,
    d_model: int,
    num_layers: int,
    num_heads: int,
    d_ff: int,
    rope_theta: float,
    weights: dict[str, Tensor],
    in_indices: Int[Tensor, " batch_size sequence_length"],
) -> Float[Tensor, " batch_size sequence_length vocab_size"]:
    device = in_indices.device

    model = TransformerLM(
        vocab_size=vocab_size,
        d_model=d_model,
        num_heads=num_heads,
        d_ff=d_ff,
        num_layers=num_layers,
        max_seq_len=context_length,  # RoPE 的 max_seq_len 就用 context_length
        theta=rope_theta,
        device=device,
        dtype=None,
    )

    # 重新对齐 key
    new_state = {}

    # token embedding
    new_state["token_embedding.embeddings"] = weights["token_embeddings.weight"]

    # per-layer weights
    for i in range(num_layers):
        prefix_old = f"layers.{i}"
        prefix_new = f"blocks.{i}"

        # norm1 / norm2
        new_state[f"{prefix_new}.norm1.g"] = weights[f"{prefix_old}.ln1.weight"]
        new_state[f"{prefix_new}.norm2.g"] = weights[f"{prefix_old}.ln2.weight"]

        # attention qkv，output
        new_state[f"{prefix_new}.attn.W_q.W"] = weights[f"{prefix_old}.attn.q_proj.weight"]
        new_state[f"{prefix_new}.attn.W_k.W"] = weights[f"{prefix_old}.attn.k_proj.weight"]
        new_state[f"{prefix_new}.attn.W_v.W"] = weights[f"{prefix_old}.attn.v_proj.weight"]
        new_state[f"{prefix_new}.attn.W_o.W"] = weights[f"{prefix_old}.attn.output_proj.weight"]

        # FFN W1/W2/W3
        new_state[f"{prefix_new}.ffn.W1.W"] = weights[f"{prefix_old}.ffn.w1.weight"]
        new_state[f"{prefix_new}.ffn.W2.W"] = weights[f"{prefix_old}.ffn.w2.weight"]
        new_state[f"{prefix_new}.ffn.W3.W"] = weights[f"{prefix_old}.ffn.w3.weight"]

    # final RMSNorm
    new_state["norm_f.g"] = weights["ln_final.weight"]

    # lm head
    new_state["lm_head.W"] = weights["lm_head.weight"]

    # load
    model.load_state_dict(new_state)

    # forward
    out = model(in_indices)
    return out


def run_rmsnorm(
    d_model: int,
    eps: float,
    weights: Float[Tensor, " d_model"],
    in_features: Float[Tensor, " ... d_model"],
) -> Float[Tensor, " ... d_model"]:

    model = RMSNorm(d_model,eps)
    state = model.state_dict()
    state["g"] = weights
    model.load_state_dict(state)
    return model(in_features)


def run_silu(in_features: Float[Tensor, " ..."]) -> Float[Tensor, " ..."]:

    return in_features * torch.sigmoid(in_features)

def run_get_batch(
    dataset: npt.NDArray, batch_size: int, context_length: int, device: str
) -> tuple[torch.Tensor, torch.Tensor]:
    return get_batch(dataset,batch_size, context_length, device)


def run_softmax(in_features: Float[Tensor, " ..."], dim: int) -> Float[Tensor, " ..."]:
    return softmax(in_features,dim)


def run_cross_entropy(
    inputs: Float[Tensor, " batch_size vocab_size"], targets: Int[Tensor, " batch_size"]
) -> Float[Tensor, ""]:

    return Cross_entropy(inputs,targets)

def run_gradient_clipping(parameters: Iterable[torch.nn.Parameter], max_l2_norm: float) -> None:

    return gradient_clipping(parameters,max_l2_norm)

def get_adamw_cls() -> Any:
    """
    Returns a torch.optim.Optimizer that implements AdamW.
    """
    return AdamW

def run_get_lr_cosine_schedule(
    it: int,
    max_learning_rate: float,
    min_learning_rate: float,
    warmup_iters: int,
    cosine_cycle_iters: int,
):
    return lr_cosine_schedule(it,max_learning_rate,min_learning_rate,warmup_iters,cosine_cycle_iters)


def run_save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    iteration: int,
    out: str | os.PathLike | BinaryIO | IO[bytes],
):
    save_checkpoint(model, optimizer, iteration, out,)

def run_load_checkpoint(
    src: str | os.PathLike | BinaryIO | IO[bytes],
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
) -> int:
    return load_checkpoint(src, model, optimizer)


def get_tokenizer(
    vocab: dict[int, bytes],
    merges: list[tuple[bytes, bytes]],
    special_tokens: list[str] | None = None,
) -> Any:

    return Tokenizer(vocab, merges, special_tokens)

def run_train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
    **kwargs,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:

    return train_bpe(input_path, vocab_size, special_tokens)
