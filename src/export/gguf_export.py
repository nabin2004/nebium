import os
import re
from pathlib import Path
from typing import Any

import numpy as np
import torch


def _map_tensor_name(py_name: str) -> str:
    """Maps PyTorch module parameter names to standard GGUF tensor names."""
    if py_name in ("token_embed.embedding.weight", "token_embed.weight"):
        return "token_embd.weight"
    if py_name in ("learned_pos.embedding.weight", "learned_pos.weight"):
        return "position_embd.weight"
    if py_name in ("final_norm.weight", "norm.weight"):
        return "output_norm.weight"
    if py_name in ("final_norm.bias", "norm.bias"):
        return "output_norm.bias"
    if py_name == "lm_head.weight":
        return "output.weight"
    if py_name == "lm_head.bias":
        return "output.bias"

    # Transformer block match: blocks.0.xxx
    block_match = re.match(r"^blocks\.(\d+)\.(.+)$", py_name)
    if block_match:
        idx = block_match.group(1)
        sub = block_match.group(2)
        mapping = {
            "attn_norm.weight": f"blk.{idx}.attn_norm.weight",
            "attn_norm.bias": f"blk.{idx}.attn_norm.bias",
            "attn.q_proj.weight": f"blk.{idx}.attn_q.weight",
            "attn.q_proj.bias": f"blk.{idx}.attn_q.bias",
            "attn.k_proj.weight": f"blk.{idx}.attn_k.weight",
            "attn.k_proj.bias": f"blk.{idx}.attn_k.bias",
            "attn.v_proj.weight": f"blk.{idx}.attn_v.weight",
            "attn.v_proj.bias": f"blk.{idx}.attn_v.bias",
            "attn.out_proj.weight": f"blk.{idx}.attn_output.weight",
            "attn.out_proj.bias": f"blk.{idx}.attn_output.bias",
            "ffn_norm.weight": f"blk.{idx}.ffn_norm.weight",
            "ffn_norm.bias": f"blk.{idx}.ffn_norm.bias",
            "ffn.w_gate.weight": f"blk.{idx}.ffn_gate.weight",
            "ffn.w_up.weight": f"blk.{idx}.ffn_up.weight",
            "ffn.w_down.weight": f"blk.{idx}.ffn_down.weight",
            "ffn.w_in.weight": f"blk.{idx}.ffn_up.weight",
            "ffn.w_out.weight": f"blk.{idx}.ffn_down.weight",
        }
        if sub in mapping:
            return mapping[sub]
        return f"blk.{idx}.{sub}"

    return py_name


def _extract_tokens_and_scores(tokenizer: Any) -> tuple[list[str], list[float]]:
    """Extracts vocabulary tokens in index order and dummy scores for GGUF metadata."""
    if hasattr(tokenizer, "tokenizer") and hasattr(tokenizer.tokenizer, "get_vocab"):
        vocab_dict = tokenizer.tokenizer.get_vocab()
    elif hasattr(tokenizer, "get_vocab"):
        vocab_dict = tokenizer.get_vocab()
    else:
        # Fallback dummy list
        vocab_size = getattr(tokenizer, "vocab_size", 5000)
        return [f"<token_{i}>" for i in range(vocab_size)], [0.0] * vocab_size

    sorted_pairs = sorted(vocab_dict.items(), key=lambda x: x[1])
    tokens = [tok for tok, _ in sorted_pairs]
    scores = [0.0] * len(tokens)
    return tokens, scores


def export_to_gguf(
    model: torch.nn.Module,
    tokenizer: Any,
    output_path: str | Path,
    precision: str = "fp16",
    architecture: str = "nebium",
) -> Path:
    """
    Exports a trained Nebium PyTorch model to GGUF format for local deployment.
    Uses the official gguf library (gguf.GGUFWriter).
    """
    try:
        import gguf
    except ImportError as exc:
        raise ImportError("gguf package is required for GGUF export. Install with: pip install gguf") from exc

    model = model.module if hasattr(model, "module") else model

    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    # Determine model parameters
    d_model = getattr(model, "d_model", None)
    if d_model is None and hasattr(model, "token_embed") and hasattr(model.token_embed, "embedding"):
        d_model = model.token_embed.embedding.embedding_dim
    d_model = d_model or 512

    max_seq_len = getattr(model, "max_seq_len", 512)
    n_layers = len(getattr(model, "blocks", [])) or 1

    if hasattr(model, "blocks") and len(model.blocks) > 0 and hasattr(model.blocks[0], "attn"):
        first_attn = model.blocks[0].attn
        n_heads = getattr(first_attn, "n_heads", 8)
        head_dim = getattr(first_attn, "head_dim", d_model // n_heads)
    else:
        n_heads = 8
        head_dim = d_model // n_heads

    tokens, scores = _extract_tokens_and_scores(tokenizer)
    vocab_size = len(tokens) if tokens else getattr(tokenizer, "vocab_size", 5000)

    # Instantiate GGUFWriter
    writer = gguf.GGUFWriter(str(out_file), architecture)

    # General metadata
    writer.add_name("Nebium-Chess-Transformer")
    writer.add_description("Causal Transformer for Chess Next-Move Prediction exported to GGUF")
    writer.add_uint32("general.file_type", 1 if precision == "fp16" else 0)

    # Architecture metadata
    writer.add_context_length(int(max_seq_len))
    writer.add_embedding_length(int(d_model))
    writer.add_block_count(int(n_layers))
    writer.add_head_count(int(n_heads))
    writer.add_head_count_kv(int(n_heads))
    writer.add_rope_dimension_count(int(head_dim))
    writer.add_layer_norm_rms_eps(1e-5)

    # Tokenizer metadata
    writer.add_tokenizer_model("gpt2")
    writer.add_token_list(tokens)
    writer.add_token_scores(scores)
    if hasattr(tokenizer, "bos_id") and tokenizer.bos_id is not None:
        writer.add_bos_token_id(int(tokenizer.bos_id))
    if hasattr(tokenizer, "eos_id") and tokenizer.eos_id is not None:
        writer.add_eos_token_id(int(tokenizer.eos_id))
    if hasattr(tokenizer, "pad_id") and tokenizer.pad_id is not None:
        writer.add_pad_token_id(int(tokenizer.pad_id))
    if hasattr(tokenizer, "sep_id") and tokenizer.sep_id is not None:
        writer.add_sep_token_id(int(tokenizer.sep_id))

    # Convert state_dict to GGUF tensors
    np_dtype = np.float16 if precision == "fp16" else np.float32
    state_dict = model.state_dict()

    for py_name, tensor in state_dict.items():
        gguf_name = _map_tensor_name(py_name)
        # Convert PyTorch tensor to numpy array in target precision
        data = tensor.detach().cpu().to(torch.float32).numpy().astype(np_dtype)
        writer.add_tensor(gguf_name, data)

    # Finalize and write GGUF file
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()

    print(f"[GGUF Export] Successfully exported model to {out_file} (Precision: {precision.upper()}, Size: {out_file.stat().st_size / (1024*1024):.2f} MB)")
    return out_file
