import torch


def count_parameters(model: torch.nn.Module) -> dict[str, int]:
    """
    Returns the parameter counts (total and trainable).
    """
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total_params": total, "trainable_params": trainable}


def estimate_flops(
    model: torch.nn.Module, 
    seq_len: int, 
    batch_size: int = 1
) -> dict[str, float]:
    """
    Estimates the number of FLOPs (floating point operations) for a single forward pass.
    Assumes standard transformer architecture (Nebium).
    """
    if not hasattr(model, "d_model") or not hasattr(model, "n_layers"):
        # We need the config or model dimensions. Try to extract them.
        try:
            d_model = model.token_embed.embedding.embedding_dim
            vocab_size = model.token_embed.embedding.num_embeddings
            n_layers = len(model.blocks)
            n_heads = model.blocks[0].attn.n_heads
        except AttributeError:
            return {"estimated_flops_forward": 0.0}
    else:
        d_model = model.d_model
        n_layers = model.n_layers
        vocab_size = model.vocab_size

    # 1. Attention: 
    # Q,K,V proj: 3 * (batch * seq_len * d_model * d_model) * 2
    # QK^T: 2 * batch * n_heads * seq_len * seq_len * (d_model/n_heads) = 2 * batch * seq_len * seq_len * d_model
    # Attention * V: 2 * batch * seq_len * seq_len * d_model
    # Out proj: 2 * batch * seq_len * d_model * d_model
    attn_flops = (
        batch_size * seq_len * d_model * d_model * 6 +
        batch_size * seq_len * seq_len * d_model * 4 +
        batch_size * seq_len * d_model * d_model * 2
    )

    # 2. FFN:
    # Usually FFN hidden is 4 * d_model or similar. SwiGLU is (8/3) * d_model.
    # We will estimate it generally as 8 * d_model * d_model operations.
    ffn_flops = batch_size * seq_len * d_model * (8 * d_model) * 2

    # 3. LM Head:
    lm_head_flops = batch_size * seq_len * d_model * vocab_size * 2
    
    total_layer_flops = (attn_flops + ffn_flops) * n_layers
    total_forward_flops = total_layer_flops + lm_head_flops
    
    return {
        "estimated_flops_forward": total_forward_flops,
        "estimated_flops_forward_per_token": total_forward_flops / (seq_len * batch_size)
    }
