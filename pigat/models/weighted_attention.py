import torch


def signed_weighted_attention(scores, src, dst, n, edge_weight, eps=1e-12):
    """
    scores: [E, H]
    edge_weight: [E]
    Returns signed attention coefficients [E, H].
    """
    w = edge_weight.to(device=scores.device, dtype=scores.dtype)

    if w.numel() != scores.shape[0]:
        raise ValueError("Edge weight count does not match edge count")

    # Weight magnitude enters the attention logits.
    logits = scores + torch.log(w.abs().clamp_min(eps)).unsqueeze(-1)

    # Destination-wise numerically stable softmax.
    max_per_dst = torch.full(
        (n, scores.shape[1]),
        -torch.inf,
        device=scores.device,
        dtype=scores.dtype
    )

    max_per_dst.scatter_reduce_(
        0,
        dst[:, None].expand_as(logits),
        logits,
        reduce="amax",
        include_self=True
    )

    shifted = logits - max_per_dst[dst]
    exp_logits = torch.exp(shifted)

    denom = torch.zeros(
        n, scores.shape[1],
        device=scores.device,
        dtype=scores.dtype
    )

    denom.index_add_(0, dst, exp_logits)

    attention = exp_logits / denom[dst].clamp_min(eps)

    # Signed message contribution.
    return attention * w.sign().unsqueeze(-1)
