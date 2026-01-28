import torch
from torch import Tensor

def cross_entropy(    # 计算交叉熵（这里没有直接使用原公式，而是使用化简后的公式）
    logits: Tensor,  # (..., V)
    targets: Tensor  # (...)
) -> Tensor:
    V = logits.shape[-1]

    # 将 logits, targets 展平，也就是把前面的所有维度合并为一维
    logits_2d = logits.reshape(-1, V).float()
    targets_1d = targets.reshape(-1).long()

    log_denom = torch.logsumexp(logits_2d, dim=-1)
    # 先取指数，再求和，最后取对数（这个函数不会数值溢出，所以不用将每个元素减去对应的最大元素以后再取指数）
    logit_y = logits_2d.gather(dim=-1, index=targets_1d[:, None]).squeeze(-1)
    # 根据 targets 取出 logits_2d 中对应的值，然后去掉最后一维
    out = (log_denom - logit_y).mean()  # (

    return out

def perplexity(losses: Tensor) -> Tensor:    # 计算困惑度
    return torch.exp(losses.mean())