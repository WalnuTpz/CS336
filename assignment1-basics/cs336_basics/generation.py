import torch
from torch import Tensor
from cs336_basics.attention import softmax

def nucleus_filter(     # 核采样过滤
    probs: Tensor,  # (B, V)
    top_p: float
) -> Tensor:
    if top_p >= 1.0:    # top_p >= 1，保持不变
        return probs
    if top_p <= 0.0:    # top_p <= 0，只保留最大项
        idx = probs.argmax(dim=-1, keepdim=True)    # (B, 1)
        # 把最大项对应的位置变为 1，其余位置全部为0
        out = torch.zeros_like(probs)
        out = out.scatter(dim=-1, index=idx, src=torch.ones_like(idx, dtype=probs.dtype))

        return out

    sorted_probs, sorted_idx = torch.sort(probs, dim=-1, descending=True)    # 降序排序后的 probs，每个新位置对应的原位置
    cum = sorted_probs.cumsum(dim=-1)    # 每个位置的累计概率
    cum_prev = cum - sorted_probs

    keep_ids = cum_prev < top_p    # 前一个位置的累计概率 < 给定概率和的时候，当前位置才会被保留
    keep_ids[..., 0] = True    # 强制保留每个序列的首位
    sorted_probs = sorted_probs * keep_ids    # 将没被保留的位置（False）的概率清零

    # 归一化
    sum_probs = sorted_probs.sum(dim=-1, keepdim=True).clamp_min(1e-12)
    sorted_probs = sorted_probs / sum_probs

    # 将归一化后的概率填回原位置
    out = torch.zeros_like(probs)
    out = out.scatter(dim=-1, index=sorted_idx, src=sorted_probs)

    return out


@torch.no_grad()
def generate(    # 产生新的 tokens
    model,
    prompt_ids: Tensor,  # (B, T) long
    max_new_tokens: int,
    *,
    temperature: float = 1.0,
    top_p: float = 1.0,
    eos_token_id: int | None = None,
) -> Tensor:  # (B, T + <=max_new_tokens)
    model.eval()
    batch = prompt_ids.shape[0]
    max_len = model.context_length
    out = prompt_ids    # 输出结果
    finished = torch.zeros(batch, device=prompt_ids.device, dtype=torch.bool)    # 标记已生成结束的序列
    eos_tensor = torch.full((batch, 1), eos_token_id, device=prompt_ids.device, dtype=prompt_ids.dtype)

    for _ in range(max_new_tokens):
        context = out[:, -max_len :]    # 保留 out 的最后至多 max_len 个元素
        logits = model(context)    # (B, T, V)，生成新的结果
        next_logits = logits[:, -1, :]    # (B, V)，将每个序列的最后一个元素作为新的 logits
        if temperature > 0:
            next_logits = next_logits / temperature    # 进行温度缩放

        # 进行 softmax 和核采样过滤
        probs = softmax(next_logits, dim=-1)
        probs = nucleus_filter(probs, top_p)

        next_id = torch.multinomial(probs, num_samples=1)    # (B, 1)，随机抽取得到下一个 token
        if eos_token_id is not None:
            next_id = torch.where(finished[:, None], eos_tensor, next_id)    # 将已经生成结束的序列的下一个 token 强制变为 eos
        out = torch.cat([out, next_id], dim=1)    # 将每个序列的下一个 token 拼接到输出结果尾部

        if eos_token_id is not None:
            finished_cur = next_id.squeeze(1) == eos_token_id  # (B, 1)，在这一步中有哪些序列生成结束了
            finished |= finished_cur  # 将生成 eos 的序列编号更新到 finished
            if finished.all():  # 所有序列都生成结束以后，直接退出
                break

    return out
