import torch

def collect_conditional_examples(
    loader,
    device,
    conditional_tokens,
    num_samples=500,
    max_batches=1000,
    previous_tokens=None,
):
    """
    Select dataset indices satisfying a token-level condition.

    Returns:
        dict with keys:
            'indices'     : (N,) dataset indices
            'last_token'  : (N,) token ids
            'prev_token'  : (N,) token ids (or None)
    """
    conditional_tokens = torch.tensor(conditional_tokens, device=device)
    if previous_tokens is not None:
        previous_tokens = torch.tensor(previous_tokens, device=device)

    selected_indices = []
    last_tokens = []
    prev_tokens = []

    N = 0

    with torch.no_grad():
        for batch_i, (x, _, idx) in enumerate(loader):
            if batch_i >= max_batches or N >= num_samples:
                break

            x = x.to(device)
            idx = idx.to(device)

            last_ok = torch.isin(x[:, -1], conditional_tokens)

            if previous_tokens is not None:
                prev_ok = torch.isin(x[:, -2], previous_tokens)
                mask = last_ok & prev_ok
            else:
                mask = last_ok

            idx_sel = torch.nonzero(mask).squeeze(-1)
            remaining = num_samples - N
            idx_sel = idx_sel[:remaining]

            if idx_sel.numel() == 0:
                continue

            selected_indices.append(idx[idx_sel].cpu())
            last_tokens.append(x[idx_sel, -1].cpu())
            prev_tokens.append(x[idx_sel, -2].cpu())

            N += idx_sel.numel()

    return {
        "indices": torch.cat(selected_indices),
        "last_token": torch.cat(last_tokens),
        "prev_token": torch.cat(prev_tokens),
    }