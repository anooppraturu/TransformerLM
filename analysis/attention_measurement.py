import torch
from analysis.attention_metrics import attention_entropy, attention_scale
from analysis.example_selection import collect_conditional_examples

def measure_attention_on_examples(
    model,
    loader,
    device,
    example_indices,
):
    """
    Measure attention statistics on a fixed set of dataset indices.

    Returns:
        dict[layer_idx] -> {
            'scale': (N, H),
            'entropy': (N, H)
        }
    """
    model.eval()
    model.enable_attention_logging()

    results = {
        layer_idx: {"scale": [], "entropy": []}
        for layer_idx in range(len(model.blocks))
    }

    with torch.no_grad():
        for x, _, idx in loader:
            idx = idx.cpu()
            mask = torch.isin(idx, example_indices)

            if not mask.any():
                continue

            x = x.to(device)
            _ = model(x)

            sel = torch.nonzero(mask).squeeze(-1)

            for layer_idx, block in enumerate(model.blocks):
                attn = block.attn.last_attention      # (B, H, T, T)
                a = attn[:, :, -1, :]                 # last query

                l = attention_scale(a)[sel].cpu()
                H = attention_entropy(a)[sel].cpu()

                results[layer_idx]["scale"].append(l)
                results[layer_idx]["entropy"].append(H)

    model.disable_attention_logging()

    for layer_idx in results:
        results[layer_idx]["scale"] = torch.cat(results[layer_idx]["scale"], dim=0)
        results[layer_idx]["entropy"] = torch.cat(results[layer_idx]["entropy"], dim=0)

    return results