import torch

def attention_scale(attn):
    """
    attn: tensor of shape (B, H, T)
    returns: (B, H) expected backward distance
    """
    T = attn.size(-1)
    i = (T - 1) - torch.arange(T, device=attn.device)
    return (i * attn).sum(dim=-1)

def attention_entropy(attn, eps=1e-9):
    """
    attn: Tensor of shape (B, H, T)
    returns: (B, H) entropy
    """
    return -(attn * (attn + eps).log()).sum(dim=-1)

def measure_attention_statistics(model, loader, device, num_batches=20):
    """
    Returns per-layer, per-head distributions of
    attention scale and entropy.
    """
    model.eval()
    model.enable_attention_logging()

    results = {
        layer_idx: {'scale': [], 'entropy': []} 
        for layer_idx in range(len(model.blocks))
    }

    with torch.no_grad():
        for i, (x,_) in enumerate(loader):
            if i >= num_batches:
                break

            x = x.to(device)
            _ = model(x)

            for layer_idx, block in enumerate(model.blocks):
                attn = block.attn.last_attention    # (B, H, T, T)
                a = attn[:, :, -1, :]               # last query

                l = attention_scale(a).cpu()
                H = attention_entropy(a).cpu()

                results[layer_idx]['scale'].append(l)
                results[layer_idx]['entropy'].append(H)

    model.disable_attention_logging()

    for layer_idx in results:
        results[layer_idx]["scale"] = torch.cat(results[layer_idx]["scale"], dim=0)
        results[layer_idx]["entropy"] = torch.cat(results[layer_idx]["entropy"], dim=0)

    return results

def measure_conditional_attention_statistics(model, loader, device, num_batches=20, white_space_token = 1):
    """
    Returns per-layer, per-head distributions of
    attention scale and entropy, but conditional on:
    1) final token is whitespace/not
    2) next token is whitespace/not
    """
    model.eval()
    model.enable_attention_logging()

    CONDITIONS = {
        "last=0,next=0": lambda lw, nw: (~lw) & (~nw),
        "last=0,next=1": lambda lw, nw: (~lw) & (nw),
        "last=1,next=0": lambda lw, nw: (lw) & (~nw),
        "last=1,next=1": lambda lw, nw: (lw) & (nw),
    }

    results = {
        cond: {
            layer_idx: {"scale": [], "entropy": []}
            for layer_idx in range(len(model.blocks))
        }
        for cond in CONDITIONS
    }
    
    with torch.no_grad():
        for i, (x,y) in enumerate(loader):
            if i >= num_batches:
                break

            x = x.to(device)
            y = y.to(device)
            _ = model(x)

            for layer_idx, block in enumerate(model.blocks):
                attn = block.attn.last_attention    # (B, H, T, T)
                a = attn[:, :, -1, :]               # last query

                l = attention_scale(a).cpu()
                H = attention_entropy(a).cpu()

                last_white = (x[:,-1] == white_space_token).cpu()
                next_white = (y[:,-1] == white_space_token).cpu()

                for cond_name, mask_fn in CONDITIONS.items():
                    mask = mask_fn(last_white, next_white)
                    if mask.any():
                        results[cond_name][layer_idx]["scale"].append(l[mask])
                        results[cond_name][layer_idx]["entropy"].append(H[mask])

    model.disable_attention_logging()

    for cond in results:
        for layer_idx in results[cond]:
            for key in ("scale", "entropy"):
                if results[cond][layer_idx][key]:
                    results[cond][layer_idx][key] = torch.cat(
                        results[cond][layer_idx][key], dim=0
                    )
                else:
                    results[cond][layer_idx][key] = None
                    
    return results

