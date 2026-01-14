import torch
import matplotlib.pyplot as plt

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

def measure_conditional_attention_statistics(model, loader, device, conditional_tokens, num_batches=20):
    """
    Returns per-layer, per-head distributions of
    attention scale and entropy, splits statistics 
    based on whether the query token (final position)
    belongs to conditional_tokens
    """
    model.eval()
    model.enable_attention_logging()

    conditional_tokens = torch.tensor(
        conditional_tokens, device=device
    )

    results = {
        cond: {
            layer_idx: {"scale": [], "entropy": []}
            for layer_idx in range(len(model.blocks))
        }
        for cond in ['last_in_set', 'last_not_in_set']
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

                last_token = torch.isin(x[:,-1], conditional_tokens).cpu()

                masks = {
                    "last_in_set": last_token,
                    "last_not_in_set": ~last_token,
                }

                for cond_name, mask in masks.items():
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

def phase_plots(phase_data):
    """
    Plot full phase distribution head centroids.
    Currently assumes depth 4 transformer with 4 heads in each layer.
    """
    fig, ax = plt.subplots(2, 4, figsize=(20,10))

    color = ['#66c2a5', '#fc8d62', '#8da0cb', '#e78ac3']

    for layer_idx in range(4):
        for head in range(4):
            # get data
            logl = torch.log(phase_data["last_in_set"][layer_idx]['scale'][:,head])
            H = phase_data["last_in_set"][layer_idx]['entropy'][:,head]
            # head centroids and standard deviations
            mean_scale = logl.mean()
            mean_H = H.mean()
            std_scale = logl.std()
            std_H = H.std()

            ax[0][layer_idx].scatter(logl, H, c=color[head], alpha=0.5)
            ax[1][layer_idx].errorbar(mean_scale, mean_H, xerr = std_scale, yerr = std_H, c=color[layer_idx], lw=2.5, capsize=3)

        ax[0][layer_idx].set_title('layer {}'.format(layer_idx))
        ax[0][layer_idx].set_xlabel('ln(<l>)')
        ax[0][layer_idx].set_ylabel('H')
        xlims = ax[0][layer_idx].get_xlim()
        ylims = ax[0][layer_idx].get_ylim()
        ax[1][layer_idx].set_xlim(xlims)
        ax[1][layer_idx].set_ylim(ylims)
        ax[1][layer_idx].set_xlabel('ln(<l>)')
        ax[1][layer_idx].set_ylabel('H')

    return fig