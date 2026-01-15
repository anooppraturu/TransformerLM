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

def measure_conditional_attention_statistics(model, loader, device, conditional_tokens, num_batches=20, previous_tokens = None):
    """
    Returns per-layer, per-head distributions of
    attention scale and entropy, retaining stats 
    only if the query token (final position) 
    belongs to conditional_tokens
    """
    model.eval()
    model.enable_attention_logging()

    conditional_tokens = torch.tensor(conditional_tokens, device=device)
    if previous_tokens is not None:
        previous_tokens = torch.tensor(previous_tokens, device=device)

    results = {
        layer_idx: {"scale": [], "entropy": []}
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

                last_token = torch.isin(x[:,-1], conditional_tokens).cpu()

                if previous_tokens is not None:
                    second_last_token = torch.isin(x[:, -2], previous_tokens).cpu()
                    mask = last_token & second_last_token
                else:
                    mask = last_token

                if mask.any():
                    results[layer_idx]['scale'].append(l[mask])
                    results[layer_idx]['entropy'].append(H[mask])

    model.disable_attention_logging()

    for layer_idx in results:
        results[layer_idx]["scale"] = torch.cat(results[layer_idx]["scale"], dim=0)
        results[layer_idx]["entropy"] = torch.cat(results[layer_idx]["entropy"], dim=0)
        
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
            logl = torch.log(1.0 + phase_data[layer_idx]['scale'][:,head])
            H = phase_data[layer_idx]['entropy'][:,head]
            # head centroids and standard deviations
            mean_scale = logl.mean()
            mean_H = H.mean()
            std_scale = logl.std()
            std_H = H.std()

            ax[0][layer_idx].scatter(logl, H, c=color[head], alpha=0.5)
            ax[1][layer_idx].errorbar(mean_scale, mean_H, xerr = std_scale, yerr = std_H, c=color[head], lw=2.5, capsize=3)

        ax[0][layer_idx].set_title('layer {}'.format(layer_idx))
        ax[0][layer_idx].set_xlabel('ln(1 + <l>)')
        ax[0][layer_idx].set_ylabel('H')
        xlims = ax[0][layer_idx].get_xlim()
        ylims = ax[0][layer_idx].get_ylim()
        ax[1][layer_idx].set_xlim(xlims)
        ax[1][layer_idx].set_ylim(ylims)
        ax[1][layer_idx].set_xlabel('ln(1 + <l>)')
        ax[1][layer_idx].set_ylabel('H')

    return fig