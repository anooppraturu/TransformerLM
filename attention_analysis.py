import torch
import matplotlib.pyplot as plt
import numpy as np

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

def measure_conditional_attention_statistics(
    model,
    loader,
    device,
    conditional_tokens,
    num_samples=500,
    max_batches=1000,
    previous_tokens=None,
):
    examples = collect_conditional_examples(
        loader=loader,
        device=device,
        conditional_tokens=conditional_tokens,
        num_samples=num_samples,
        max_batches=max_batches,
        previous_tokens=previous_tokens,
    )

    stats = measure_attention_on_examples(
        model=model,
        loader=loader,
        device=device,
        example_indices=examples["indices"],
    )

    return stats, examples

def get_control_statistics(examples, model, loader, device, conditional_tokens):
    """
    given collection of conditional examples, compute attention statistics on 
    conditional_tokens with second to last token fixed to match statistics of
    prev_token in examples
    """
    prev_controls, counts = examples['prev_token'].unique(return_counts=True)

    control_indices = []

    for tok, cnt in zip(prev_controls, counts):
        ctrl_examples = collect_conditional_examples(
            loader,
            device,
            conditional_tokens=conditional_tokens,
            previous_tokens=[tok.item()],
            num_samples=cnt.item()
        )
        control_indices.append(ctrl_examples["indices"])

    control_indices = torch.cat(control_indices)

    control_stats = measure_attention_on_examples(
        model, loader, device, control_indices
    )

    return control_stats


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


def define_phase_bins(l_all, H_all, num_bins_l=40, num_bins_H=40):
    """
    Returns bin edges for ln(1+<l>) and entropy H.
    """
    qs = torch.linspace(0, 1, num_bins_l + 1, device = l_all.device)
    l_bins = torch.quantile(l_all, qs)
    rs = torch.linspace(0, 1, num_bins_H + 1, device = H_all.device)
    H_bins = torch.quantile(H_all, rs)

    l_bins = torch.unique(l_bins)
    H_bins = torch.unique(H_bins)

    return l_bins, H_bins


def jsd_phase_space(l1, H1, l2, H2, l_bins, H_bins, eps=1e-12):
    """
    Compute Jensen–Shannon divergence between
    two phase-space distributions.
    """
    l1 = l1.cpu().numpy()
    H1 = H1.cpu().numpy()
    l2 = l2.cpu().numpy()
    H2 = H2.cpu().numpy()
    l_bins = l_bins.cpu().numpy()
    H_bins = H_bins.cpu().numpy()

    hist1, _, _ = np.histogram2d(
        l1, H1,
        bins=[l_bins, H_bins]
    )
    hist2, _, _ = np.histogram2d(
        l2, H2,
        bins=[l_bins, H_bins]
    )
    if hist1.sum() == 0 or hist2.sum() == 0:
        return np.nan
    
    # get normalized dists, add small regularizer to avoid log 0
    P = hist1 / hist1.sum()
    Q = hist2 / hist2.sum()
    P = P + eps
    Q = Q + eps
    P /= P.sum()
    Q /= Q.sum()

    M = 0.5*(P+Q)

    D_PM = (P * np.log(P / M)).sum()
    D_QM = (Q * np.log(Q / M)).sum()

    return 0.5*(D_PM + D_QM)


def compute_jsd_matrix(stats_base, stats_rel):
    """
    Returns JSD[layer, head]
    TODO: implement marginal JSD over just scale/entropy. Compare joint JSD to sum of marginal JSD -> assess if changes come from changes in marginals or coupling between scale and entropy
    """
    assert stats_base.keys() == stats_rel.keys()

    l_all = []
    H_all = []
    for _, val in stats_base.items():
        l_all.append(val['scale'].flatten())
        H_all.append(val['entropy'].flatten())
    for _, val in stats_rel.items():
        l_all.append(val['scale'].flatten())
        H_all.append(val['entropy'].flatten())

    l_all = torch.log(1.0 + torch.cat(l_all))
    H_all = torch.cat(H_all)

    l_bins, h_bins = define_phase_bins(l_all, H_all)

    jsd = []
    for layer in stats_base:
        assert stats_base[layer]['scale'].shape[1] == stats_rel[layer]['scale'].shape[1]
        layer_jsds = []
        for head in range(stats_base[layer]['scale'].shape[1]):
            l1 = torch.log(1+stats_base[layer]['scale'][:,head])
            H1 = stats_base[layer]['entropy'][:,head]
            l2 = torch.log(1+stats_rel[layer]['scale'][:,head])
            H2 = stats_rel[layer]['entropy'][:,head]

            layer_jsds.append(jsd_phase_space(l1, H1, l2, H2, l_bins, h_bins))
        jsd.append(layer_jsds)
    
    return np.asarray(jsd)

def plot_jsd_heatmap(jsd, title=None, vmax=None):
    """
    jsd: array of shape (num_layers, num_heads)
    """
    fig, ax = plt.subplots(figsize=(1.2*jsd.shape[1], 1.2*jsd.shape[0]))

    im = ax.imshow(
        jsd,
        origin="lower",
        aspect="auto",
        cmap="magma",
        vmax=vmax
    )

    ax.set_xlabel("Head")
    ax.set_ylabel("Layer")

    ax.set_xticks(range(jsd.shape[1]))
    ax.set_yticks(range(jsd.shape[0]))

    if title is not None:
        ax.set_title(title)

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Jensen–Shannon Divergence")

    plt.tight_layout()
    plt.show()