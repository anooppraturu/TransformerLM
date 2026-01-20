import matplotlib.pyplot as plt
import torch

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