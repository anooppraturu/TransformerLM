import torch
import numpy as np

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