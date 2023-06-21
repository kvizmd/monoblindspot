from scipy.optimize import linear_sum_assignment

import torch


def bipartite_match(cost_matrix, split_nums, maximize=False):
    factory_args = {'dtype': torch.int64, 'device': cost_matrix.device}

    # Avoid that matrix contains invalid numeric entries
    cost_matrix = torch.nan_to_num(cost_matrix)

    # hangarian matching
    indices = [
        linear_sum_assignment(c[i].detach().cpu(), maximize=maximize)
        for i, c in enumerate(cost_matrix.split(split_nums, -1))
    ]
    indices = [(
        torch.as_tensor(i, **factory_args),
        torch.as_tensor(j, **factory_args))
        for i, j in indices]
    return indices
