import random

import numpy as np
import torch


def fix_random_state(seed: int = 42, benchmark=False):
    seed = get_random_seed(seed)

    # Python library
    random.seed(seed)

    # Numpy
    np.random.seed(seed)

    # Pytorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = benchmark
    if benchmark:
        print('Benchmark Mode')
    # torch.use_deterministic_algorithms(True)  # RuntimeError
    torch.backends.cudnn.deterministic = True

    torch.backends.cudnn.allow_tf32 = False

    print('Fixed seed: {}'.format(seed))


def get_random_seed(seedval=None):
    if seedval is None:
        return random.randint(1, 1000000)
    return seedval


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
