"""Pin PyTorch's CPU thread count: tiny test models run faster on few threads, results do
not depend on machine load or core count (reduction order changes with thread count),
and parallel CPU jobs do not thrash each other."""

import torch

torch.set_num_threads(2)
