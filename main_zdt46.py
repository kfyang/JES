from pymoo.config import Config

Config.show_compile_hint = False
import numpy as np
import pandas as pd
import torch
from botorch.utils.multi_objective import Hypervolume
from botorch.utils.multi_objective.pareto import is_non_dominated
from botorch.utils.transforms import normalize, unnormalize

from jes.utils.bo_loop import bo_loop
from problems import RE, WOSGZ, ZDT

import os
import sys
path = os.getcwd()

# adjust this for ZDT, RE problems 
d = 10



M = 2
n = min(60, 6 * d)
n_iteration = 170
problem_id = int(sys.argv[1])
run_id = int(sys.argv[2])


i = problem_id 
for i in range(problem_id, problem_id+1):
    problem = ZDT(id=i, dim=d, num_objectives=M, negate=True)
    bounds = problem.bounds
    standard_bounds = torch.zeros(2, d)
    standard_bounds[1] = 1.0

    train_X = (bounds[1, :] - bounds[0, :]) * torch.rand(n, d) + bounds[0, :]
    train_Y = problem(train_X)

    hv_exact = []
    for j in range(n_iteration):
        x = bo_loop(
            train_X=normalize(train_X, bounds),
            train_Y=train_Y,
            num_outputs=M,
            bounds=standard_bounds,
            acquisition_type="jes_lb",
            num_pareto_samples=10,
            num_pareto_points=10,
            num_greedy=10,
            num_samples=128,
            num_restarts=10,
            raw_samples=1000,
            batch_size=1,
        )
        x = unnormalize(x, bounds)
        y = problem(x)

        train_X = torch.cat((train_X, x), 0)
        train_Y = torch.cat((train_Y, y), 0)

        idx = is_non_dominated(train_Y)
        pf = train_Y[idx, :]
        hv = Hypervolume(problem.ref_point)
        hv_exact.append(hv.compute(pf.squeeze(0)))

    N = train_X.shape[0]
    df = pd.DataFrame(
        np.c_[
            [0] * n + list(range(1, n_iteration + 1)),
            train_X.numpy(),
            train_Y.numpy(),
            [0] * N,
            [0] * N,
            [0] * N,
            [0] * N,
            [0] * N,
            [0] * N,
            [0] * n + hv_exact,
        ],
        columns=["iterID"]
        + [f"x{i+1}" for i in range(d)]
        + [
            "f1",
            "f2",
            "Expected_f1",
            "Uncertainty_f1",
            "Acquisition_f1",
            "Expected_f2",
            "Uncertainty_f2",
            "Acquisition_f2",
            "Hypervolume_indicator",
        ],
    )
    rst_path = path + '/nips_jes_results/' + problem.__class__.__name__.lower() + str(i) + '/default/' + 'jes/'
    os.makedirs(rst_path, exist_ok=True) 
    df.to_csv(rst_path + str(run_id) + '.csv')
           
