import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np
import scipy as sp
import scipy.linalg
import yaml
from tqdm import tqdm, trange

from regdpp.general import DATA_DIR, ROOT_DIR
from regdpp.metrics import A_opt_criterion
from regdpp.plot import plot_results
from regdpp.sample import RegDPP
from regdpp.sdp import get_optimal_weights
from regdpp.utils import load_libsvm_data


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=str)
    args = parser.parse_args()
    return args


def main(config):
    X = load_libsvm_data(Path(DATA_DIR, config["dataset"]))
    n, d = X.shape

    shared = config["shared"]

    k_range = np.linspace(
        shared["min_size_scale"] * d, shared["max_size_scale"] * d, shared["n_sizes"], dtype=int
    )
    n_repeat = shared["n_repeat"]
    criteria = defaultdict(lambda: np.empty((len(k_range), n_repeat)))

    for method, params in config["methods"].items():
        print(params["name"])
        for k_id, k in tqdm(list(enumerate(k_range))):
            print(f"Looking for subset of size {k}")
            A = 1.0 / n * np.eye(d)

            if "reg_dpp" in method:
                sampler = RegDPP()
                if params["sdp"]:
                    p = get_optimal_weights(X, A, k)
                else:
                    p = np.ones(n) * k / n
            else:
                raise KeyError("sampler undefined")

            for rep_id in trange(n_repeat):
                S = sampler(X, A, p)

                X_S = X[S]
                subset_cov = X_S.T @ X_S
                A_optimal_criterium = A_opt_criterion(subset_cov, A)
                criteria[params["name"]][k_id, rep_id] = A_optimal_criterium
        print(
            f"Time of sampling: {np.mean(sampler.time_cnts):.3f} +- {1.96 * np.std(sampler.time_cnts):.3f}"
        )

    plot_results(
        criteria,
        k_range,
        d,
        dataset=config["dataset"],
        savepath=Path(ROOT_DIR, config["figpath"], f'{config["dataset"]}.pdf'),
    )


if __name__ == "__main__":
    args = parse_arguments()
    config = yaml.safe_load(Path(args.config).open("r"))
    main(config)
