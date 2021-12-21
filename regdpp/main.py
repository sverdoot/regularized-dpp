import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np
import scipy as sp
import scipy.linalg
import yaml
from matplotlib import pyplot as plt
from tqdm import tqdm, trange

from regdpp.general import DATA_DIR, ROOT_DIR
from regdpp.metrics import A_opt_criterion
from regdpp.plot import plot_results
from regdpp.sample import SamplerRegistry
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
    times = dict()

    A = 1.0 / n * np.eye(d)

    for method, params in config["methods"].items():
        print(params["name"])
        target = params["target"]
        sampler = SamplerRegistry.create_sampler(target, **params["params"])
        for k_id, k in tqdm(list(enumerate(k_range))):
            print(f"Looking for subset of size {k}")

            for rep_id in range(n_repeat):
                S = sampler(X, A, k)

                X_S = X[S]
                subset_cov = X_S.T @ X_S
                A_optimal_criterium = A_opt_criterion(subset_cov, A)
                criteria[params["name"]][k_id, rep_id] = A_optimal_criterium
        sampler.time_cnts = np.array(sampler.time_cnts).reshape((len(k_range), n_repeat))
        if sampler.__class__.__name__ == "Greedy":
            sampler.time_cnts = np.cumsum(sampler.time_cnts[:, [0]], axis=0)
        times[params["name"]] = sampler.time_cnts
        print(
            f"Time of sampling: {np.mean(sampler.time_cnts):.3f} +- {1.96 * np.std(sampler.time_cnts):.3f}"
        )

    plot_results(
        criteria,
        k_range,
        d,
        ylabel="A-optimality value",
        ylim=config["ylim"] if config["ylim"] else None,
        dataset=config["dataset"],
        savepath=Path(ROOT_DIR, config["figpath"], f'{config["dataset"]}'),
    )

    plot_results(
        times,
        k_range,
        d,
        ylabel="Wall time",
        dataset=config["dataset"],
        yscale="log",
        savepath=Path(ROOT_DIR, config["figpath"], f'{config["dataset"]}_time'),
    )

    cov = X.T @ X
    baseline = np.array([A_opt_criterion(k / n * cov, A) for k in k_range])
    criteria = {
        config["methods"][key]["name"]: criteria[config["methods"][key]["name"]] / baseline[:, None]
        for key in ["plain_reg_dpp", "reg_dpp_sdp"]
    }
    fig = plot_results(criteria, k_range, d, dataset=config["dataset"])
    plt.ylabel(r"$f_A(X_S^{\top}X_S)/f_A(\frac{k}{n}\Sigma_X)$")
    plt.axhline(y=1.0, color="black", linestyle="--")
    fig.tight_layout()
    plt.savefig(Path(ROOT_DIR, config["figpath"], f'{config["dataset"]}_baseline'))
    plt.close()


if __name__ == "__main__":
    args = parse_arguments()
    config = yaml.safe_load(Path(args.config).open("r"))
    main(config)
