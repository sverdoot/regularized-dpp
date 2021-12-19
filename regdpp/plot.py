from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from scipy import stats

sns.set_theme()


def plot_results(
    criteria: Dict[str, np.ndarray],
    k_range: np.ndarray,
    d: int,
    dataset: str = "",
    ylabel: str = "",
    savepath: Optional[Path] = None,
    show: bool = False,
    yscale: str = "linear",
    ylim: Optional[Tuple[float, float]] = None,
):
    fig = plt.figure()
    for name, a_value in criteria.items():
        p = plt.plot(k_range / d, a_value.mean(1), label=name)

        if (np.std(a_value, 1) > 0).all():
            ci = stats.bootstrap(
                np.copy((a_value.T,)), np.mean, axis=0, confidence_level=0.95
            ).confidence_interval
            plt.fill_between(k_range / d, ci.low, ci.high, alpha=0.3, color=p[0].get_color())

    plt.xlabel("Subset size (multiple of d)")
    plt.ylabel(ylabel)
    plt.yscale(yscale)
    if ylim:
        plt.ylim(ylim)
    plt.title(f"dataset {dataset}")
    plt.legend()
    if savepath:
        plt.savefig(savepath)
        plt.close()
    elif show:
        plt.show()
    else:
        return fig
