from pathlib import Path
from typing import Dict, Optional

import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from scipy import stats

sns.set_theme()


def plot_results(
    criteria: Dict[str, np.ndarray],
    k_range: np.ndarray,
    d: int,
    savepath: Optional[Path] = None,
    dataset: str = "",
):
    for name, a_value in criteria.items():
        plt.plot(k_range / d, a_value.mean(1), label=name)

        ci = stats.bootstrap(
            (a_value.T,), np.mean, axis=0, confidence_level=0.95
        ).confidence_interval
        plt.fill_between(k_range / d, ci.low, ci.high, alpha=0.3)

    # plt.ylim(0, 11)
    plt.xlabel("Subset size (multiple of d)")
    plt.ylabel("A-optimality value")
    plt.title(f"dataset {dataset}")
    plt.legend()
    if savepath:
        plt.savefig((savepath))
    else:
        plt.show()
