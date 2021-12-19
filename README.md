# Bayesian experimental design using regularized determinantal point processes

- [Bayesian experimental design using regularized determinantal point processes](#bayesian-experimental-design-using-regularized-determinantal-point-processes)
  - [Intro](#intro)
  - [Installation](#installation)
  - [Usage](#usage)
  - [Citing](#citing)

## Intro
This repo provides implementation of the method proposed in "Bayesian experimental design using regularized determinantal point processes" \[[arxiv](https://arxiv.org/pdf/1906.04133.pdf)\]

## Installation

```bash
conda create -n regdpp python=3.9
conda activate regdpp
pip install poetry
poetry install
chmod +x **.sh
```

```bash
./get_data.sh
```

To obtain optimal weights for method with SDP we use ```cvxpy.MOSEK``` solver, which requires license. The official cite provides academic license. You can try using another solver (e.g. ```cvxpy.SCS```).

## Usage

```bash
python main.py configs/[dataset_name].yaml
```

```python
from regdpp.metrics import A_opt_criterion
from regdpp.sample import SamplerRegistry
from regdpp.sdp import get_optimal_weights

n, d = X.shape
k = 2 * d
sampler = SamplerRegistry.create_sampler("RegDPP")
p = get_optimal_weights(X, A, k)
S = sampler(X, A, p, k)
value = A_opt_criterion(X[S].T @ X[S], A, )

```

<!-- ## Example

![](./figs/mg_scale.pdf) -->

## Citing

```
@InProceedings{pmlr-v108-derezinski20a,
  title = 	 {Bayesian experimental design using regularized determinantal point processes},
  author =       {Derezinski, Michal and Liang, Feynman and Mahoney, Michael},
  booktitle = 	 {Proceedings of the Twenty Third International Conference on Artificial Intelligence and Statistics},
  pages = 	 {3197--3207},
  year = 	 {2020},
  editor = 	 {Chiappa, Silvia and Calandra, Roberto},
  volume = 	 {108},
  series = 	 {Proceedings of Machine Learning Research},
  month = 	 {26--28 Aug},
  publisher =    {PMLR},
  pdf = 	 {http://proceedings.mlr.press/v108/derezinski20a/derezinski20a.pdf},
  url = 	 {https://proceedings.mlr.press/v108/derezinski20a.html}
}
```

