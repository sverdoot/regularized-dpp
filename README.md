# Bayesian experimental design using regularized determinantal point processes

- [Bayesian experimental design using regularized determinantal point processes](#bayesian-experimental-design-using-regularized-determinantal-point-processes)
  - [Intro](#intro)
  - [Installation](#installation)
  - [Usage](#usage)
  - [Example](#example)
      - [mg_scale](#mg_scale)
      - [bodyfat_scale](#bodyfat_scale)
      - [space_ga_scale](#space_ga_scale)
  - [Citing](#citing)

## Intro
This repo provides implementation of the method proposed in "Bayesian experimental design using regularized determinantal point processes" \[[arxiv](https://arxiv.org/pdf/1906.04133.pdf)\]

You can run a Colab notebook:  [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1PNC590nFG_8tFvtAnFMwSj5MXsFE3vIF?usp=sharing)


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
python regdpp/main.py configs/[dataset_name].yaml
```

To reproduce all the experiments:

```bash
python ./runs/run.sh
```

General usage: 

```python
from regdpp.metrics import A_opt_criterion
from regdpp.sample import SamplerRegistry
from regdpp.sdp import get_optimal_weights

# X - n x d data matrix
# A - d x d precision matrix

n, d = X.shape
k = 2 * d   # size of set of indices to choose
sampler = SamplerRegistry.create_sampler("RegDPP", **{"sdp"=True})  # define a sampler
S = sampler(X, A, k)
value = A_opt_criterion(X[S].T @ X[S], A) # get A-optimality value

```

## Example


#### mg_scale 
<img src="./figs/mg_scale.png" alt="Dependence of A-optimality value on size $k$" width="350"/>  <img src="./figs/mg_scale_baseline.png" alt="A-optimality value devided by baseline" width="350"/> <img src="./figs/mg_scale_time.png" alt="Time comparison" width="350"/>

#### bodyfat_scale 
<img src="./figs/bodyfat_scale.png" alt="Dependence of A-optimality value on size $k$" width="350"/>  <img src="./figs/bodyfat_scale_baseline.png" alt="A-optimality value devided by baseline" width="350"/> <img src="./figs/bodyfat_scale_time.png" alt="Time comparison" width="350"/>

#### space_ga_scale 
<img src="./figs/space_ga_scale.png" alt="Dependence of A-optimality value on size $k$" width="350"/>  <img src="./figs/space_ga_scale_baseline.png" alt="A-optimality value devided by baseline" width="350"/> <img src="./figs/space_ga_scale_time.png" alt="Time comparison" width="350"/>


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

