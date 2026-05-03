# Geometric Domain Adaptation via Optimal Transport for Linear Regression in $\mathbb{R}^2$


Code accompanying the paper Geometric Domain Adaptation via Optimal Transport for Linear Regression in $\mathbb{R}^2$ on transferring linear
regression between source and target domains using optimal transport.

The method estimates a geometric transformation (rotation, translation, or
homothety) between the source and target domains by transporting K-means
centroids of the source onto the target via optimal transport, and then
applies a closed-form rule (Lemma 1) to map the source-domain regression
parameters into the target domain.

We compare against three baselines:

- **TO** — Target-Only OLS regression (fit directly on the small target set).
- **LFT** — Linear Fine-Tuning (gradient descent on the target, initialized from source OLS).
- **TL** — Trans-Lasso (Lasso fitted on the transported centroids).

Our proposed method is referred to as **ALR** (Adapted Linear Regression).

## Repository structure

```
.
├── src/
│   ├── data.py            # sample generation, rotations, translations
│   ├── transport.py       # Lp distance matrix, centroid OT
│   ├── estimators.py      # rotation / translation / homothety estimators
│   ├── regression.py      # OLS, Lemma 1 rules, estimate_f_parameters
│   ├── baselines.py       # finetuned_regression, trans_lasso_regression
│   └── metrics.py         # MSE
├── experiments/
│   ├── exp1_rotation.py
│   ├── exp2_translation.py
│   └── exp3_homothety.py
├── notebooks/
│   └── figures.ipynb      # reads CSVs from results/ and produces the paper figures
├── results/               # CSV outputs of the experiment scripts
├── figures/               # PDF/PNG figures produced by the notebook
├── requirements.txt
├── LICENSE
└── README.md
```

## Installation

```bash
git clone <this-repo-url>
cd trans-ot-regression

# (optional but recommended) create a virtual environment
python -m venv .venv
source .venv/bin/activate   # on Windows: .venv\Scripts\activate

pip install -r requirements.txt
```

The code has been tested with Python 3.10.

## Reproducing the paper results

Run the three experiment scripts from the repository root:

```bash
python -m experiments.exp1_rotation
python -m experiments.exp2_translation
python -m experiments.exp3_homothety
```

Each script writes a CSV with all per-replication MSEs to `results/`.
All scripts use a fixed seed (`SEED = 42`) for reproducibility.

Then open `notebooks/figures.ipynb` to reproduce the figures and tables
from the paper. The notebook only reads the CSVs in `results/`, so
re-rendering figures does not require re-running the experiments.


## Using the methods on your own data

```python
import numpy as np
from src.regression import estimate_f_parameters
from src.baselines import finetuned_regression, trans_lasso_regression

# Xs : (ns, 2) source samples;  Xt : (nt, 2) target samples (typically nt << ns)
a_to, b_to, a_alr, b_alr = estimate_f_parameters(
    Xs, Xt,
    f_type="R",        # "R"otation, "T"ranslation, or "H"omothety
    p=2,               # Lp norm for the OT cost matrix
    n_rep=1000,        # bootstrap iterations
    sampling_frac=0.8, # subsampling fraction of Xs per iteration
    rng=np.random.default_rng(0),
)

a_lft, b_lft = finetuned_regression(Xs, Xt)
a_tl,  b_tl  = trans_lasso_regression(Xs, Xt)
```

## Citation

If you use this code, please cite:

```bibtex
@article{britos2026geometric,
  title   = {Geometric Domain Adaptation via Optimal Transport for Linear Regression in $\mathbb{R}^2$},
  author  = {Britos, Brian and Bourel, Mathias},
  journal = {Pattern Recognition Letters},
  year    = {2026},
  publisher = {Elsevier},
  note    = {In press},
}
```


## License

MIT — see [LICENSE](LICENSE).
