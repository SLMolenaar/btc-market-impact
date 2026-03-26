# Price Impact: What Formula Does a Neural Network Learn?

Applying symbolic regression to distill what an MLP trained on Binance
BTC/USDT order flow has learned about price impact, and whether it
recovers the square-root law.

## Setup

```bash
git clone https://github.com/SLMolenaar/price-impact-research
cd price-impact-research
pip install -r requirements.txt
```

## Reproducing the data pipeline

```bash
# Step 1: download raw data (~3 months BTC/USDT from data.binance.vision)
python src/fetch_data.py

# Step: merge trades + order book, compute impact and features
python src/process_data.py

# Step 3: open notebooks in order
jupyter lab
```

## Notebooks

| Notebook | Contents |
|---|---|
| `01_data.ipynb` | Data checks, log-log plots, empirical power law exponent |
| `02_benchmark.ipynb` | OLS power law fit, Almgren-Chriss form |
| `03_mlp.ipynb` | MLP training, out-of-sample evaluation |
| `04_interpretability.ipynb` | Sensitivity analysis, PySR symbolic regression, regime analysis |
| `05_results.ipynb` | Summary table, Diebold-Mariano tests |

## Data

Raw data is downloaded from [data.binance.vision](https://data.binance.vision)
(free, no account required). Not included in this repo.