# Price Impact: What Formula Does a Neural Network Learn?

Training an MLP on synthetic metaorders reconstructed from Binance BTC/USDT
public trade data, using symbolic regression to extract the formula it learns,
and comparing that formula across calm and stressed market conditions.

**Research question:** Does a neural network trained on reconstructed metaorder impact data rediscover the square-root law I(Q) proportional to sqrt(Q), and does the learned formula change across market regimes (volatility, volume, time of day)?



**Key methodological step:** Since Binance does not provide trader IDs, we
reconstruct synthetic metaorders using the algorithm of Maitrier, Loeper &
Bouchaud (2025, arXiv:2503.18199), which assigns synthetic trader IDs to
individual trades and groups consecutive same-sign trades per trader into
metaorders.

**Two full experiments:**
- Calm regime (Dec 2025 / Jan 2026): empirical delta ~ 0.1, far from the
  theoretical 0.5. Train OLS and MLP. Extract formula via PySR.
- Stress regime (Feb 2026, market crash): empirical delta shifts toward 0.5.
  Train fresh OLS and MLP. Extract formula via PySR. Compare to calm regime.

## Setup

```bash
git clone https://github.com/SLMolenaar/price-impact-research
cd price-impact-research
pip install -r requirements.txt
```

## Reproducing the data pipeline

```bash
# Step 1: download BTC/USDT aggTrades from data.binance.vision
python src/fetch_data.py

# Step 2: compute impact and rolling features
python src/process_data.py

# Step 3: reconstruct synthetic metaorders (Maitrier et al. 2025)
python src/reconstruct_metaorders.py

# Step 4: open notebooks in order
jupyter lab
```

## Notebooks

| Notebook | Contents |
|---|---|
| `01_data.ipynb` | Data checks, individual trade baseline, metaorder reconstruction |
| `02_benchmark.ipynb` | OLS benchmarks, calm regime |
| `03_mlp.ipynb` | MLP-A and MLP-B, calm regime |
| `04_interpretability.ipynb` | Sensitivity analysis, PySR symbolic regression, calm regime |
| `05_stress.ipynb` | Full pipeline repeat on Feb 2026 stress regime |
| `06_results.ipynb` | Comparison tables, Diebold-Mariano tests |

## Key references

- Maitrier, Loeper & Bouchaud (2025). *Generating realistic metaorders from public data.* arXiv:2503.18199
- Sato & Kanazawa (2024). *Strict universality of the square-root law.* arXiv:2411.13965

## Data

Downloaded from [data.binance.vision](https://data.binance.vision). Free,
no account required. Not included in this repo.

**Calm regime:**
- Symbol: BTC/USDT, Dec 2025 and Jan 2026
- 64.5M individual trades, 1.56M reconstructed metaorders

**Stress regime:**
- Symbol: BTC/USDT, Feb 2026 (market crash)
- Processed separately with the same pipeline