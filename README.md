# Price Impact: What Formula Does a Neural Network Learn?

Training an MLP on synthetic metaorders reconstructed from Binance BTC/USDT
public trade data, then using symbolic regression to extract the formula it
learns — and asking whether that formula changes across market regimes.

**Research question:** Does a neural network trained on reconstructed metaorder
impact data rediscover the square-root law I(Q) ∝ √Q, and does the learned
formula change across market regimes (volatility, book thickness, order imbalance)?

**Key methodological step:** Since Binance does not provide trader IDs, we
reconstruct synthetic metaorders using the algorithm of Maitrier, Loeper &
Bouchaud (2025, arXiv:2503.18199), which assigns synthetic trader IDs to
individual trades and groups consecutive same-sign trades per trader into
metaorders. This approach has been shown to recover all established stylized
facts of metaorder impact including the square-root law.

## Setup

```bash
git clone https://github.com/SLMolenaar/price-impact-research
cd price-impact-research
pip install -r requirements.txt
```

## Reproducing the data pipeline

```bash
# Step 1: download 3 months of BTC/USDT aggTrades from data.binance.vision
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
| `01_data.ipynb` | Individual trade level: data checks, baseline δ ≈ 0 |
| `01b_metaorders.ipynb` | Metaorder reconstruction, verify δ ≈ 0.5 recovered |
| `02_benchmark.ipynb` | OLS power law fit, Almgren-Chriss form |
| `03_mlp.ipynb` | MLP training, out-of-sample evaluation |
| `04_interpretability.ipynb` | Sensitivity analysis, PySR, regime analysis |
| `05_results.ipynb` | Summary table, Diebold-Mariano tests |

## Key references

- Maitrier, Loeper & Bouchaud (2025). *Generating realistic metaorders from public data.* arXiv:2503.18199
- Sato & Kanazawa (2024). *Strict universality of the square-root law.* arXiv:2411.13965

## Data

Downloaded from [data.binance.vision](https://data.binance.vision). Free,
no account required. Not included in this repo.

- Symbol: BTC/USDT
- Period: December 2025 - February 2026
- ~117M individual trades