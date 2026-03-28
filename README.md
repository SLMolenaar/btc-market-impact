# Market Impact in Crypto: Does the Equity Model Apply?

The question we are answering: Does the functional form of crypto market impact match the equity model, or does the data support a different structure? Does the answer change across market regimes?

The square-root law of market impact, I(Q) proportional to sqrt(Q), is the
standard model used in equity execution. This project asks whether the
functional form of crypto impact matches that model, or whether the data
supports a different structure, and whether the answer changes across
market regimes.

Using public Binance BTC/USDT trade data and the metaorder reconstruction
algorithm of Maitrier, Loeper & Bouchaud (2025), we reconstruct synthetic
metaorders from individual trades, train a neural network on the impact
function in two market regimes, and use symbolic regression to extract the
formula the network learned in each case.

## Results so far (calm regime, Dec 2025 / Jan 2026)

The standard equity impact model does not apply. The empirical exponent is
~0.1, far from the theoretical 0.5. The formula PySR extracts from the
trained MLP is:

```
I ~ Q^(-0.246) / sigma
```

Size-dependence is weak and inverted relative to theory. The dominant factor
is daily volatility, encoded as a denominator rather than the linear scaling
assumed by Almgren-Chriss. This result is stable across PySR random seeds
and reconstruction parameter choices.

## Open question (stress regime, Feb 2026)

Does the impact function change during a market crash? The empirical delta
shifts toward 0.5 in the February 2026 data. Whether the full functional
form moves closer to the equity model under stress is what the second
experiment is designed to answer.

This is preliminary evidence on a single asset over three months, not a
definitive result. The goal is to demonstrate the methodology and point at
something worth verifying with real order flow.

## Methodology

Binance does not provide trader IDs, so metaorders cannot be observed
directly. We reconstruct synthetic metaorders using the algorithm of
Maitrier, Loeper & Bouchaud (2025, arXiv:2503.18199), which assigns
synthetic trader IDs to individual trades and groups consecutive same-sign
trades per trader into metaorders. Maitrier et al. show that this
reconstruction reliably recovers the square-root law and other metaorder
stylized facts regardless of parameter choices; the sensitivity check in
01_data.ipynb confirms this holds on this dataset.

Three models are compared in each regime:
1. OLS benchmarks (power law, Almgren-Chriss)
2. MLP trained on metaorder features
3. Closed-form formula extracted from the MLP via PySR symbolic regression

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

## Data

Downloaded from [data.binance.vision](https://data.binance.vision). Free,
no account required. Not included in this repo.

**Calm regime:** BTC/USDT, Dec 2025 and Jan 2026. 64.5M trades, 1.56M reconstructed metaorders.

**Stress regime:** BTC/USDT, Feb 2026 (market crash). Processed separately with the same pipeline.

## Limitations

- Single asset over three months. Not enough to generalize.
- The "calm regime" is not uniform. December 2025 was genuinely low-volatility,
  but late January 2026 saw a sharp macro-driven selloff (Warsh shock, large ETF
  outflows). The calm label applies cleanly to December; January is better
  described as mixed.
- sigma_daily has near-categorical variation (one value per day), limiting
  reliable estimation of the volatility-impact relationship.
- Crypto microstructure differs from equities in ways that make direct
  comparison to the equity literature approximate.

## References

- Maitrier, Loeper & Bouchaud (2025). *Generating realistic metaorders from public data.* arXiv:2503.18199
- Sato & Kanazawa (2024). *Strict universality of the square-root law.* arXiv:2411.13965
- Almgren & Chriss (2001). *Optimal execution of portfolio transactions.* Journal of Risk.
- Bouchaud, Farmer & Lillo (2009). *How markets slowly digest changes in supply and demand.*