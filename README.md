# Market Impact in Crypto: Does the Equity Model Apply?

The question: does the functional form of crypto market impact match the equity model, or does the data support a different structure? Does the answer change across market regimes?

The square-root law of market impact, I(Q) proportional to sqrt(Q), is the standard model used in equity execution. This project asks whether the functional form of crypto impact matches that model, and whether the answer changes across market regimes.

Using public Binance BTC/USDT trade data and the metaorder reconstruction algorithm of Maitrier, Loeper & Bouchaud (2025), we reconstruct synthetic metaorders from individual trades, train a neural network on the impact function in two market regimes, and use symbolic regression to extract the formula the network learned in each case.

## Results

The equity sqrt law does not hold on 2025 BTC/USDT in either regime. The empirical size exponent is ~0.1 in both calm and stress conditions, far from the theoretical 0.5. The functional form differs between regimes, but not in the direction of the equity model.

### Calm regime (Dec 2025 / Jan 2026)

The formula PySR extracts from the trained MLP is:

```
I ~ Q^(-0.246) / sigma
```

Size-dependence is weak and inverted relative to theory. The dominant factor is daily volatility in the denominator. This result is stable across PySR random seeds and reconstruction parameter choices.

### Stress regime (Feb 2026, market crash)

Sigma spikes 4.7x vs calm. The PySR formula changes structurally:

```
log I ≈ (8.132 / log_sigma) + (-19.628 / log_V)
```

Log_Q disappears entirely. Under stress, metaorder size carries no predictive value for normalized impact. Volatility and volume determine impact; size does not.

### Regime comparison

|  | Calm | Stress |
|---|---|---|
| OLS delta | 0.107 | 0.096 |
| MLP-A mean local slope | 0.044 | 0.041 |
| PySR formula | Q^(-0.246) / sigma | f(sigma, V) only |
| log_Q in formula | Yes (negative) | No |
| Matches equity model | No | No |

The crash did not move delta toward 0.5. It stayed flat and the impact function became less dependent on size, not more.

![Impact formula slope comparison](outputs/figures/formula_comparison_regimes.png)

*All empirical lines (calm and stress, MLP and PySR) lie in a narrow band near zero slope. The sqrt law reference reaches 3.5 log-units above them at the right edge. The stress PySR is flat, consistent with log_Q being absent from the formula.*

![Size exponent by method and regime](outputs/figures/delta_comparison.png)

*OLS delta and MLP mean local slope are stable across regimes and both far below the theoretical 0.5.*

### Model performance (OOS MSE)

| Model | Calm | Stress |
|---|---|---|
| Power law (delta=0.5, constrained) | 1.282 | 0.994 |
| Power law (delta fitted) | 1.063 | 0.618 |
| Almgren-Chriss | 0.672 | 0.608 |
| MLP-A | 0.685 | 0.622 |
| MLP-B | 0.604 | 0.561 |

MLP-A matches Almgren-Chriss in calm. MLP-B beats it by ~10% in both regimes. R² is near zero for all models in stress, reflecting the low predictability of the crash period.

## Prior work

Donier & Bonart (2014) confirm the sqrt law on Bitcoin/USD using a complete dataset with real trader IDs from 2014, when Bitcoin was a small, illiquid market with near-zero statistical arbitrage. This project uses the Maitrier et al. (2025) reconstruction on 2025 data, where the market structure is fundamentally different: 10x the volume, professional market makers, and continuous arbitrage with perpetual futures. The difference in findings is consistent with genuine market structure differences, though reconstruction quality without ground-truth trader IDs cannot be ruled out as a contributing factor.

## Methodology

Binance does not provide trader IDs, so metaorders cannot be observed directly. We reconstruct synthetic metaorders using the algorithm of Maitrier, Loeper & Bouchaud (2025, arXiv:2503.18199), which assigns synthetic trader IDs to individual trades and groups consecutive same-sign trades per trader into metaorders.

Three models are compared in each regime:
1. OLS benchmarks (power law, Almgren-Chriss)
2. MLP trained on metaorder features
3. Closed-form formula extracted from the MLP via PySR symbolic regression

![Metaorder price impact vs size](outputs/figures/loglog_metaorders.png)

*Log-log plot of binned median impact vs metaorder size (calm regime). The OLS fit gives delta = 0.114. The sqrt law (delta = 0.5) is shown for reference.*

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

For the stress regime, set `MONTHS = ["2026-02"]` in `fetch_data.py` and `process_data.py` and save the output as `impact_data_stress.parquet` before running `05_stress.ipynb`.

## Notebooks

| Notebook | Contents |
|---|---|
| `01_data.ipynb` | Data checks, individual trade baseline, metaorder reconstruction |
| `02_benchmark.ipynb` | OLS benchmarks, calm regime |
| `03_mlp.ipynb` | MLP-A and MLP-B, calm regime |
| `04_interpretability.ipynb` | Sensitivity analysis, PySR symbolic regression, calm regime |
| `05_stress.ipynb` | Full pipeline repeat on Feb 2026 stress regime |
| `06_results.ipynb` | Comparison tables, Diebold-Mariano tests, final figures |

## Data

Downloaded from [data.binance.vision](https://data.binance.vision). Free, no account required. Not included in this repo.

**Calm regime:** BTC/USDT, Dec 2025 and Jan 2026. 64.5M trades, 1.56M reconstructed metaorders.

**Stress regime:** BTC/USDT, Feb 2026 (market crash). 1.24M reconstructed metaorders. Processed separately with the same pipeline.

## Limitations

- Single asset over three months. Not enough to generalize.
- The calm regime is not uniform. December 2025 was genuinely low-volatility, but late January 2026 saw a sharp macro-driven selloff (Warsh shock, large ETF outflows). The calm label applies cleanly to December; January is better described as mixed.
- sigma_daily has near-categorical variation (one value per day), limiting reliable estimation of the volatility-impact relationship.
- No ground-truth trader IDs. The reconstruction produces synthetic metaorders whose size distribution may not reflect real institutional flow, particularly on retail-dominated spot markets.
- Crypto microstructure differs from equities in ways that make direct comparison to the equity literature approximate.

## References

- Donier & Bonart (2014). *A million metaorder analysis of market impact on the Bitcoin.* arXiv:1412.4503
- Maitrier, Loeper & Bouchaud (2025). *Generating realistic metaorders from public data.* arXiv:2503.18199
- Sato & Kanazawa (2024). *Strict universality of the square-root law.* arXiv:2411.13965
- Almgren & Chriss (2001). *Optimal execution of portfolio transactions.* Journal of Risk.
- Bouchaud, Farmer & Lillo (2009). *How markets slowly digest changes in supply and demand.*