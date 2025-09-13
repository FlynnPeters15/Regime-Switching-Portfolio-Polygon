# Regime Switching Portfolio (Polygon.io)

**A Python-based implementation of a regime-switching portfolio strategy that uses Polygon.io data to detect market regimes (Bull, Bear, High-Volatility) with a Hidden Markov Model, dynamically reallocates assets, and backtests performance against the SPY ETF benchmark.**

![Python](https://img.shields.io/badge/Language-Python-blue?logo=python)  
![Status](https://img.shields.io/badge/Status-Active-brightgreen)  
![Project](https://img.shields.io/badge/Project-Regime_Switching_Portfolio-blue)

---

## Overview

This repository contains a **quantitative trading research project** that implements regime-switching portfolio allocation using **Hidden Markov Models (HMMs)**.  
The example implementation is based on **my personal portfolio** of BTC-USD, VOO, VUG, and GLDM, benchmarked against SPY. However, the framework is flexible — anyone is welcome to add their own assets into the model by editing the `ASSETS` list and updating the allocation policy. This makes the project both a teaching tool and a customizable backtesting framework.

---

## Features & Approach

- **Polygon.io Integration**: Fetches historical crypto and ETF data.  
- **Feature Engineering**: Returns, volatility, momentum, and drawdown.  
- **Hidden Markov Model**: Detects Bull, Bear, and High-Volatility regimes.  
- **Dynamic Allocation Policy**: Adjusts portfolio weights by regime.  
- **Backtesting**: Compares cumulative returns vs. SPY.  
- **Visualization**: Matplotlib plots of regime strategy vs benchmark.  

---

## Portfolio Policy

| Regime      | BTC-USD | VOO  | VUG  | GLDM |
|-------------|---------|------|------|------|
| **BULL**    | 40%     | 40%  | 10%  | 10%  |
| **BEAR**    | 10%     | 20%  | 0%   | 70%  |
| **HIGHVOL** | 20%     | 30%  | 10%  | 40%  |

---

## Example Output

### Console
```
Downloading Polygon data for ['BTC-USD', 'VOO', 'VUG', 'GLDM', 'SPY']...
Downloaded price shape: (730, 5), dates: 2023-09-15 → 2025-09-13
Final cumulative returns:
Strategy 2.16
SPY Benchmark 2.05
```

### Plot
The script outputs a chart of cumulative returns:

- **Strategy**: regime-based allocations  
- **SPY Benchmark**: 100% SPY buy & hold  

![example_plot](https://via.placeholder.com/700x350.png?text=Strategy+vs+SPY+Benchmark)

---

## Getting Started

### Prerequisites
- Python 3.x  
- Libraries: `pandas`, `numpy`, `matplotlib`, `hmmlearn`, `pydantic`, `polygon-api-client`

### Installation & Usage
```bash
# Clone the repository
git clone https://github.com/<your-username>/regime-switching-portfolio-polygon.git
cd regime-switching-portfolio-polygon

# Install dependencies
pip install -r requirements.txt

# Run the strategy
python regime_switching_portfolio_polygon.py
```

Before running, set your **Polygon REST API key**:
```bash
export POLYGON_API_KEY="your_polygon_api_key"
```

---

## License
MIT License
