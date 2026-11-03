# Machine Learning for Momentum Investing

## Comparing Machine Learning Models for Momentum-Based Stock Selection in the S&P 500

This project investigates whether machine-learning algorithms can improve momentum-based stock selection within the S&P 500.

Using daily equity data from 2018 to 2025, a series of momentum factors are constructed and used to predict one-month forward returns. Predicted returns are transformed into monthly stock rankings, from which equal-weight portfolios are created and evaluated.

Six machine-learning models are compared:

- Linear Regression
- Ridge Regression
- Lasso Regression
- Random Forest
- Gradient Boosting
- Simple Multi-Layer Perceptron (MLP) Neural Network

Performance is evaluated using cumulative return, Sharpe ratio, maximum drawdown and out-of-sample R².

---

## Project Overview

The primary objective of this study is to examine whether different machine-learning algorithms can extract useful investment signals from momentum features and whether more complex models outperform traditional linear approaches.

Particular attention is given to the relationship between prediction accuracy and portfolio performance, as successful investment strategies do not necessarily require accurate point forecasts.

---

## Key Results

| Model | Return | Sharpe | Drawdown | R² |
|---------|---------|---------|---------|---------|
| MLP Neural Network | **74.98%** | 1.014 | -2.90% | -0.018 |
| Random Forest | 45.91% | **1.060** | **-0.44%** | -0.080 |
| Gradient Boosting | 43.70% | 0.719 | -1.51% | -0.020 |
| Lasso Regression | 38.08% | 0.619 | -6.28% | -0.024 |
| Ridge Regression | 33.88% | 0.604 | -4.53% | -0.026 |
| Linear Regression | 33.88% | 0.604 | -4.53% | -0.026 |

Key findings include:

- The MLP Neural Network generated the highest cumulative return.
- Random Forest achieved the strongest risk-adjusted performance.
- Nonlinear models generally outperformed linear models.
- All models produced negative out-of-sample R² values despite generating profitable portfolios, highlighting the importance of ranking ability in quantitative investing.

---

## Repository Guide

### `Walkthrough.ipynb`

The main project deliverable and recommended starting point.

The notebook is written in a research-paper format and contains:

- Research motivation
- Data collection
- Feature engineering
- Momentum factor construction
- Model development
- Portfolio construction
- Results analysis
- Discussion and conclusions

### `Requirments.txt`

To install the required packages:

```bash
pip install -r Scripts/requirements.txt
```
pip install -r Scripts/requirements.txt

### `Scripts/`

Contains the underlying Python implementation and project requirements used throughout the analysis, including data collection, feature engineering, model training, portfolio construction and performance evaluation.

### `Results/`

Contains generated outputs from the analysis, including:

- Model performance statistics
- Sharpe ratio comparison
- Total return comparison
- Maximum drawdown comparison

---

## Methodology

### Data

- Universe: S&P 500 Constituents
- Source: Yahoo Finance (`yfinance`)
- Sample Period: January 2018 – January 2025

### Features

- Daily Return
- 3-Month Momentum
- 6-Month Momentum
- 9-Month Momentum
- 12-Month Momentum

### Portfolio Construction

For each month:

1. Predict future one-month returns.
2. Rank stocks by predicted return.
3. Select the top 50 securities.
4. Form an equal-weight portfolio.
5. Rebalance monthly.

---

## Limitations

Several limitations should be considered when interpreting the results:

- Current S&P 500 constituents introduce survivorship bias.
- Transaction costs are not incorporated.
- Models were evaluated largely using default hyperparameters.
- Only momentum features were included.
- A single chronological train-test split was used.

---

## Future Work

Potential extensions include:

- Walk-forward validation
- Time-series cross-validation
- Hyperparameter optimisation
- Transaction cost modelling
- Multi-factor investing frameworks
- XGBoost and LightGBM implementations
- Advanced neural-network architectures

---

## Author

**Muadh Malallah**  
