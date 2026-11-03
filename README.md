# s-p500-machine-learning-comparison-using-momentum-trading
Comparing simple machine learning models for momentum-based stock selection in the S&amp;P 500.

# Machine Learning for Momentum Investing

## Overview

This project compares several machine-learning algorithms for predicting one-month forward stock returns using momentum factors derived from S&P 500 equities.

Stocks are ranked according to predicted future returns and combined into a monthly rebalanced equal-weight portfolio. The performance of multiple models is evaluated using Total Return, Sharpe Ratio, Maximum Drawdown and R².

## Momentum Factors

- 1-Month Momentum
- 3-Month Momentum
- 6-Month Momentum
- 9-Month Momentum
- 12-Month Momentum

## Models

- Linear Regression
- Ridge Regression
- Lasso Regression
- Random Forest
- Gradient Boosting
- MLP Neural Network

## Portfolio Construction

- Universe: S&P 500
- Rebalancing Frequency: Monthly
- Portfolio Size: Top 50 Stocks
- Weighting Scheme: Equal Weight
