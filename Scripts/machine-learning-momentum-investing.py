import yfinance as yf
import pandas as pd
from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor
)

from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.linear_model import (
    LinearRegression,
    Ridge,
    Lasso
)
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

#imports all thats needed and gets rid of filterwarnings

sp500 = pd.read_csv(
    "https://raw.githubusercontent.com/datasets/s-and-p-500-companies/main/data/constituents.csv"
)

tickers = sp500["Symbol"].tolist()

tickers = [
    ticker.replace(".", "-")
    for ticker in tickers
]

print("Number of stocks:", len(tickers))

#this code imports the S&P 500 tickers from a CSV file and replaces any periods in the ticker symbols with hyphens, which is the format used by Yahoo Finance. It then prints the number of stocks in the S&P 500.

prices = yf.download(
    tickers,
    start="2018-01-01",
    end="2025-01-01"
)

#imports the tickers and downloads the data from yahoo finance for the given time period

close_prices = prices["Close"]

close_long = close_prices.stack()

close_long = close_long.reset_index()

close_long.columns = ["Date", "Ticker", "Close"]

#reformats from wide to long format, resets the index and renames the columns accordingly

close_long["Returns"] = (
    close_long
    .groupby("Ticker")["Close"]
    .pct_change()
)

close_long["Momentum_1M"] = (
    close_long
    .groupby("Ticker")["Close"]
    .pct_change(21)
)

close_long["Momentum_3M"] = (
    close_long
    .groupby("Ticker")["Close"]
    .pct_change(63)
)

close_long["Momentum_6M"] = (
    close_long
    .groupby("Ticker")["Close"]
    .pct_change(126)
)

close_long["Momentum_9M"] = (
    close_long
    .groupby("Ticker")["Close"]
    .pct_change(189)
)

close_long["Momentum_12M"] = (
    close_long
    .groupby("Ticker")["Close"]
    .pct_change(252)
)

#adds the returns and momentum factors to the dataframe

close_long["Future_Return_1M"] = (
    close_long
    .groupby("Ticker")["Close"]
    .pct_change(21) #Look at the percentage change over the last 21 days
    .shift(-21) #Shift the values up by 21 rows to get the future return
)

#adds the future return factor to the dataframe

close_long = close_long.dropna() #Drop any remaining null values for modeling

x = close_long[
    [
        "Returns",
        "Momentum_3M",
        "Momentum_6M",
        "Momentum_9M",
        "Momentum_12M",
    ]
]

#x is what the model will use to predict the future return

y = close_long["Future_Return_1M"]

#y is what the model will try to predict

x_train, x_test, y_train, y_test = train_test_split(
    x,
    y,
    test_size=0.2, #20% of the data will be used for testing
    shuffle=False #Normally sklearn randomizes rows. We don't want that because we want to keep the time series order
)

model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

#empty random forest model

model.fit(x_train, y_train) #model is trained on the training data

print("\n--- FEATURE IMPORTANCE ---")

for feature, importance in sorted(
    zip(x.columns, model.feature_importances_),
    key=lambda x: x[1],
    reverse=True
):
    print(f"{feature}: {importance:.2%}") #prints the feature importance of each factor in the random forest model with 2 decimal places

predictions = model.predict(x_test) #model given the test data to make predictions and stores them in the predictions variable

prediction_df = close_long.loc[x_test.index].copy() #creates a new dataframe with the same index as the test data and copies the values from the original dataframe

prediction_df["Month"] = (
    prediction_df["Date"]
    .dt.to_period("M")
) #adds a new column to the prediction dataframe that contains the month of each date in the original dataframe

prediction_df["Actual"] = y_test.values #adds the actual future returns to the prediction dataframe

prediction_df["Predicted"] = predictions #adds the predicted future returns to the prediction dataframe

prediction_df["Rank"] = (
    prediction_df
    .groupby("Month")["Predicted"]
    .rank(ascending=False)
)

#sort stocks from best to worst based on the predicted future returns and asign this to "Rank" column

top_stocks = prediction_df[
    prediction_df["Rank"] <= 50
]

#Keep only the stocks ranked 1 through 50 on each month

strategy_return = top_stocks["Actual"].mean()

market_return = prediction_df["Actual"].mean()

portfolio_returns = (
    top_stocks
    .groupby("Month")["Actual"]
    .mean()
)

##take the average of the actual future returns for the top 50 stocks each month

cumulative_returns = (
    1 + portfolio_returns
).cumprod()

#cumprod is used to calculate the cumulative product of the returns, which gives the overall growth of the portfolio over time.

sharpe_ratio = (
    portfolio_returns.mean()
    /
    portfolio_returns.std()
)

#sharpe ratio is a measure of risk-adjusted return, calculated as the mean return divided by the standard deviation of returns. A higher Sharpe ratio indicates better risk-adjusted performance.

running_max = cumulative_returns.cummax() # Calculate the running maximum of cumulative returns

drawdown = (
    cumulative_returns - running_max
) / running_max #current cumulative return minus the running maximum divided by the running maximum to get the drawdown

max_drawdown = drawdown.min() # Find the maximum drawdown

results = []

results.append({
    "Model": "Random Forest",
    "Return": cumulative_returns.iloc[-1] - 1,
    "Sharpe": sharpe_ratio,
    "Drawdown": max_drawdown,
    "R2": r2_score(y_test, predictions)
})

#add results to the results list, which will be used to compare the performance of different models, now we repeat for the other models, starting with linear regression

model = LinearRegression()

model.fit(x_train, y_train)

predictions = model.predict(x_test)

prediction_df = close_long.loc[x_test.index].copy()

prediction_df["Month"] = (
    prediction_df["Date"]
    .dt.to_period("M")
)

prediction_df["Actual"] = y_test.values

prediction_df["Predicted"] = predictions

prediction_df["Rank"] = (
    prediction_df
    .groupby("Month")["Predicted"]
    .rank(ascending=False)
)

top_stocks = prediction_df[
    prediction_df["Rank"] <= 50
]

portfolio_returns = (
    top_stocks
    .groupby("Month")["Actual"]
    .mean()
)

cumulative_returns = (
    1 + portfolio_returns
).cumprod()

sharpe_ratio = (
    portfolio_returns.mean()
    /
    portfolio_returns.std()
)

running_max = cumulative_returns.cummax()

drawdown = (
    cumulative_returns - running_max
) / running_max

max_drawdown = drawdown.min()

results.append({
    "Model": "Linear Regression",
    "Return": cumulative_returns.iloc[-1] - 1,
    "Sharpe": sharpe_ratio,
    "Drawdown": max_drawdown,
    "R2": r2_score(y_test, predictions)
})

print("Running Linear Regression")

#now repeat for ridge regression

model = Ridge()

model.fit(x_train, y_train)

predictions = model.predict(x_test)

prediction_df = close_long.loc[x_test.index].copy()

prediction_df["Month"] = (
    prediction_df["Date"]
    .dt.to_period("M")
)

prediction_df["Actual"] = y_test.values

prediction_df["Predicted"] = predictions

prediction_df["Rank"] = (
    prediction_df
    .groupby("Month")["Predicted"]
    .rank(ascending=False)
)

top_stocks = prediction_df[
    prediction_df["Rank"] <= 50
]

portfolio_returns = (
    top_stocks
    .groupby("Month")["Actual"]
    .mean()
)

cumulative_returns = (
    1 + portfolio_returns
).cumprod()

sharpe_ratio = (
    portfolio_returns.mean()
    /
    portfolio_returns.std()
)

running_max = cumulative_returns.cummax()

drawdown = (
    cumulative_returns - running_max
) / running_max

max_drawdown = drawdown.min()

results.append({
    "Model": "Ridge Regression",
    "Return": cumulative_returns.iloc[-1] - 1,
    "Sharpe": sharpe_ratio,
    "Drawdown": max_drawdown,
    "R2": r2_score(y_test, predictions)
})

#now repeat for lasso regression

print("Running Ridge")

model = Lasso(alpha=0.0001)

model.fit(x_train, y_train)

predictions = model.predict(x_test)

prediction_df = close_long.loc[x_test.index].copy()

prediction_df["Month"] = (
    prediction_df["Date"]
    .dt.to_period("M")
)

prediction_df["Actual"] = y_test.values

prediction_df["Predicted"] = predictions

prediction_df["Rank"] = (
    prediction_df
    .groupby("Month")["Predicted"]
    .rank(ascending=False)
)

top_stocks = prediction_df[
    prediction_df["Rank"] <= 50
]

portfolio_returns = (
    top_stocks
    .groupby("Month")["Actual"]
    .mean()
)

cumulative_returns = (
    1 + portfolio_returns
).cumprod()

sharpe_ratio = (
    portfolio_returns.mean()
    /
    portfolio_returns.std()
)

running_max = cumulative_returns.cummax()

drawdown = (
    cumulative_returns - running_max
) / running_max

max_drawdown = drawdown.min()

results.append({
    "Model": "Lasso Regression",
    "Return": cumulative_returns.iloc[-1] - 1,
    "Sharpe": sharpe_ratio,
    "Drawdown": max_drawdown,
    "R2": r2_score(y_test, predictions)
})

#now repeat for gradient boosting

print("Running Lasso")

model = GradientBoostingRegressor(
    random_state=42
)

model.fit(x_train, y_train)

predictions = model.predict(x_test)

prediction_df = close_long.loc[x_test.index].copy()

prediction_df["Month"] = (
    prediction_df["Date"]
    .dt.to_period("M")
)

prediction_df["Actual"] = y_test.values

prediction_df["Predicted"] = predictions

prediction_df["Rank"] = (
    prediction_df
    .groupby("Month")["Predicted"]
    .rank(ascending=False)
)

top_stocks = prediction_df[
    prediction_df["Rank"] <= 50
]

portfolio_returns = (
    top_stocks
    .groupby("Month")["Actual"]
    .mean()
)

cumulative_returns = (
    1 + portfolio_returns
).cumprod()

sharpe_ratio = (
    portfolio_returns.mean()
    /
    portfolio_returns.std()
)

running_max = cumulative_returns.cummax()

drawdown = (
    cumulative_returns - running_max
) / running_max

max_drawdown = drawdown.min()

results.append({
    "Model": "Gradient Boosting",
    "Return": cumulative_returns.iloc[-1] - 1,
    "Sharpe": sharpe_ratio,
    "Drawdown": max_drawdown,
    "R2": r2_score(y_test, predictions)
})

#now repeat for MLP neural network

model = MLPRegressor(
    hidden_layer_sizes=(50,50),
    max_iter=500,
    random_state=42
)

model.fit(x_train, y_train)

predictions = model.predict(x_test)

prediction_df = close_long.loc[x_test.index].copy()

prediction_df["Month"] = (
    prediction_df["Date"]
    .dt.to_period("M")
)

prediction_df["Actual"] = y_test.values

prediction_df["Predicted"] = predictions

prediction_df["Rank"] = (
    prediction_df
    .groupby("Month")["Predicted"]
    .rank(ascending=False)
)

top_stocks = prediction_df[
    prediction_df["Rank"] <= 50
]

portfolio_returns = (
    top_stocks
    .groupby("Month")["Actual"]
    .mean()
)

cumulative_returns = (
    1 + portfolio_returns
).cumprod()

sharpe_ratio = (
    portfolio_returns.mean()
    /
    portfolio_returns.std()
)

running_max = cumulative_returns.cummax()

drawdown = (
    cumulative_returns - running_max
) / running_max

max_drawdown = drawdown.min()

results.append({
    "Model": "MLP Neural Network",
    "Return": cumulative_returns.iloc[-1] - 1,
    "Sharpe": sharpe_ratio,
    "Drawdown": max_drawdown,
    "R2": r2_score(y_test, predictions)
})

print("\n--- RESULTS ---")

results_df = pd.DataFrame(results)

print(results_df)

#prints the results of all the models in a dataframe format, showing the return, Sharpe ratio, drawdown, and R2 score for each model.

results_df = pd.DataFrame(results)

results_df = results_df.sort_values(
    "Sharpe",
    ascending=False
)

plt.figure(figsize=(10,6))

plt.bar(
    results_df["Model"],
    results_df["Sharpe"]
)

plt.title("Sharpe Ratio Comparison")

plt.ylabel("Sharpe Ratio")

plt.xticks(rotation=45)

plt.tight_layout()

plt.show()

#plots a bar chart comparing the Sharpe ratios of the different models, allowing for a visual comparison of their risk-adjusted performance.

plt.figure(figsize=(10,6))

plt.bar(
    results_df["Model"],
    results_df["Return"]
)

plt.title("Total Return Comparison")

plt.ylabel("Total Return")

plt.xticks(rotation=45)

plt.tight_layout()

plt.show()

#plots a bar chart comparing the total returns of the different models, allowing for a visual comparison of their overall performance.

plt.figure(figsize=(10,6))

plt.bar(
    results_df["Model"],
    abs(results_df["Drawdown"])
)

plt.title("Maximum Drawdown Comparison")

plt.ylabel("Drawdown (%)")

plt.xticks(rotation=45)

plt.tight_layout()

plt.show()

#plots a bar chart comparing the maximum drawdowns of the different models, allowing for a visual comparison of their risk exposure.

results_df.to_csv(
    "model_comparison_results.csv",
    index=False
)
#saves the results dataframe to a CSV file for further analysis 
