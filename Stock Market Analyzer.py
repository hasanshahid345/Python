import pandas as pd
import numpy as np
from alpha_vantage.timeseries import TimeSeries
import time
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

#FANNI'S PART

api_key = 'PUJWZX4VJ485DED6'
api_key = 'OKSMUM5PD257WPHI'
ts = TimeSeries(key=api_key, output_format='pandas')

# Choosen market is the technology market
# Apple, Microsoft, Google, Amazon, NVDIA, Tesla, Metas

#chosen_stocks = ['AAPL', 'MSFT'] #Example one to give more running chance
chosen_stocks = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'TSLA', 'META']
stock_market_data = []

for stock in chosen_stocks:
    data, meta_data = ts.get_daily(symbol=stock, outputsize='compact')
    data['Stock'] = stock  #Stock column to diffentiate different stocks
    stock_market_data.append(data)

final_data = pd.concat(stock_market_data)
final_data.reset_index(inplace=True)
final_data.columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'Stock']

#Explaination: the final data is a dataframe (pandas) datatype with the columns
#Date: exact day in YYYY-MM-DD format
#Open: opening price
#High: highest price on that day
#Low: lowest price on that day
#Close: closing price
#Volume: total number of shares bought and sold
#Stock: stock name

print("Stocks included in data:", final_data['Stock'].unique())

#Indicator Calculations - Hasan

def sma(prices, window=5):
    sma = []
    for i in range(len(prices)):
        if i < window - 1:
            sma.append(np.nan)
        else:
            avg = sum(prices[i - window + 1:i + 1]) / window
            sma.append(avg)
    return sma

def ema(prices, span):
    ema = []
    k = 2 / (span + 1)
    for i in range(len(prices)):
        if i == 0:
            ema.append(prices[i])
        else:
            current_ema = (prices[i] * k) + (ema[-1] * (1 - k))
            ema.append(current_ema)
    return ema

def rsi(prices, window=14):
    rsi = []
    gains = []
    losses = []

    for i in range(1, len(prices)):
        change = prices[i] - prices[i - 1]
        gains.append(max(change, 0))
        losses.append(abs(min(change, 0)))

    for i in range(len(prices)):
        if i < window:
            rsi.append(np.nan)
        else:
            avg_gain = sum(gains[i - window:i]) / window
            avg_loss = sum(losses[i - window:i]) / window
            if avg_loss == 0:
                rsi_value = 100
            else:
                rs = avg_gain / avg_loss
                rsi_value = 100 - (100 / (1 + rs))
            rsi.append(rsi_value)

    rsi = [np.nan] * (len(prices) - len(rsi)) + rsi
    return rsi

#TypicalPrice

def tp(high, low, close):
    return [(h + l + c) / 3 for h, l, c in zip(high, low, close)]

def vwap(tp, volume):
    vwap = []
    cumulative_price_volume = 0
    cumulative_volume = 0
    for i in range(len(tp)):
        cumulative_price_volume += tp[i] * volume[i]
        cumulative_volume += volume[i]
        if cumulative_volume == 0:
            vwap.append(np.nan)
        else:
            vwap.append(cumulative_price_volume / cumulative_volume)
    return vwap

def calculate_indicators(df):
    df = df.reset_index(drop=True)

    close_prices = df['Close'].tolist()
    high = df['High'].tolist()
    low = df['Low'].tolist()
    volume = df['Volume'].tolist()

    df['SMA_5'] = sma(close_prices, window=5)
    df['EMA_5'] = ema(close_prices, span=5)
    df['EMA_12_manual'] = ema(close_prices, span=12)
    df['EMA_26_manual'] = ema(close_prices, span=26)
    df['MACD'] = pd.Series(df['EMA_12_manual']) - pd.Series(df['EMA_26_manual'])
    df['RSI'] = rsi(close_prices, window=14)
    df['TP'] = tp(high, low, close_prices)
    df['VWAP'] = vwap(df['TP'], volume)

    return df



#Apply Indicators and Correlation Analysis - Ghazal

processed_data = final_data.groupby('Stock', group_keys=False).apply(calculate_indicators).reset_index(drop=True)

correlation_data = processed_data[['Close', 'SMA_5', 'EMA_5', 'MACD', 'RSI', 'VWAP']].corr()

print("\n📊 Correlation Matrix:\n")
print(correlation_data)

print("\n📄 Sample of Data with Indicators:\n")
print(processed_data.head())

unique_stocks = processed_data['Stock'].unique()

for stock in unique_stocks:
    print(f"\n🔍 Modeling for {stock}")

    stock_df = processed_data[processed_data['Stock'] == stock].dropna(subset=['SMA_5', 'EMA_5', 'MACD', 'RSI', 'VWAP', 'Close'])

    if len(stock_df) < 30:
        print(f"⚠️ Not enough data to model {stock}. Skipping.")
        continue

    features = ['SMA_5', 'EMA_5', 'MACD', 'RSI', 'VWAP']
    X = stock_df[features]
    y = stock_df['Close']

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, shuffle=False)

    model = LinearRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f"Mean Squared Error: {mse:.2f}")
    print(f"R-squared Score: {r2:.2f}")

    plt.figure(figsize=(12,6))
    plt.plot(stock_df['Date'].iloc[-len(y_test):], y_test.values, label='Actual Close', color='blue')
    plt.plot(stock_df['Date'].iloc[-len(y_test):], y_pred, label='Predicted Close', color='red', linestyle='--')
    plt.title(f"{stock} - Linear Regression: Close Price Prediction")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.xticks(rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.show()

    print("Feature Coefficients (Impact on Prediction):")
    for feature, coef in zip(features, model.coef_):
        print(f"{feature}: {coef:.2f}")

    most_important = max(zip(features, model.coef_), key=lambda x: abs(x[1]))
    print(f"💡 Most Influential Indicator: {most_important[0]} ({most_important[1]:.2f})")