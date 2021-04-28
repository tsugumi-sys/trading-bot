import yfinance as yf
import pandas as pd
import numpy as np
import datetime as dt
import tensorflow as tf
from tensorflow.keras.models import load_model
import requests

def Zero_One_Scale(df):
    df_scaled = (df - df.min()) / (df.max() - df.min())
    return df_scaled

def One_One_Scale(df):
    df_scaled = 2 * (df - df.min()) / (df.max() - df.min()) - 1
    return df_scaled

def Normalize(df):
    df_normalized = (df - df.mean(axis=0)) / df.std(axis=0)

def RSI(x):
    up, down = [i for i in x if i > 0], [i for i in x if i <= 0]
    if len(down) == 0:
        return 100
    elif len(up) == 0:
        return 0
    else:
        up_average = sum(up) / len(up)
        down_average = - sum(down) / len(down)
        return 100 * up_average / (up_average + down_average)
    
def SlowK(x):
    min_price = min(x)
    max_price = max(x)
    k = (x[-1] - min_price) / (max_price - min_price)
    return k


def make_prediction(df, model):
    df = df.copy()
    before_len = len(df)
    df['return'] = np.log(df['Adj Close'] / df['Adj Close'].shift(1))
    df['direction'] = np.where(df['return'] > 0, 1, -1)
    df['direction'] = df['direction'].shift(-1)
    df['return'] = One_One_Scale(df['return'])
    df['return'] = df['return'].shift(-1)
    # feature calculation
    # basic information
    df['price-change'] = df['Adj Close'] - df['Adj Close'].shift(1)
    df['price-change-percentage'] = df['Adj Close'] / df['Adj Close'].shift(1)
    df['volume'] = Zero_One_Scale(df['Volume'])
    df['amount'] = df['Adj Close'] * df['Volume']
    df['amount'] = Zero_One_Scale(df['amount'])
    # simple moving average
    df['sma10'] = df['Adj Close'].rolling(30).mean()
    df['sma10-FP'] = (df['sma10'] - df['sma10'].shift(1)) / df['sma10'].shift(1)
    df['sma10-FP'] = One_One_Scale(df['sma10-FP'])
    df['sma10'] = Zero_One_Scale(df['sma10'])
    # Moving Average Convergence Divergence
    df['macd'] = df['Adj Close'].rolling(12).mean() - df['Adj Close'].rolling(26).mean()
    df['macd-SG'] = df['macd'].rolling(9).mean()
    df['macd-histogram'] = df['macd'] - df['macd-SG']
    df['macd-histogram'] = np.where(df['macd-histogram'] > 0, 1, -1)
    df['macd-SG'] = np.where(df['macd-SG'] > 0, 1, -1)
    df['macd'] = np.where(df['macd'] > 0, 1, -1)
    # Commodity Channel Index in 24 days
    df['typical-price'] = (df['High'] + df['Low'] + df['Close']) / 3
    df['sma-cci'] = df['typical-price'].rolling(24).mean()
    df['mean-deviation'] = np.abs(df['typical-price'] - df['sma-cci'])
    df['mean-deviation'] = df['mean-deviation'].rolling(24).mean()
    df['cci'] = (df['typical-price'] - df['sma-cci']) / (0.015 * df['mean-deviation'])
    df['cci-SG'] = np.where(df['cci'] > 0, 1, -1)
    # MTM 10
    df['mtm10'] = df['Adj Close'] - df['Adj Close'].shift(10)
    df['mtm10'] = np.where(df['mtm10'] > 0, 1, -1)
    # Rate of Change in 10 days
    df['roc'] = (df['Adj Close'] - df['Adj Close'].shift(10)) / df['Adj Close'].shift(10)
    df['roc-SG'] = np.where(df['roc'] > 0, 1, -1)
    df['roc-FP'] = (df['roc'] - df['roc'].shift(1))
    df['roc-FP'] = One_One_Scale(df['roc-FP'])
    # Relative Strength Index in 5 days
    df['rsi'] = df['price-change'].rolling(5).apply(RSI) / 100
    df['rsi-FP'] = (df['rsi'] - df['rsi'].shift(1))
    df['rsi-FP'] = One_One_Scale(df['rsi-FP'])
    # Slow K and Slow D
    df['slow-k'] = df['Adj Close'].rolling(14).apply(SlowK)
    df['slow-d'] = df['slow-k'].rolling(14).mean()
    df['slow-k-FP'] = df['slow-k'] - df['slow-k'].shift(1)
    df['slow-d-FP'] = df['slow-d'] - df['slow-d'].shift(1)
    df['slow-k'] = Zero_One_Scale(df['slow-k'])
    df['slow-d'] = Zero_One_Scale(df['slow-d'])
    df['slow-k-FP'] = One_One_Scale(df['slow-k-FP'])
    df['slow-d-FP'] = One_One_Scale(df['slow-d-FP'])
    # ADOSC
    df['adosc'] = ((2 * df['Close'] - df['High'] - df['Low']) / (df['High'] - df['Low'])) * df['Volume']
    df['adosc'] = df['adosc'].cumsum()
    df['adosc-ema3'] = df['adosc'].ewm(span=3, adjust=False).mean()
    df['adosc-ema10'] = df['adosc'].ewm(span=10, adjust=False).mean()
    df['adosc-SG'] = np.where((df['adosc-ema3'] - df['adosc-ema10']) > 0, 1, -1)
    df['adosc'] = Zero_One_Scale(df['adosc'])
    # AR 26
    hp_op = (df['High'] - df['Open']).rolling(26).sum()
    op_lp = (df['Open'] - df['Low']).rolling(26).sum()
    df['ar26'] = hp_op / op_lp
    df['ar26'] = Zero_One_Scale(df['ar26'])
    # BR 26
    hp_cp = (df['High'] - df['Close']).rolling(26).sum()
    cp_lp = (df['Close'] - df['Low']).rolling(26).sum()
    df['br26'] = hp_cp / cp_lp
    df['br26'] = Zero_One_Scale(df['br26'])
    # VR 26
    
    # BIAS 20
    sma20 = df['Adj Close'].rolling(20).mean()
    df['bias20'] = (df['Adj Close'] - sma20) / sma20
    df['bias20'] = np.where(df['bias20'] > 0, 1, -1)
    
    
    df['price-change'] = One_One_Scale(df['price-change'])
    df['price-change-percentage'] = One_One_Scale(df['price-change-percentage'])
    # drop row contains NaN
    df.dropna(inplace=True)
    after_len = len(df)
    
    
    cols = ['return', 'price-change', 'price-change-percentage', 'volume', 'amount', 'sma10', 'sma10-FP',
        'macd', 'macd-SG', 'macd-histogram', 'cci-SG', 'mtm10', 'roc-SG', 'roc-FP', 'rsi', 'rsi-FP', 'slow-k', 'slow-d',
        'slow-k-FP', 'slow-d-FP', 'adosc', 'adosc-SG', 'ar26', 'br26', 'bias20']
    # Normalization
    X = df.copy()[cols]
    y = X.pop('return')
    float_cols = [i for i in X.columns if X[i].dtype == float]
    int_cols = [i for i in X.columns if X[i].dtype == int]
    X[float_cols] = (X[float_cols] - X[float_cols].mean(axis=0)) / X[float_cols].std(axis=0)
    
    # make prediction
    count = 0
    result = [np.nan for i in range(30 + (before_len - after_len))]
    for i in range(30,len(df)):
        data = X.iloc[count:i].values
        data = np.reshape(data, (1, 30, 24))
        y = model.predict(data)
        count += 1
        result.append(y[0][-1])
    
    
    return result

def lstm(symbol):
    start_day = dt.datetime.now() - dt.timedelta(days=120)
    df = yf.download(symbol, start=start_day, end=dt.datetime.now(), interval='1d')
    model = load_model('./model.h5')
    prediction = make_prediction(df, model)
    response = prediction[-1] if len(prediction) > 0 else None
    return str(response)