import pandas as pd
import numpy as np

EPS=1e-12

# --- 截面算子 ---
def win(window) -> int:
    
    return max(int(round(float(window))), 1)


def to_df(value, like: pd.DataFrame) -> pd.DataFrame:
    if isinstance(value, pd.DataFrame):
        return value
    if isinstance(value, pd.Series):
        return pd.DataFrame({col: value for col in like.columns}, index=like.index)
    return pd.DataFrame(value, index=like.index, columns=like.columns)


def sanitize(df):
    if isinstance(df, (pd.DataFrame, pd.Series)):
        return df.replace([np.inf, -np.inf], np.nan)
    return df

def rank(df):
    return sanitize(df).rank(axis=1, pct=True)


def delay(df, period=1):
    return df.shift(win(period))


def delta(df, period=1):
    return df.diff(win(period))


def ts_sum(df, window=10):
    return df.rolling(win(window)).sum()


def ts_mean(df, window=10):
    return df.rolling(win(window)).mean()


def stddev(df, window=10):
    return df.rolling(win(window)).std()


def correlation(x, y, window=10):
    return sanitize(x).rolling(win(window)).corr(sanitize(y))


def covariance(x, y, window=10):
    return sanitize(x).rolling(win(window)).cov(sanitize(y))


def ts_rank(df, window=10):
    w = win(window)
    return df.rolling(w).apply(lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False)


def product(df, window=10):
    return df.rolling(win(window)).apply(np.prod, raw=True)


def ts_min(df, window=10):
    return df.rolling(win(window)).min()


def ts_max(df, window=10):
    return df.rolling(win(window)).max()


def ts_argmax(df, window=10):
    return df.rolling(win(window)).apply(np.argmax, raw=True) + 1


def ts_argmin(df, window=10):
    return df.rolling(win(window)).apply(np.argmin, raw=True) + 1


def signed_power(x, a):
    return np.sign(x) * np.power(np.abs(x), a)


def scale(df, k=1):
    denom = np.abs(df).sum(axis=1).replace(0, np.nan)
    return df.mul(k).div(denom, axis=0)


def decay_linear(df, period=10):
    w = win(period)
    weights = np.arange(1, w + 1, dtype=float)
    weights /= weights.sum()
    return df.rolling(w).apply(lambda x: np.dot(x, weights), raw=True)


def safe_div(x, y, eps=EPS):
    return x / y.where(np.abs(y) > eps, np.nan)


def max_df(x, y):
    return pd.DataFrame(np.maximum(x.values, y.values), index=x.index, columns=x.columns)


def min_df(x, y):
    return pd.DataFrame(np.minimum(x.values, y.values), index=x.index, columns=x.columns)


def bool_to_float(df):
    
    return df.astype(float)


