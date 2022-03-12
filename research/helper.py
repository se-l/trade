import numpy as np


def first_nn(x):
    return np.argmax(~np.isnan(x))


def daily_delta_returns(x: np.array):
    x = x[first_nn(x):]
    x2 = x / x[-1]
    return np.concatenate(([0], np.subtract(x2[1:], x2[:-1])))


def daily_delta_neg_returns(x: np.array):
    x = daily_delta_returns(x)
    return np.where(x > 0, x, 0)


def sharpe(x):
    returns = daily_delta_returns(x)
    return returns.mean() / returns.std() * np.sqrt(len(returns))


def sortino(x):
    returns = daily_delta_neg_returns(x)
    return returns.mean() / returns.std() * np.sqrt(len(returns))


def r2(x, y):
    correlation_matrix = np.corrcoef(x, y)
    correlation_xy = correlation_matrix[0, 1]
    r_squared = correlation_xy ** 2
    return r_squared


def y_val(df):
    a, b = df.iloc[0], df.iloc[-1]
    return np.array([a + i * (b - a) / len(df) for i in range(len(df))])


def ret(x):
    x = x[first_nn(x):]
    return (x[-1] - x[0]) / x[0]