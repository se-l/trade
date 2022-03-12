import numpy as np

x = [1, 2, 3, 4]
y = [1, 2.1, 2.9, 4]


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
    return ret(x) / np.std(daily_delta_returns(x))


def sortino(x):
    return ret(x) / np.std(daily_delta_neg_returns(x))


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
    return (x[-1] - x[0]) / x[-1]


def r2(x, y):
    correlation_matrix = np.corrcoef(x, y)
    correlation_xy = correlation_matrix[0, 1]
    r_squared = correlation_xy ** 2
    return r_squared

sharpe(np.array(x))

# r2(x, y)
# r2(daily_delta_returns(x), daily_delta_returns(y))
#
# r2(np.sin([i * np.pi / 8 for i in range(17)]), list(range(17)))
# r2(daily_delta_returns(np.sin([i * np.pi / 8 for i in range(17)])), list(range(17)))
# r2(daily_delta_returns(np.sin([i * np.pi / 8 for i in range(17)])), daily_delta_returns(np.sin([i * np.pi / 8 for i in range(17)])))
#
# r2(daily_delta_returns(np.sin([i * np.pi / 8 for i in range(21)])), [1 / 21 for i in range(21)])
# r2(np.sin([i * np.pi / 8 for i in range(21)]), np.cumsum([1 / 21 for i in range(21)]))
#
# np.std(daily_delta_returns(np.sin([i * np.pi / 8 for i in range(21)])))
# np.std(daily_delta_neg_returns(np.sin([i * np.pi / 8 for i in range(21)])))
