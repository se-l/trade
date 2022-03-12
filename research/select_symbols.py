from importlib import reload
from sklearn.metrics import mean_squared_error, r2_score
from collections import defaultdict
import helper
reload(helper)
from helper import r2, y_val, ret, sortino, sharpe, daily_delta_returns, first_nn

len(etfs)

qb = QuantBook()
step = 500
#for i in range(0, len(etfs), step):
for i in range(0, 500, step):
    symbols = [qb.AddEquity(ticker, Resolution.Daily).Symbol for ticker in etfs[i:min(i+step, len(etfs)-1)]]
    history = qb.History(qb.Securities.Keys, 5*365, Resolution.Daily)
    dfs.append(history['close'].unstack(level=0))

df = pd.concat(dfs, axis=1)

results = defaultdict(list)
mx = len(df.columns)
for day_factor in range(5, 4, -1):
    days = 252 * day_factor
    for i, c in enumerate(df.columns):
        if i % 1000 == 0:
            print(f'{days} - {100*(i/mx)}%')
        results[days].append({
            'col': c,
            'days': days,
            'days_recorded': first_nn(df[c].values),
            'r2': r2(df[c].iloc[-days:].values, y_val(df[c].iloc[-days:])),
            'return': ret(df[c].iloc[-days:]),
            'sharpe': sharpe(df[c].iloc[-days:].values),
            'sortino': sortino(df[c].iloc[-days:].values)
        })
r = pd.concat([pd.DataFrame(df) for df in results.values()])
r = r[r['col'].isin([c for c in r['col'] if not any((c.startswith(e) for e in leveraged))])]
r['sym'] = r['col'].str.split(' ').apply(lambda x: x[0])
r['days_recorded'] = r['days_recorded'].mask(r['days_recorded']==0, r['days'].max())
r = r.sort_values(['sym', 'days', 'days_recorded'], ascending=False).reset_index(drop=True)
r = r[~r['sym'].duplicated()].reset_index(drop=True)
r[r['return']>0][['return', 'sharpe']].plot(x='return', y='sharpe', kind='scatter', figsize=(15,10))
r = r[(r['days_recorded'] > 252*3) &
      (r['return'].notna()) &
      (r['sharpe'].notna()) &
      (r['r2'] > 0.6) &
      (r['return'] > 1.5)
     ].sort_values('return', ascending=False)
r[r['return']>0][['r2', 'return']].plot(x='r2', y='return', kind='scatter', figsize=(15,10))
syms = [qb.AddEquity(ticker, Resolution.Daily).Symbol for ticker in r['col'].values]
history = qb.History(syms, 5*252, Resolution.Daily).close.unstack(level=0)
correlation_matrix = history.corr()
thresh = 0.99
sym_lst = r.sort_values('return')['col'].values
ix_remaining = []
remaining = []
for ix, row in r.iterrows():
    corrs = correlation_matrix[row['col']][correlation_matrix[row['col']] > thresh].index.values
#     print(corrs)
    if not any((s in remaining for s in corrs)):
        remaining.append(row['col'])
        ix_remaining.append(ix)
remaining = r.loc[ix_remaining]
print('Kicked out...')
r.loc[set(r.index).difference(set(ix_remaining))]
print('Kicked out...')
r.loc[set(r.index).difference(set(ix_remaining))]
remaining['sym'].values

df[df.index.isin(r['col'].values)]

from scipy import stats
import numpy as np
import pandas as pd
df = pd.DataFrame(pd.DataFrame([['a']*100, list(range(100)), list(range(100))]).transpose().values, columns=['sym', 'r2', 'return'])
df['r2_zscore'], df['return_zscore'] = stats.zscore(df['r2']), stats.zscore(df['return'])
for p in range(99, 0, -1):
    s1 = df['return_zscore'] > np.percentile(df['return_zscore'], p)
    s2 = df['r2_zscore'] > np.percentile(df['r2_zscore'], p)
    if len(set(df['sym'][s1]).intersection(set(df['sym'][s2]))) > 10 or p == 1:
        print(p)
