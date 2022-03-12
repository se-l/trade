import os
import json
from collections import defaultdict
from pprint import pprint

root = r'C:\repos\quantconnect\lean_cloud\Equity_1\optimizations'
p_opts = '2021-07-27_23-22-51'
summary = defaultdict(dict)
for _, dirs, fns in os.walk(os.path.join(root, p_opts), False):
    for fn in fns:
        if fn.endswith('.json') and not fn.endswith('events.json') and not fn.endswith('results.json'):
            with open(os.path.join(_, fn), 'r') as f:
                dd = json.load(f)
                try:
                    summary[fn] = {**dd['Statistics'], **dd['RuntimeStatistics']}
                except:
                    continue
interest = ['Sharpe Ratio', 'Total Trades', 'Drawdown', 'Compounding Annual Return', 'Probabilistic Sharpe Ratio']
pprint({k: {i: v.get(i) for i in interest} for k, v in summary.items()})
# pprint({k: {i: v.get(i) for i in interest} for k, v in summary.items() if float(v.get('Total Trades', '99')) < 1800})

