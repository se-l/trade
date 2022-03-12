import os
import pandas as pd
from datetime import date


def cp_universe(dt: date) -> list:
    path_root = r'C:\repos\quantconnect\lean_cloud\data\equity\usa\fundamental\coarse'
    path_target = r'C:\repos\quantconnect\lean_cloud\Equity_1\universes.py'
    dt_str = dt.strftime('%Y%m%d')
    df = pd.read_csv(os.path.join(path_root, dt_str + '.csv'), header=None)
    syms = df[1].to_list()
    with open(path_target, 'a') as f:
        f.write(f'''syms{dt_str} = {str(syms)}\n''')
    # df.iloc[:, [0, 1]].to_csv(os.path.join(path_target, dt.strftime('%Y%m%d.csv')), header=False, index=False)


if __name__ == '__main__':
    for dt in [
        date(2019, 6, 3),
        # date(2020, 12, 31)
               ]:
        cp_universe(dt)
