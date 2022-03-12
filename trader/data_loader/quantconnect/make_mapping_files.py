import datetime
import os
import io
import pandas as pd

from zipfile import ZipFile

path_target = r'C:\repos\quantconnect\Lean\Data\equity\usa\map_files'
path_equity = r'C:\repos\quantconnect\Lean\Data\equity\usa\daily'

if __name__ == '__main__':
    prim_exchange = 'Q'
    for _, dir_, fns in os.walk(path_equity):
        for fn in fns:
            symbol = fn.split('.')[0]
            with ZipFile(os.path.join(_, fn), 'r') as zip:
                buffer = io.StringIO(zip.read(zip.filelist[0]).decode())
                df = pd.read_csv(buffer)
            try:
                dt_start = pd.to_datetime(df.iloc[0].index[0].split(' ')[0]).date()
                dt_end = pd.to_datetime(df.iloc[-1][0].split(' ')[0]).date()
            except Exception as e:
                continue
            if dt_end.year >= 2021:
                dt_end = datetime.date(2050, 12, 31)
            new_dct = [
                {'date': dt_start.strftime('%Y%m%d'), 'sym': symbol, 'exchange': 'Q'},
                {'date': dt_end.strftime('%Y%m%d'), 'sym': symbol, 'exchange': 'Q'}
            ]
            save_path = os.path.join(path_target, f'{symbol}.csv')
            if not os.path.exists(save_path):
                pd.DataFrame(new_dct).to_csv(save_path, index=False, header=False)
