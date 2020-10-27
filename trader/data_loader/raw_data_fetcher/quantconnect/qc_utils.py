import os, re
import datetime as dt
import pandas as pd
import numpy as np
from utils.utilFunc import date_day_range
import datetime
from globals import OHLC, qc_bitfinex_crypto
from decimal import Decimal

class QcUtils():
    def __init__(self):
        pass

    @staticmethod
    def select_series_type(df, series):
        if series == 'trade':
            return df
        elif series in ['ask', 'ask_size']:
            return df.iloc[:, [0] + list(range(6, 11))]
        elif series in ['bid', 'bid_size']:
            return df.iloc[:, list(range(0, 6))]
        else:
            raise ('Specify trade or which quote side')

    @staticmethod
    def merge_qc_sec(dfd, date, resolution='1S', series=None):
        dfd.columns = ['ts'] + OHLC + ['volume']
        dfd['ts'] = dfd['ts'] // 1000 + (date - dt.datetime(1970, 1, 1)).total_seconds()
        dfd.ts = dfd['ts'].apply(lambda x: dt.datetime.utcfromtimestamp(x))
        dfd.index = dfd.ts
        dfd = dfd.drop('ts', axis=1)
        if series in ['trade', None]:
            dfd = dfd.resample(resolution).agg({'open': 'first',
                                            'high': 'max',
                                            'low': 'min',
                                            'close': 'last',
                                            'volume': 'sum'})
        elif series in ['bid', 'ask']:
            dfd = dfd.resample(resolution).agg({'open': 'first',
                                                'high': 'max',
                                                'low': 'min',
                                                'close': 'last',
                                                'volume': 'last'})
        elif series in ['ask_size', 'bid_size']:
            dfd = dfd['volume'].resample(resolution).agg(['first', 'max', 'min', 'last', 'mean', 'count'])
        return dfd

    @staticmethod
    def load_qc_bitfinex(folder, day_range, month_range=[1, 12], res='1S'):
        first = True
        for root, dirs, filenames in os.walk(folder):
            for file in filenames:
                if re.search('trade', file) is not None:
                    month = int(file[4:6])
                    day = int(file[6:8])
                    if month in np.arange(month_range[0], month_range[1] + 1):
                        if day in np.arange(day_range[0], day_range[1] + 1):
                            if first:
                                df = pd.read_csv(os.path.join(folder, file), header=None)
                                df = QcUtils.merge_qc_sec(df, day, month, resolution=res)
                                first = False
                            else:
                                dfd = pd.read_csv(os.path.join(folder, file), header=None)
                                dfd = QcUtils.merge_qc_sec(dfd, day, month, resolution=res)
                                df = df.append(dfd)
        return df.sort_index()

    @staticmethod
    def load_qc_date(folder, start, end, series='trade', res='1S'):
        # date_tupels = list(date_day_range(start, end))
        if series in ['trade', None]:
            qt_snippet = 'trade'
        else:
            qt_snippet = 'quote'
        first = True
        for root, dirs, filenames in os.walk(folder):
            for file in filenames:
                if qt_snippet in file:
                    date = datetime.datetime.strptime(file[0:8], '%Y%m%d')
                    if start <= date <= end:
                        if first:
                            df = pd.read_csv(os.path.join(folder, file), header=None)
                                             # converters={1: Decimal, 2: Decimal, 3: Decimal, 4: Decimal})
                            df = QcUtils.merge_qc_sec(
                                QcUtils.select_series_type(df, series),
                                date, resolution=res, series=series)
                            first = False
                        else:
                            dfd = pd.read_csv(os.path.join(folder, file), header=None)
                                              # converters={1: Decimal, 2: Decimal, 3: Decimal, 4: Decimal})
                            dfd = QcUtils.merge_qc_sec(
                                QcUtils.select_series_type(dfd, series),
                                date, resolution=res, series=series)
                            df = df.append(dfd)
        # this resample missing seconds in between loaded days +/- 10 sec around midnight
        try:
            if series in ['trade', None]:
                df = df.resample(res).agg({'open': 'first',
                                   'high': 'max',
                                   'low': 'min',
                                   'close': 'last',
                                   'volume': 'sum'})
            elif series in ['ask', 'bid']:
                df = df.resample(res).agg({'open': 'first',
                                           'high': 'max',
                                           'low': 'min',
                                           'close': 'last',
                                           'volume': 'last'})
            elif series in ['ask_size', 'bid_size']:
                df = df['volume'].resample(res).agg(['first', 'max', 'min', 'last', 'mean', 'count'])
            else:
                print('series type is unknown. returning None ohlc.')
                return None
            return df.sort_index()
        except NameError:
            print('df is not defined. {} No data could be loaded since start date: {}'.format(folder, start))
            return None

    @staticmethod
    def inflate_low_value_assets(df, sym):
        if 'xrp' in sym:
            for c in OHLC:
                df[c] = df[c] * 10**6
        return df
