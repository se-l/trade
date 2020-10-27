import pandas as pd
import numpy as np
import datetime as dt
import os
import zipfile
from common.globals import OHLC, qc_bitfinex_crypto
from common.utils.util_func import precision_and_scale

range = [(9, 1), (9, 22)]  # doesnt work acorss month
req_start = dt.datetime(2018, range[0][0], range[0][1])
req_end = dt.datetime(2018, range[1][0], range[1][1])
symbol = ['btcusd', 'ethusd', 'ltcusd', 'xrpusd', 'xmrusd', 'neousd'][1]

# Bitfinex notes. When e.g. July 30 is selected in the calendar, the received file contains trade data
# from July 29. Hence uploading up to today's date. The format for QC is different. QC file of July 30 contains trade
# data from julty 30

def resampleIndex(df, timePeriod=None, aggCol=['volume'], priceCol=['price'], **kwargs):
    """
    """
    if aggCol is not None:
        vol = df[aggCol].resample(rule=timePeriod).sum()
    if priceCol is not None:
        price = pd.concat([
            df[priceCol].resample(rule=timePeriod).first(),
            df[priceCol].resample(rule=timePeriod).max(),
            df[priceCol].resample(rule=timePeriod).min(),
            df[priceCol].resample(rule=timePeriod).last(),
            # df[priceCol].resample(rule=timePeriod).mean(),
            df[priceCol].resample(rule=timePeriod).count()
        ],
        axis=1
        )
        price.columns = ['open','high','low','close','count']
    if aggCol is not None:
        return pd.concat([price, vol], axis=1)
    else:
        return pd.concat([price], axis=1)

def load_ex_bitfinex_trades(res, res_p=None):
    if res == 'minute':
        res_p = '1T'
    elif res == 'second':
        res_p = '1S'
    #load multiple files
    folder = r'C:\Users\seb\Desktop\bitfin_data\{}'.format(symbol)
    df = pd.DataFrame()
    for root, dirs, filenames in os.walk(folder):
        for file in filenames:
            st = dt.datetime.strptime(file[:10], '%Y-%m-%d')
            end = dt.datetime.strptime(file[11:21], '%Y-%m-%d')
            if st >= req_start - dt.timedelta(days=1) and end <= req_end + dt.timedelta(days=1):
                if len(df) == 0:
                    df = pd.read_csv(os.path.join(folder, file))
                else:
                    df = df.append(pd.read_csv(os.path.join(folder, file)))
        break

    df.index = pd.to_datetime(df['Timestamp'], unit='s')
    df = df.drop(['Timestamp', 'TradeId', 'Type'], axis=1)
    # if t_res == 's':
    #     res_p = '1S'
    # else:
    #     res_p = '1T'
    df = resampleIndex(df, timePeriod=res_p, aggCol=['Amount'], priceCol=['Price'])
    df = df.drop(['count'], axis=1)
    df = df.dropna(axis=0, how='any')
    # for c in OHLC:
    #     df[c] = np.round(df[c], 2)
    return df

def pick_idx(df, day, month, res):
    df_day = df.iloc[np.where((df.index.day == day) & (df.index.month == month))[0], :]
    df_day.index = (df_day.index - dt.datetime(2018, month, day)).total_seconds()
    df_day.index = pd.to_numeric(df_day.index * 1000, downcast='integer')
    return df_day

def fn_t(day, month):
    return os.path.join(dir, '2018{}{}_{}_{}_trade.csv'.format(month, day, symbol, res))
# fn_t = lambda x: os.path.join(dir, '201801{}_{}_{}_trade.csv'.format(x, symbol, res))
def fn_q(day, month):
    return os.path.join(dir, '2018{}{}_{}_{}_quote.csv'.format(day, month, symbol, res))
# fn_q = lambda x: os.path.join(dir, '201801{}_{}_{}_quote.csv'.format(x, symbol, res))
def zip_t(day, month):
    return os.path.join(dir, '2018{}{}_trade.zip'.format(month, day))
# zip_t = lambda x: os.path.join(dir, '201801{}_trade.zip'.format(x))
def zip_q(day, month):
    return os.path.join(dir, '2018{}{}_quote.zip'.format(month, day))
# zip_q = lambda x: os.path.join(dir, '201801{}_quote.zip'.format(x))

def df_day_trade(df, day, month):
    df.to_csv(fn_t(day=day, month=month), header=False)
    zipItAway(fn_t(day=day, month=month), zip_t(day=day, month=month))

def df_day_quote(df, day, month, spread):
    # convert from trades into quote. assuming a spread of 2 * 1/10**scale
    df['Amount'] = 1000
    ask_ohlc = ['ask' + c for c in OHLC]
    df[OHLC] = df[OHLC] - spread
    df[ask_ohlc] = df[OHLC] + spread
    df['ask_amount'] = 1000
    # save and zip
    df.to_csv(fn_q(day=day, month=month), header=False)
    zipItAway(fn_q(day=day, month=month), zip_q(day=day, month=month))

def zipItAway(fn, zipName):
    with zipfile.ZipFile(zipName, 'w', compression=zipfile.ZIP_DEFLATED) as myzip:
        myzip.write(fn, os.path.basename(fn))
    os.remove(fn)

if __name__ == '__main__':
    for res in ['second', 'minute']:
        dir = os.path.join(qc_bitfinex_crypto, r'{}\{}'.format(res, symbol))
        month = range[0][0]
        df = load_ex_bitfinex_trades(res)
        precision, scale = precision_and_scale(df.iloc[len(df)//2, df.columns.get_loc('close')])
        spread = 1 / 10 ** scale
        print('Applying a spread of 2 * {}'.format(spread))
        for i in np.arange(range[0][1], range[1][1]+1):
            dfd = pick_idx(df, day=i, month=month, res=res)
            df_day_trade(dfd, day="%02d" % (i,), month="%02d" % (month,))
            df_day_quote(dfd, day="%02d" % (i,), month="%02d" % (month,), spread=spread)