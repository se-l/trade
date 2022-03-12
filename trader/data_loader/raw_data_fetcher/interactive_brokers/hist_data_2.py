from ib_insync import *


def main():
    ib = IB()
    ib.connect('127.0.0.1', 7497, clientId=1)

    contract = Forex('EURUSD')
    bars = ib.reqHistoricalData(
        contract, endDateTime='', durationStr='1 Y',
        barSizeSetting='1 min', whatToShow='MIDPOINT', useRTH=True)

    # convert to pandas dataframe:
    df = util.df(bars)
    print(df[['date', 'open', 'high', 'low', 'close']])


if __name__ == '__main__':
    main()
