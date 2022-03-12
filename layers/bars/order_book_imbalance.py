import datetime
import pandas as pd

from common.modules.assets import Assets
from common.modules.logger import logger
from common.modules.enums import Exchange

from layers.bars.base_bar import BaseBar
from layers.exchange_reader import ExchangeDataReader


class Imbalance(BaseBar):
    """EWA of order book sizes, its imbalance in ethusd. Emit event whenever delta imbalance is greater than thresh. Is before price change isnt it, but not all levels..."""
    information = 'imbalance'

    def __init__(self, exchange: Exchange, sym, start, end, unit, **kwargs):
        super().__init__(exchange, sym, start, end, unit, **kwargs)

    def resample(self) -> pd.DataFrame:
        df = ExchangeDataReader.load_quotes(self.exchange, self.sym, self.start, self.end)
        df = self.add_measurable(df)
        df['imbalance_cumsum'] = df['measurable'].cumsum() // self.unit_size
        # Changes
        df2 = df[(df['imbalance_size'] - df['imbalance_size'].shift(1)).fillna(0).astype(int) != 0]
        return df[['timestamp', 'imbalance_size']].groupby('timestamp').sum()

    def add_measurable(self, df):
        if self.unit == 'tick':
            df['measurable'] = df['side']
        elif self.unit == self.sym:
            df['measurable'] = df['askSize'] - df['bidSize']
            df['mid_price'] = df['bidPrice'] + (df['askPrice'] - df['bidPrice']) / 2
        elif self.unit == 'usd':
            df['measurable'] = df['side'] * df['grossValue']
        elif self.unit == 'xbt':
            df['measurable'] = df['side'] * df['homeNotional']
        else:
            raise NotImplemented
        return df


if __name__ == '__main__':
    """tick: 500, usd: 100M, ethusd: 500k"""
    unit = 'ethusd'
    unit_size = 500_000

    bar = Imbalance(
        exchange=Exchange.bitmex,
        sym=Assets.ethusd,
        start=datetime.datetime(2022, 2, 6),
        end=datetime.datetime(2022, 3, 2),
        unit=unit,
        unit_size=unit_size
    )
    df = bar.resample()
    logger.info(f'Resampled df of shape: {df.shape}')
    print(df.head())
    print(df.tail())
    # bar.to_influx(df)
    logger.info('Done')
