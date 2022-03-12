import datetime
import pandas as pd

from common.modules.assets import Assets
from common.modules.logger import logger
from common.modules.enums import Exchange
from layers.bars.base_bar import BaseBar
from layers.exchange_reader import ExchangeDataReader


class Imbalance(BaseBar):
    information = 'imbalance'

    def __init__(self, exchange: Exchange, sym, start, end, unit, unit_size):
        super().__init__(exchange, sym, start, end, unit, **{'unit_size': unit_size})

    def resample(self) -> pd.DataFrame:
        df = ExchangeDataReader.load_trades(self.exchange, self.sym, self.start, self.end)
        self.add_measurable(df)
        df['imbalance_bar'] = df['measurable'].cumsum() // self.unit_size
        df['imbalance_size'] = (df['imbalance_bar'] - df['imbalance_bar'].shift(1)).fillna(0).astype(int)
        df = df[df['imbalance_size'] != 0]
        return df[['timestamp', 'imbalance_size']].groupby('timestamp').sum()

    def add_measurable(self, df):
        if self.unit == 'tick':
            df['measurable'] = df['side']
        elif self.unit == self.sym:
            df['measurable'] = df['side'] * df['size']
        elif self.unit == 'usd' and self.exchange == Exchange.bitmex:
            df['measurable'] = df['side'] * df['grossValue']
        elif self.unit == 'xbt':
            df['measurable'] = df['side'] * df['homeNotional']
        else:
            raise NotImplemented
        return df


if __name__ == '__main__':
    """
    bitfinex:   tick: 50, usd: , ethusd: 50
    """
    unit = 'usd'
    unit_size = 100_000

    bar = Imbalance(
        exchange=Exchange.bitfinex,
        sym=Assets.ethusd,
        start=datetime.datetime(2022, 2, 7),
        end=datetime.datetime(2022, 3, 2),
        unit=unit,
        unit_size=unit_size
    )
    df = bar.resample()
    logger.info(f'Resampled df of shape: {df.shape}')
    print(df.head())
    print(df.tail())
    bar.to_influx(df)
    logger.info('Done')
