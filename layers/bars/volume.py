import datetime

from common.modules.assets import Assets
from common.modules.logger import logger
from common.modules.enums import Exchange
from layers.bars.base_bar import BaseBar
from layers.exchange_reader import ExchangeDataReader


class TradeVolumeBars(BaseBar):
    information = 'volume'

    def __init__(self, exchange, sym, start, end, unit, side=None, **kwargs):
        self.side = side
        super().__init__(exchange, sym, start, end, unit, side=side, **kwargs)

    def resample(self):
        """Total volume across buy sell or single side volume when side is set"""
        df = ExchangeDataReader.load_trades(self.exchange, self.sym, self.start, self.end)
        self.add_measurable(df)
        if self.side:
            df = df[df['side'] == self.side]
            df['measurable'] = df['measurable'].abs()
        df['measurable_cumsum'] = df['measurable'].cumsum() // self.unit_size
        df['volume_size'] = (df['measurable_cumsum'] - df['measurable_cumsum'].shift(1)).fillna(0).astype(int)
        df = df[df['volume_size'] != 0]
        return df[['timestamp', 'volume_size']].groupby('timestamp').sum()

    def add_measurable(self, df):
        if self.unit == 'tick':
            df['measurable'] = df['side']
        elif self.unit == self.sym:
            df['measurable'] = df['side'] * df['size']
        elif self.unit == 'usd':
            df['measurable'] = df['side'] * df['size'] * df['price']
        # elif self.unit == 'xbt':
        #     df['measurable'] = df['side'] * df['homeNotional']
        else:
            raise NotImplemented
        return df


if __name__ == '__main__':
    """
    bitmex:     tick: 500, usd: 100M, ethusd: 100k
    bitfinex:   tick: , usd: 100_000, ethusd: 
    """
    unit = 'usd'
    unit_size = 100_000

    for side in [None, -1, 1]:
        bar = TradeVolumeBars(
            exchange=Exchange.bitfinex,
            sym=Assets.ethusd,
            start=datetime.datetime(2022, 2, 6),
            end=datetime.datetime(2022, 3, 2),
            unit=unit,
            unit_size=unit_size,
            side=side,
        )
        df = bar.resample()
        logger.info(f'Resampled df of shape: {df.shape}')
        print(df.head())
        print(df.tail())
        bar.to_influx(df)
        logger.info('Done')

