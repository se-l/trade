import datetime

from common.modules.assets import Assets
from common.modules.logger import logger
from common.modules.enums import Exchange
from layers.bars.base_bar import BaseBar
from layers.exchange_reader import ExchangeDataReader


class PriceBars(BaseBar):
    information = 'price'

    def __init__(self, exchange, sym, start, end, unit, unit_size):
        self.unit_size = unit_size
        super().__init__(exchange, sym, start, end, unit, unit_size=unit_size)

    def resample(self):
        """Total volume across buy sell or single side volume when side is set"""
        df = ExchangeDataReader.load_trades(self.exchange, self.sym, self.start, self.end)
        if self.unit_size == 1:
            df['price'] = df['price'].round(0)
        df['price_changed'] = (df['price'] - df['price'].shift(1)).fillna(0)
        df = df[df['price_changed'] != 0]
        return df[['timestamp', 'price']].groupby('timestamp').last()

    def add_measurable(self, df): pass


if __name__ == '__main__':
    unit = 'ethusd'
    unit_size = 1

    bar = PriceBars(
        exchange=Exchange.bitfinex,
        sym=Assets.ethusd,
        start=datetime.datetime(2022, 2, 6),
        end=datetime.datetime(2022, 3, 2),
        unit=unit,
        unit_size=unit_size,
    )
    df = bar.resample()
    logger.info(f'Resampled df of shape: {df.shape}')
    print(df.head())
    print(df.tail())
    bar.to_influx(df)
    logger.info('Done')

