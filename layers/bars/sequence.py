import numpy as np
import datetime
import pandas as pd

from common.modules.assets import Assets
from common.modules.logger import logger
from common.modules.enums import Exchange
from layers.bars.base_bar import BaseBar
from layers.exchange_reader import ExchangeDataReader


class SequenceBar(BaseBar):
    information = 'sequence'

    def __init__(self, exchange, sym, start, end, unit, unit_size):
        super().__init__(exchange, sym, start, end, unit, **dict(unit_size=unit_size))

    def resample(self) -> pd.DataFrame:
        df = ExchangeDataReader.load_trades(self.exchange, self.sym, self.start, self.end)

        df_up = df[df['side'] == 1]
        self.add_measurable(df_up)
        df_up['measurable_cum_up'] = df_up['measurable'].cumsum()

        df_down = df[df['side'] == -1]
        self.add_measurable(df_down)
        df_down['measurable_cum_down'] = df_down['measurable'].cumsum()

        df = pd.concat((df_up, df_down), sort=False).sort_values('timestamp')
        df['measurable_cum_down'] = df['measurable_cum_down'].ffill().fillna(0).abs()
        df['measurable_cum_up'] = df['measurable_cum_up'].ffill().fillna(0)
        bars = []
        while True:
            iloc_down = np.argmax(df['measurable_cum_down'].values >= self.unit_size)
            iloc_up = np.argmax(df['measurable_cum_up'].values >= self.unit_size)
            if iloc_up == iloc_down == 0:
                break
            iloc_up = 10**99 if iloc_up == 0 else iloc_up
            iloc_down = 10**99 if iloc_down == 0 else iloc_down

            if iloc_down == iloc_up:
                direction = 0
            elif iloc_up < iloc_down:
                direction = 1
            elif iloc_down < iloc_up:
                direction = -1
            else:
                raise KeyError

            iloc = min((iloc_up, iloc_down))
            df['measurable_cum_down'] -= df.iloc[iloc]['measurable_cum_down']
            df['measurable_cum_up'] -= df.iloc[iloc]['measurable_cum_up']

            bars.append({'sequence_direction': direction, 'timestamp': df.iloc[iloc]['timestamp']})
            df = df.iloc[iloc:]

        return pd.DataFrame(bars).set_index('timestamp')[['sequence_direction']]

    def add_measurable(self, df):
        if self.unit == 'tick':
            df['measurable'] = df['side']
        elif self.unit == self.sym:
            df['measurable'] = df['side'] * df['size']
        elif self.unit == 'usd':
            df['measurable'] = df['side'] * df['grossValue']
        elif self.unit == 'xbt':
            df['measurable'] = df['side'] * df['homeNotional']
        else:
            raise NotImplemented
        return df


if __name__ == '__main__':
    """tick: 500, usd: 1B, ethusd: 1M"""
    unit = 'ethusd'
    unit_size = 100

    bar = SequenceBar(
        exchange=Exchange.bitfinex,
        sym=Assets.ethusd,
        start=datetime.datetime(2022, 2, 7),
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
