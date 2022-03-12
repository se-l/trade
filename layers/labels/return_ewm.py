import datetime
import pandas as pd
import numpy as np
import re

from common.modules.assets import Assets
from common.modules.logger import logger
from common.modules.enums import Exchange, Direction
from connector.influxdb.influxdb_wrapper import influx
from layers.exchange_reader import ExchangeDataReader


class LabelReturn:
    """Volatilty dependent symmetric threshold to determins the side of the bet. Volatility potentially determined by volumne
    instead of price series for better statistical properties.
    3 barrier:
        1) min long return with stop loss
        2) min short return with stop-loss
        3) ewm_span
    Volatility measrued as moving ewa with span equal ewm_span period for now.
    """

    def __init__(self, exchange: Exchange, sym, start, end, resampling_rule, ewm_span):
        self.exchange = exchange
        self.sym = sym
        self.start = start
        self.end = end
        self.resampling_rule = resampling_rule
        self.ewm_span = ewm_span

    def label(self) -> pd.DataFrame:
        df = ExchangeDataReader.load_trades(self.exchange, self.sym, self.start, self.end).set_index('timestamp')
        ps = df['price'].resample(rule=self.resampling_rule).last().ffill()
        # return
        ps = (ps - ps.shift(1)).fillna(0) / ps
        alpha = 2/(self.int_span + 1)
        weights = np.array([(1 - alpha) ** i for i in range(self.int_span)])

        forward_return = [(arr*weights + 1).prod() for arr in np.lib.stride_tricks.sliding_window_view(ps.values, window_shape=self.int_span)]
        return pd.DataFrame(forward_return, index=ps.index[:-self.int_span+1], columns=['forward_return_ewm'])

    @property
    def int_span(self) -> int:
        freq_resampled = int(re.search(r'(\d*)', self.resampling_rule).group(1))
        ewm_span = int(re.search(r'(\d*)', self.ewm_span).group(1))
        return ewm_span // freq_resampled

    def to_influx(self, df: pd.DataFrame):
        influx.write(
            record=df,
            data_frame_measurement_name='label',
            data_frame_tag_columns={
                'exchange': exchange.name,
                'asset': sym.name,
                'resampling_rule': self.resampling_rule,
                'ewm_span': self.ewm_span,
            },
        )


if __name__ == '__main__':
    exchange = Exchange.bitfinex
    sym = Assets.ethusd

    bar = LabelReturn(
        exchange=exchange,
        sym=sym,
        start=datetime.datetime(2022, 2, 7),
        end=datetime.datetime(2022, 3, 2),
        resampling_rule='1min',
        ewm_span='5min',  # letter must match resampling letter
    )
    df = bar.label()
    logger.info(f'Resampled df of shape: {df.shape}')
    print(df.head())
    print(df.tail())
    # should reference the underlying volatilty curve somewhere and add as parameter
    assert len(df.index.unique()) == len(df), 'Timestamp is not unique. Group By time first before uploading to influx.'
    bar.to_influx(df)
    logger.info('Done')