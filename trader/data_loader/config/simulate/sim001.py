import datetime

from common.utils.util_func import SeriesTickType
from trader.train.config.reinforced import ParamsBase
from common.modules import exchanges


class Params(ParamsBase):
    data_start = ts_start = datetime.datetime(2019, 8, 1)
    data_end = ts_end = datetime.datetime(2019, 8, 4, 23, 59, 59)
    exchange = exchanges.bitmex
    ex = None
    wave_params = [
        {
            'wavelength': 1800,
            'zero_line': 250,
            'amplitude': 25,
        },
        {
            'wavelength': 60,
            'zero_line': 0,
            'amplitude': 3,
        }
    ]
    asset = 'sim001'
    series_tick_type = SeriesTickType('ts', 1, 'second')
    bid_ask_spread = 0.05
