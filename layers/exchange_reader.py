import os
import datetime
import pandas as pd
from common.modules.exchange import Exchange
from common.modules.logger import logger
from common.paths import Paths
from common.refdata import date_formats
from layers.bitfinex_reader import BitfinexReader
from layers.bitmex_reader import BitmexReader


class ExchangeDataReader:
    # need a unified return type here
    map_reader = {
        Exchange.bitmex: BitmexReader,
        Exchange.bitfinex: BitfinexReader
    }

    @classmethod
    def load_trades(cls, exchange: Exchange, *args, **kwargs) -> pd.DataFrame:
        return cls.map_reader[exchange].load_trades(*args, **kwargs)

    @classmethod
    def load_quotes(cls, exchange: Exchange, *args, **kwargs) -> pd.DataFrame:
        return cls.map_reader[exchange].load_quotes(*args, **kwargs)
