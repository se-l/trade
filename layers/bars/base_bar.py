import pandas as pd

from common.modules.logger import logger
from abc import abstractmethod
from common.modules.enums import Exchange, Assets
from connector.npy.client import npy_client
from julia import Main
Main.eval('''
include("C://repos//trade//connector//ts2hdf5//client.jl")
using Dates
using PyCall
np = pyimport("numpy")

function send(py_ts, vec)
    v_ts=Nanosecond.(py_ts.astype(np.int64)) + DateTime(1970)
    ClientTsHdf5.upsert(
        meta,
        hcat(v_ts, vec)
    )
end
''')


class BaseBar:
    information = None

    def __init__(self, exchange: Exchange, sym: Assets, start, end, unit, **tags):
        # logger.warning(
        #     'Refactor to automatically find a unit size that has a sufficient resolution. Say 1000 per day. '
        #     'During training derive upsampled features, cumsum. moving averages.'
        #     'Expecting lowest resolution having no predictive value, while upsampled cumsum"s will have.'
        # )
        self.exchange = exchange
        self.sym = sym
        self.start = start
        self.end = end
        self.unit = unit
        self.tags = tags or {}
        self.__dict__.update(self.tags)

    @abstractmethod
    def resample(self) -> pd.DataFrame: pass

    @abstractmethod
    def add_measurable(self, df): pass

    def to_npy(self, df):
        if not self.information:
            raise ValueError('No information tag set.')
        assert len(df.index.unique()) == len(df), 'Timestamp is not unique. Group By time first before uploading to influx.'
        if isinstance(df, pd.DataFrame):
            for col, ps in df.iteritems():
                Main.eval(f'''
                        meta=Dict(
                            "measurement_name" => "trade bars",
                            "exchange" => "{self.exchange}",
                            "asset" => "{self.sym}",
                            "information" => "{self.information}",
                            "unit" => "{self.unit}",
                            "col" => "{col}"
                        )
                        ''')
                Main.send(df.index.values, ps.values)
        elif isinstance(df, pd.Series):
            Main.eval(f'''
                    meta=Dict(
                        "measurement_name" => "trade bars",
                        "exchange" => "{self.exchange}",
                        "asset" => "{self.sym}",
                        "information" => "{self.information}",
                        "unit" => "{self.unit}",
                    )
                    ''')
            Main.send(df.index.values, df.values)
        else:
            raise ValueError("Type of df not clear")
        # npy_client.upsert(
        #     meta={
        #         'measurement_name': 'trade bars',
        #         'exchange': self.exchange,
        #         'asset': self.sym,
        #         'information': self.information,
        #         'unit': self.unit,
        #     },
        #     data=df,
        # )
