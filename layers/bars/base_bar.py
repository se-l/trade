import pandas as pd

from common.modules.logger import logger
from connector.influxdb.influxdb_wrapper import influx
from abc import abstractmethod
from common.modules.enums import Exchange, Assets


class BaseBar:
    information = None

    def __init__(self, exchange: Exchange, sym: Assets, start, end, unit, **tags):
        logger.warning(
            'Refactor to automatically find a unit size that has a sufficient resolution. Say 1000 per day. '
            'During training derive upsampled features, cumsum. moving averages.'
            'Expecting lowest resolution having no predictive value, while upsampled cumsum"s will have.'
        )
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

    def to_influx(self, df):
        if not self.information:
            raise ValueError('No information tag set.')
        assert len(df.index.unique()) == len(df), 'Timestamp is not unique. Group By time first before uploading to influx.'
        influx.write(
            record=df,
            data_frame_measurement_name='trade bars',
            data_frame_tag_columns={**{
                'exchange': self.exchange,
                'asset': self.sym,
                'information': self.information,
                'unit': self.unit,
            }, **self.tags},
        )
