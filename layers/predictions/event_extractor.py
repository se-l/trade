import typing
import numpy as np
import pandas as pd

from common.modules.logger import logger


class EventExtractor:
    def __init__(self, thresholds=None, method='std'):
        self.method = method
        self.thresholds = thresholds or [None, None]

    def __call__(self, ps: pd.Series, max_events: typing.Union[int, float] = None) -> pd.Index:
        iloc_nnan_start = np.argmin(np.isnan(ps.values))
        iloc_nnan_end = len(ps) - np.argmin(np.isnan(ps.values[::-1]))
        ps2 = ps.iloc[iloc_nnan_start:iloc_nnan_end]
        if ps2.isna().sum() > 0:
            logger.info(f'{ps.name} contained nan values')
        # elif not is_stationary(ps2.values):
        #     logger.info(f'{ps.name} is not stationary')
        #     return pd.Index([])
        elif self.method == 'std':
            threshold = ps2.std() * self.thresholds[-1]
            ix = ps2.index[ps2.abs() >= threshold]
            logger.info(f'Sampling {len(ix)} events for {ps.name}')
            if len(ix) > max_events:
                return pd.Index([])
            else:
                return ix
        else:
            raise NotImplementedError('Unclear method how to establish a range')
