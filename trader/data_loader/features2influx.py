import numpy as np
import pandas as pd

from functools import reduce
from common.modules.series import Series
from common.paths import Paths
from trader.data_loader.utils_features import get_ohlc
from connector.influxdb.influxdb_wrapper import InfluxClientWrapper as Influx


def insert_tick_n(pdf):
    # ensure there's a ts column
    if 'ts' not in pdf.columns:
        # assume it's the index. todo: include check + warning
        pdf['ts'] = pdf.index
        # pdf = pdf.set_index('ts', drop=False)
    pdf['tick_n'] = 0
    pdf = pdf.sort_values('ts')
    pdf = pdf.reset_index(drop=True)
    ix_dup_all = pdf.index[pdf['ts'].duplicated(keep=False)]
    ix_dup = pdf.index[pdf['ts'].duplicated(keep="first")]
    ix_dup_p1 = ix_dup + 1
    ix_dup_m1 = ix_dup - 1
    ix_group_start = np.setdiff1d(ix_dup_m1, ix_dup)
    ix_group_end = np.setdiff1d(ix_dup_p1, ix_dup)
    assert len(ix_group_start) == len(ix_group_end)
    tick_n = [list(range(ix_group_end[i] - ix_group_start[i])) for i in range(len(ix_group_end))]
    tick_n = reduce(lambda x, y: x + y, tick_n, [])
    if len(tick_n) != len(pdf.loc[ix_dup_all]):
        print('fix tick n')
    pdf.loc[ix_dup_all, 'tick_n'] = tick_n
    return pdf.set_index('ts', drop=False)


def run(params):
    influx = Influx()
    kwargs = {name: params.__getattribute__(name) for name in ['exchange', 'series_tick_type', 'asset']}
    for series in [Series.trade, Series.quote]:
        pdf = get_ohlc(
            start=params.data_start,
            end=params.data_end,
            series=series,
            **kwargs
        )
        insert_tick_n(pdf)
        pdf.index = pd.to_datetime(pdf.index)

        influx.write_pdf(pdf,
                         measurement='ohlcv',
                         tags=dict(
                             asset=params.asset.lower(),
                             exchange=params.exchange.name,
                             series_name=series.name,
                             tick_type=params.series_tick_type.type,
                             resample_val=params.series_tick_type.resample_val,
                         ),
                         field_columns=pdf.columns,
                         tag_columns=['tick_n']
                         )


if __name__ == '__main__':
    import importlib
    params_ = importlib.import_module('{}.{}'.format(Paths.path_config_reinforced, 'ethusd')).Params()
    run(params_)
