import importlib
import click

from common.modules import series
from common import Paths
from common.modules import dotdict
from connector import store_ohlc
from trader.data_loader.utils_features import get_ohlc


@click.command('ohlcv_into_influx')
@click.pass_context
def ohlcv_into_influx(ctx):
    params = importlib.import_module('{}.{}'.format(Paths.path_config_simulate, ctx.obj.fn_params)).Params()
    pdf = get_ohlc(
        start=params.data_start,
        end=params.data_end,
        asset=params.asset,
        exchange=params.exchange,
        series_tick_type=params.series_tick_type,
        series=series.trade
    )
    store_ohlc(pdf, params)


@click.command()
@click.pass_context
def main(ctx):
    ctx.obj = dotdict(dict(
        fn_params='sim001'
    ))
    ctx.forward(ohlcv_into_influx)


if __name__ == '__main__':
    main()
