import zipfile

import os

import datetime

import math

import importlib
import click
import numpy as np
import pandas as pd
from common.modules import ctx
from common.modules import dotdict
from common import Paths
from common.refdata import date_formats
from common.utils import fluent
from common.utils.util_func import standard_params_setup, date_day_range


class Simulator:
    def __init__(s, params):
        s.params = params
        s.save_dir = os.path.join(Paths.qc_bitmex_crypto, s.params.series_tick_type.folder, s.params.asset)
        s.out_waves = {}
        s.df_quotes = None
        s.df_trades = None

    def generate_wave(s):
        waves = []
        for w in s.params.wave_params:
            single_wave = 2 * np.pi * np.arange(0, w['wavelength'], 1) / w['wavelength']
            single_wave = np.sin(single_wave) * w.get('amplitude', 1)
            single_wave += w.get('zero_line', 0)
            extend_f = math.floor((1+(s.params.data_end - s.params.data_start).total_seconds()) / w['wavelength'])
            single_wave = single_wave.tolist() * extend_f
            waves.append(single_wave)
        min_len = min((len(w) for w in waves))
        mid_wave = np.sum([w[:min_len] for w in waves], axis=0)
        return mid_wave

    @fluent
    def generate_bid_ask(s):
        mid_wave = s.generate_wave()
        ask = pd.Series(mid_wave + (s.params.bid_ask_spread / 2))
        bid = pd.Series(mid_wave - (s.params.bid_ask_spread / 2))
        s.df_quotes = pd.concat([
            s.gen_ts(), bid, bid, bid, bid, s.size(mid_wave), ask, ask, ask, ask, s.size(mid_wave)
        ], axis=1)
        s.df_quotes = s.df_quotes.set_index(0)

    @fluent
    def generate_trade(s):
        mid_wave = pd.Series(s.generate_wave())
        s.df_trades = pd.concat([
            s.gen_ts(), mid_wave, mid_wave, mid_wave, mid_wave, s.size(mid_wave)
        ], axis=1)
        s.df_trades = s.df_trades.set_index(0)

    def size(s, mid_wave):
        return pd.Series(np.ones(len(mid_wave)) * 10000)

    def gen_ts(s):
        return pd.date_range(s.params.data_start, s.params.data_end, freq='S').to_series().reset_index(drop=True)

    def save(s):
        for date in date_day_range(s.params.data_start, s.params.data_end + datetime.timedelta(days=1)):
            df_quotes_intts = s.pick_idx(s.df_quotes, date)
            df_trades_intts = s.pick_idx(s.df_trades, date)
            s.df_day_qt(df_trades_intts, date.strftime(date_formats.Ymd), qt='trade')
            s.df_day_qt(df_quotes_intts, date.strftime(date_formats.Ymd), qt='quote')

    @staticmethod
    def pick_idx(df, date):
        df_day = df.iloc[np.where((df.index.day == date.day) & (df.index.month == date.month) & (df.index.year == date.year))[0], :]
        df_day.index = (df_day.index - date).total_seconds()
        df_day.index = pd.to_numeric(df_day.index * 1000, downcast='integer')
        return df_day

    def fn_qt(s, str_date, qt):
        return os.path.join(s.save_dir, f'{str_date}_{s.params.asset}_{s.params.series_tick_type.folder}_{qt}.csv')

    def zip_qt(s, str_date, qt):
        return os.path.join(s.save_dir, f'{str_date}_{qt}.zip')

    def df_day_qt(s, df, str_date, qt, header=False):
        df.to_csv(s.fn_qt(str_date, qt), header=header)
        # better use io.text buffer instead of actually writing to disk twice.
        s.zip_away(s.fn_qt(str_date, qt), s.zip_qt(str_date, qt))

    @staticmethod
    def zip_away(fn, zip_name):
        with zipfile.ZipFile(zip_name, 'w', compression=zipfile.ZIP_DEFLATED) as myzip:
            myzip.write(fn, os.path.basename(fn))
        os.remove(fn)


@click.command('train_rl')
@click.pass_context
def train_rl(ctx: ctx):
    params = importlib.import_module('{}.{}'.format(Paths.path_config_simulate, ctx.obj.fn_params)).Params()
    standard_params_setup(params, Paths.simulate)
    sim = Simulator(params)
    sim.generate_bid_ask()
    sim.generate_trade()
    sim.save()


@click.command()
@click.pass_context
def main(ctx):
    ctx.obj = dotdict(dict(
        fn_params='sim001'
    ))
    ctx.forward(train_rl)


if __name__ == '__main__':
    main()
