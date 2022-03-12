import os

from pathlib import Path

log_fn = 'log_{}.txt'
fp = Path(__file__)
lib_path = fp.resolve().parents[1]


class Paths:
    """
    Project paths for easy reference.
    """
    lib_path = lib_path
    projectDir = lib_path
    trader = os.path.join(lib_path, 'trader')
    files = os.path.join(lib_path, 'files')
    dir_norm_tal = os.path.join(lib_path, 'norm_tal')

    qc_data = os.path.join(lib_path, 'data')
    # qc_data = r'C:\repos\quantconnect\Lean3\Data'
    qc_forex = os.path.join(lib_path, 'data', 'forex')
    # qc_forex = r'C:\repos\quantconnect\Lean3\Data\forex'
    qc_crypto = os.path.join(lib_path, 'data', 'crypto')
    # qc_crypto = r'C:\repos\quantconnect\Lean3\Data\crypto'
    qc_bitfinex_crypto = os.path.join(lib_path, 'data', 'crypto', 'bitfinex')
    # qc_bitmex_crypto = r'C:\repos\quantconnect\Lean3\Data\crypto\bitmex'
    qc_bitmex_crypto = os.path.join(lib_path, 'data', 'crypto', 'bitmex')
    bitmex_raw = os.path.join(lib_path, 'data', 'crypto', 'bitmex', 'raw')
    bitfinex_tick = os.path.join(lib_path, 'data', 'crypto', 'bitfinex', 'tick')
    bitmex_raw_online_quote = r'https://s3-eu-west-1.amazonaws.com/public.bitmex.com/data/quote/{}.csv.gz'
    bitmex_raw_online_trade = r'https://s3-eu-west-1.amazonaws.com/public.bitmex.com/data/trade/{}.csv.gz'

    trade_ini_fn = os.path.join(lib_path, 'trade_ini.json')
    model_features = os.path.join(lib_path, 'model', 'model_features.json')
    trade_model = os.path.join(lib_path, 'model', 'supervised')
    backtests = os.path.join(lib_path, 'model', 'backtests')
    simulate = os.path.join(lib_path, 'model', 'simulate')
    model_rl = os.path.join(lib_path, 'model', 'reinforcement')
    path_buffer = os.path.join(projectDir, 'subprocess_buffer')

    path_config_supervised = 'trader.train.config.supervised'
    path_config_reinforced = 'trader.train.config.reinforced'
    path_config_backtest = 'trader.backtest.config'
    path_config_labeler = 'trader.train.config.labeler'
    path_config_simulate = 'trader.data_loader.config.simulate'
