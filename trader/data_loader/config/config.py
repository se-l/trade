from common.modules.exchange import Exchange
from common.paths import Paths

ccy2folder = dict(xrpxbt='xrpusd', bchxbt='bchusd')
resolution2folder = {'1S': 'second', 1: 'second', '60S': 'minute', 60: 'minute'}
resample_sec2resample_str = lambda x: f'{x}S'
exchange2asset_class = {
    Exchange.fxcm: Paths.qc_forex,
    Exchange.bitmex: Paths.qc_crypto,
    Exchange.bitfinex: Paths.qc_bitfinex_crypto,
}

drop_features = [
    # 'ADX_real',
    'APO_real',
    'AROON_aroondown',
    'AROON_aroonup',
    'AROONOSC_real',
    'BOP_real',
    'CCI_real',
    'CMO_real',
    # 'DX_real',
    # 'MACD_macd', 'MACD_macdsignal', 'MACD_macdhist',
    'MOM_real',
    'PPO_real',
    'ROC_real',  # 'ROCP_real',
    'RSI_real',
    'STOCH_slowk', 'STOCH_slowd',
    'ULTOSC_real',
    'WILLR_real',
    'ADOSC_real'
    'STDDEV_real',
    # 'VAR_real',
    'NATR_real',
    # 'ADXR_real', 'ATR_real'
]
