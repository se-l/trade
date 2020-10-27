# from globals import *
import json
from scrape.BitmexTradesDownloader import main as bitmex_trade_downloader
from qc.inputDataConversion.convertBitMexToQC import main as bitmex_qc_conversion, copy_xbt_to_btc
from qc.inputDataConversion.convertBitMexToQC import ConvertBitmexToQC
from trade_mysql.update_next_fill import main as update_bt_fills
from trade_mysql.update_ohlcv import main as update_ohlc_bid_ask, UpdateSqlOhlcv
from qc.starter_2 import Starter
from qc.params import params_starter as params
import datetime
from trade_mysql.mysql_conn import Db


def main(params):
    bitmex_trade_downloader()
    last_raw_date = ConvertBitmexToQC.identify_latest_file_date(folder='raw')
    print(f"Latest downloaded date: {last_raw_date}")
    bitmex_qc_conversion(['xbtusd', 'ethusd', 'xrp', 'bch'])
    copy_xbt_to_btc()
    for symbol in ['ethusd', 'xbtusd', 'xrpxbt', 'bchxbt']:
        start_sql_ts = UpdateSqlOhlcv.get_min_of_max_ts(symbol, tables=['ohlcv', 'ohlcv_bid', 'ohlcv_ask'])
        if last_raw_date - start_sql_ts < datetime.timedelta(hours=12):
            continue
        params.asset = symbol
        params.data_start = start_sql_ts
        params.data_end = last_raw_date
        update_ohlc_bid_ask(params)

    # Here RUN QC VS indicator DB fill. BEFORE predictions
    return
    # pass dates to TickForeacast and starter 3 class and run it. Using Talib only, no QC indicators.
    # Difference are too minor

    for symbol in ['xbtusd', 'ethusd']:  ##, 'xrpusd']:
        db = Db()
        params.asset = symbol
        params.data_start = start_sql_ts
        params.data_end = last_raw_date + datetime.timedelta(days=1)
        if False:
            start_sql_ts = db.fetchall(f'''select max(ts) from trade.fills where asset = '{params.asset}' ''')[0][0]
            if last_raw_date - start_sql_ts < datetime.timedelta(hours=36):
                continue
            update_bt_fills(params)

        with open(os.path.join(projectDir, 'trade_ini.json'), 'rt') as f:
            trade_ini = json.load(f)
        use_vs = True
        start_sql_ts = db.fetchall('''select max(ts) from trade.{}predictions where asset = '{}' '''.format(
            'vs_' if use_vs else '',
            params.asset.lower()
        ))[0][0]
        if start_sql_ts is not None:
            # The indicators to be generated have a large unstable period, therefore a warmup period of 2 days.
            # Fails like this. first 2 days get overwritten with bad data. need to cut off 2 days warm up!!!
            # will be performed in FeatureEngineering section
            if params.data_end - start_sql_ts < datetime.timedelta(hours=12):
                continue
            if not use_vs:
                params.data_start = start_sql_ts - datetime.timedelta(days=2)
            else:
                params.data_start = start_sql_ts - datetime.timedelta(seconds=1)
        starter = Starter(params, ex_dir_name=trade_ini['live_ex_path_rel'][symbol], use_vs=use_vs)
        starter.run()
        # store_y_label(params)
        # perhaps repartition
    db.close()

if __name__ == '__main__':
    main(params)
