import sys
sys.path.append(r'C:\repos\trade')
from globals import *
import json
from scrape.BitmexTradesDownloader import main as bitmex_trade_downloader
from qc.inputDataConversion.convertBitMexToQC import main as bitmex_qc_conversion, copy_xbt_to_btc
from qc.inputDataConversion.convertBitMexToQC import ConvertBitmexToQC
from qc.params import params_starter as params
from utils.utilFunc import dotdict
import datetime


def main(params):
    bitmex_trade_downloader()
    last_raw_date = ConvertBitmexToQC.identify_latest_file_date(folder='raw')
    print(f"Latest downloaded date: {last_raw_date}")
    bitmex_qc_conversion(['xbtusd'])  # , 'ethusd', 'xrp', 'bch'])
    copy_xbt_to_btc()

    # Here RUN QC VS indicator DB fill. BEFORE predictions
    return


if __name__ == '__main__':
    main(params)
