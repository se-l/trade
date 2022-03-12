import os
import datetime
import time

from urllib.request import urlretrieve
from concurrent.futures import ThreadPoolExecutor as PoolExecutor

from common.utils.util_func import date_day_range
from common.paths import Paths


class BitmexTradesDownloader:
    quote = Paths.bitmex_raw_online_quote
    trade = Paths.bitmex_raw_online_trade
    utc_now = int(time.time()) * 1000
    utc_target = utc_now - 1 * 84600 * 1000
    pp = 4

    @classmethod
    def get_target_dir(cls, qt: str):
        return os.path.join(Paths.bitmex_raw, qt)

    @classmethod
    def pp_download(cls, qt_date):
        qt, date = qt_date
        date = date.strftime('%Y%m%d')
        print('Downloading {} - {}'.format(qt, date))
        try:
            urlretrieve(cls.__getattribute__(cls, qt).format(date),
                    os.path.join(cls.get_target_dir(qt), '{}.csv.gz'.format(date)))
        except:
            print('Not yet available: {} - {}'.format(qt, date))

    @classmethod
    def get_qt_dates(cls):
        qt_dates = []
        # for qt in ['quote', 'trade']:
        for qt in ['trade']:
            a = list(os.walk(cls.get_target_dir(qt)))[0][2]
            latest_file_date = max([int(o[0:8]) for o in a])
            latest_file_date = datetime.datetime.strptime(str(latest_file_date), '%Y%m%d')
            end_date = datetime.datetime.fromtimestamp(cls.utc_target // 1000) + datetime.timedelta(days=1)
            for date in date_day_range(latest_file_date+datetime.timedelta(days=1), end_date):
                qt_dates.append((qt, date))
        print('Downloading {} files in {} threads'.format(len(qt_dates), cls.pp))
        return qt_dates


def main():
    qt_dates = BitmexTradesDownloader.get_qt_dates()
    with PoolExecutor(max_workers=BitmexTradesDownloader.pp) as executor:
        for _ in executor.map(BitmexTradesDownloader.pp_download, qt_dates):
            pass


if __name__ == '__main__':
    main()
