from trade_mysql.mysql_conn import Db
from utils.utils import Logger
import datetime, os


class DbStatus:
    def __init__(s):
        s.db = Db()
        s.sql1 = '''select max(ts) from trade.{0} where asset='{1}'; '''
        s.ts_tables = ['vs_indicators', 'vs_predictions']
        s.assets = ['xbtusd', 'ethusd', 'xrpusd', 'bchusd']

    def run(s):
        Logger.init_log(os.path.join(r'C:\repos\trade\qc\daily_updates', 'log_db_status.txt'))
        Logger.info(datetime.date.today())
        for tbl in s.ts_tables:
            for asset in s.assets:
                Logger.debug('{} - {} - {}'.format(
                    tbl, asset, [v[0].strftime('%Y-%m-%d %H:%M:%S') for v in s.db.fetchall(s.sql1.format(tbl, asset))]
                ))


if __name__ == '__main__':
    dbStatus = DbStatus()
    dbStatus.run()
