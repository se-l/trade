from qc.live_sim_recon.bitmex_exec_tradeHistory import trade_history
import numpy as np
import datetime
from trade_mysql.mysql_conn import Db

values = []
ts_cols = ["transactTime", "timestamp"]
for trade in trade_history:
    for col in ts_cols:
        trade[col] = np.datetime64(trade[col]).astype(datetime.datetime)
    values.append(tuple(trade.values()))
target_cols = list(trade_history[0].keys())

db = Db()
sql = '''insert into trade.bitmex_execution_history (`{0}`) values ({1})
                                    on duplicate key update {2};'''.format(
            '`, `'.join(target_cols),
            ', '.join(['%s'] * len(target_cols)),
            ', '.join(['`{0}`=values(`{0}`)'.format(col) for col in target_cols])
        )
db.single_thread_insert(values, sql, db)


db.close()

