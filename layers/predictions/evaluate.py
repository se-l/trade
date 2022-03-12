import pickle
import datetime
import pandas as pd
import os
import numpy as np

from sklearn.metrics import f1_score
from common.modules.assets import Assets
from common.modules.exchange import Exchange
from common.paths import Paths
from connector.influxdb.influxdb_wrapper import influx
from layers.predictions.classify.entry_side import EstimateSide

exchange = Exchange.bitfinex
sym = Assets.ethusd
start=datetime.datetime(2022, 2, 7)
end=datetime.datetime(2022, 3, 2)
ex = 'ex2022-03-06_165641-ethusd'

df_label = influx.query(query=influx.build_query(predicates={'_measurement': 'label', 'exchange': exchange.name, 'asset': sym.name,
                                                                     'expiration_window': '180min', '_field': 'label'},
                                                         start=start,
                                                         end=end),
                                return_more_tables=False,
                                name='label'
                                )

with open(os.path.join(Paths.trade_model, ex, 'f1_ho.p'), 'rb') as f:
    preds = pickle.load(f)
f1 = preds.merge(df_label, how='inner', right_index=True, left_index=True)
for i, side in enumerate(['short', 'flat', 'long']):
    print(f"{side}: {f1_score(np.where(f1['label'] == i, 1, 0), f1[side].round().values)}")
