from influxdb import DataFrameClient
import pandas as pd
from common.modules.logger import logger


class InfluxClientWrapper:
    df_client = DataFrameClient(
        host='localhost',
        port=8086,
        username='root',
        password='root',
        database='trade')

    def write_pdf(s,
                  pdf: pd.DataFrame,
                  measurement: str,
                  field_columns: list,
                  tags: dict = None,
                  tag_columns: list = None,
                  overwrite: bool = False
                  ):
        if overwrite:
            s.df_client.delete_series(measurement=measurement, tags=tags)
        s.df_client.write_points(
            pdf,
            measurement=measurement,
            tags=tags,
            tag_columns=tag_columns,
            field_columns=field_columns,
            protocol='line',
            batch_size=10000
        )

    def q(s, sql):
        return s.df_client.query(sql)

    def load_p(s, asset, model, ex, from_ts, to_ts, tbl='model_preds') -> pd.DataFrame:
        logger.info('Loading features from model_preds...')
        #            #            # GROUP BY time(1m)
        sql = f'''select p from {tbl} where asset='{asset}' and ex='{ex}' and model='{model}' and time >='{from_ts}' and time <='{to_ts}' order by time '''
        res = s.q(sql)[tbl]
        return res
        # return res.reset_index().pivot(index='index', columns='model', values='p').tz_localize(None)

    def close(s):
        s.df_client.close()
