import pandas as pd

from common.modules import features
from common.modules.data_store import DataStore
from common.utils.util_func import total_profit2
from connector.influxdb.influxdb_wrapper import InfluxClientWrapper as Influx
from trader.data_loader.features2influx import insert_tick_n


def ensure_ts_index(pdf: pd.DataFrame):
    if not pd.api.types.is_datetime64_any_dtype(pdf.index):
        pdf.index = pd.to_datetime(pdf['ts'])
        pdf.drop('ts', inplace=True, axis=1)
    return pdf


def ensure_tick_n(pdf, params):
    return insert_tick_n(pdf) if 'volume' in params.series_tick_type.type else pdf


def norm_index(pdf, params):
    pdf = ensure_tick_n(pdf, params)
    return ensure_ts_index(pdf)


def influx_store_results(params, data: DataStore):
    if not params.influx_store_results:
        return
    ndf = data.load_get(f'preds_{params.target_the}')
    pdf = pd.DataFrame(ndf, columns=ndf.dtype.names)
    # get a ts AND check for duplicate ts requiving vol tick n
    pdf.index = data.load_get(f'ohlc_{params.target_the}')['ts'].values
    db_insert_preds(norm_index(pdf, params), params, training_set=params.train)


def db_insert_preds(pdf, params, training_set):
    for model in [c for c in pdf.columns if c not in ['ts', 'tick_n']]:
        include_cols = [model] + ['tick_n'] if 'tick_n' in pdf.columns else [model]
        Influx().write_pdf(pdf[include_cols].rename({model: 'p'}, axis=1),
                           measurement='model_preds',
                           tags=dict(
                               asset=params.asset.lower(),
                               exchange=params.exchange.name,
                               ex=params.ex,
                               model=model,
                               tick_type=params.series_tick_type.type,
                               resample_val=params.series_tick_type.resample_val,
                               training_set=training_set
                           ),
                           field_columns=['p'],
                           tag_columns=['tick_n'] if 'volume' in params.series_tick_type.type else []
                           )


def store_labels(pdf: pd.DataFrame, params):
    pdf = norm_index(pdf, params)
    # for labels storing to display in UI we dont need the tick_n tag, for now...
    if 'tick_n' in pdf.columns:
        pdf = pdf[pdf['tick_n'] == 0]
    for feature in pdf.columns:
        try:
            features(feature)
            include_cols = [feature]  # + ['tick_n'] if 'tick_n' in pdf.columns else [feature]
            Influx().write_pdf(pdf[include_cols][pdf[feature] > 0].rename({feature: 'y'}, axis=1),
                               measurement='label',
                               tags=dict(
                                   ex=params.ex,
                                   asset=params.asset.lower(),
                                   exchange=params.exchange.name,
                                   feature=feature,
                                   tick_type=params.series_tick_type.type,
                               ),
                               field_columns=['y'],
                               # tag_columns=['tick_n'] if 'volume' in params.series_tick_type.type else []
                               )
        except ValueError:
            continue


def store_ohlc(pdf, params):
    pdf = norm_index(pdf, params)
    # for labels storing to display in UI we dont need the tick_n tag, for now...
    if 'tick_n' in pdf.columns:
        pdf = pdf[pdf['tick_n'] == 0]
    Influx().write_pdf(pdf,
                       measurement='ohlcv',
                       tags=dict(
                           asset=params.asset.lower(),
                           exchange=params.exchange.name,
                           tick_type=params.series_tick_type.type,
                       ),
                       field_columns=pdf.columns,
                       # tag_columns=[]
                       )


def store_rewards(pdp, field_columns, bt_i, params, overwrite=True):
    # rewards storage is for review only, not processing hence can reduce because of
    # influxdb.exceptions.InfluxDBClientError: 400: {"error": "partial write: max-values-per-tag limit exceeded (105417/100000): measurement=\"rl_training_preds\
    if pdp.empty:
        return
    pdp = norm_index(pdp, params)
    Influx().write_pdf(
        pdf=pdp[field_columns],  # .iloc[::10],
        measurement='rl_training_preds',
        field_columns=field_columns,
        tags=dict(
            ex=params.ex,
            asset=str(params.asset),
            bt_i=str(bt_i)
        ),
        overwrite=overwrite
    )


def store_bt_pnl(orders: list, bt_i, params, overwrite=True):
    pdf = pd.DataFrame([o.to_dict() for o in orders])
    pdf['total_pnl'] = total_profit2(orders)
    if pdf.empty:
        return
    pdf['total_pnl'] = pdf['total_pnl'].astype(float)
    pdf = pdf.set_index('ts_fill', drop=True)
    Influx().write_pdf(
        pdf=pdf,
        measurement='rl_training_pnl',
        field_columns=[c for c in pdf.columns if c not in ['ts_signal', 'ts_cancel', 'ts_order_place']],
        tags=dict(
            ex=params.ex,
            asset=str(params.asset),
            bt_i=str(bt_i)
        ),
        overwrite=overwrite
    )


def q_fields(measurement):
    from pprint import pprint
    res_set = Influx().q(f'''SHOW FIELD KEYS ON trade FROM {measurement}''')
    pprint(res_set)
    pprint([tup[0] for tup in res_set.raw.get('series')[0].get('values')])
    print('-' * 30)


def q_tags(measurement, key) -> list:
    from pprint import pprint
    res_set = Influx().q(f'''SHOW TAG VALUES ON trade FROM {measurement} WITH KEY = {key}''')
    # Influx().q(f'''SHOW TAG VALUES ON trade FROM {measurement}''')
    try:
        tags = [tup[1] for tup in res_set.raw.get('series')[0].get('values')]
        pprint(tags)
        print('-' * 30)
        return tags
    except TypeError:
        print(f'No tags for {measurement} {key}')
    print('-' * 30)
    return []


def d_series_by_tag(measurement, tags: dict):
    print(f'Deleting {measurement} - {tags}')
    Influx.df_client.delete_series('trade', measurement, tags=tags)


if __name__ == '__main__':
    measurement = 'model_preds'
    q_fields(measurement)
    key = 'model'
    tags = q_tags(measurement, key)
    # for tag_val in tags:
    #     d_series_by_tag(measurement, {key: tag_val})
    q_tags(measurement, key)
