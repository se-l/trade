import re
import datetime
import pandas as pd
import influxdb_client
from influxdb_client.client.write_api import SYNCHRONOUS, ASYNCHRONOUS
# You can generate an API token from the "API Tokens Tab" in the UI
from common.utils.window_aggregator import WindowAggregator

token = "2KWQnyVoTekdn08KdnUcACrzyFOZ0ag4wBjGUe4YJ1_q21SBbGbyzl6wo8_IFD5XCyVVbqAgZl9KjpBMagKZ9Q=="
org = "Seb"
bucket = "trading"
url = "http://localhost:8086"


class Influx:
    """https://docs.influxdata.com/influxdb/cloud/api-guide/client-libraries/python/"""
    _instance = None

    def __new__(cls):
        if not cls._instance:
            cls._instance = object.__new__(cls)
        return cls._instance

    def __init__(self):
        self.client = influxdb_client.InfluxDBClient(
            url=url,
            token=token,
            org=org
        )
        self.api_write = self.client.write_api(write_options=SYNCHRONOUS)
        self.api_query = self.client.query_api()
        self.api_delete = self.client.delete_api()

    def query(self, query: str, return_more_tables=False, name: str = None) -> pd.DataFrame:
        """query = f'''
                    from(bucket:"trading")
                    |> range(start: 2020-01-01T00:00:00Z, stop: 2020-01-05T00:00:00Z)
                    |> filter(fn:(r) => r._measurement == "trade bars" and
                                        r.weighting == "ewm" and
                                        r._field == "volatility"
                                        )
                    '''
        """
        result = self.api_query.query(org=org, query=query)
        dfs = []
        for table in result:
            records = []
            window_aggregator = self.query2WindowAggregator(query)
            tags = {**{'_measurement': table.records[0].get_measurement(),
                       '_field': table.records[0].get_field()},
                    **({'_window_aggregator': window_aggregator} if window_aggregator else {}),
                    **{k: v for k, v in table.records[0].values.items() if not k.startswith('_') and k not in ('result', 'table')}}
            for record in table.records:
                records.append({'timestamp': record.get_time(), 'field': record.get_field(), 'value': record.get_value()})
            dfs.append(pd.DataFrame(records).pivot(index='timestamp', columns='field')['value'].
                       rename(columns={table.records[0].get_field(): '|'.join([f'{tag}-{val}' for tag, val in tags.items()]) if not name else name}))
        if not return_more_tables and len(dfs) > 1:
            raise ValueError('Too many tables. Need to pick more tags')
        elif return_more_tables:  # need to start naming columns here according to tags...
            return pd.concat(dfs, axis=1, sort=False)
        else:
            return dfs[0]

    @staticmethod
    def query2WindowAggregator(query: str) -> [WindowAggregator, None]:
        match = re.search(r'aggregateWindow\(every: (\w.*), fn: (\D.*)\)', query)
        if match:
            return WindowAggregator(match.group(1), match.group(2))

    @staticmethod
    def build_query(predicates: dict, start: datetime, end: datetime, window_aggregator: WindowAggregator = None) -> str:
        preds = ' and '.join([f'r.{col} == "{value}"' for col, value in predicates.items()])
        aggregation = f'|> aggregateWindow(every: {window_aggregator.window}, fn: {window_aggregator.aggregator})' if window_aggregator else ''
        return f'''
            from(bucket:"trading")
            |> range(start: {start.isoformat()}Z, stop: {end.isoformat()}Z)
            |> filter(fn:(r) => {preds}) {aggregation}
        '''

    def write_dataframe(self, df, data_frame_measurement_name: str, data_frame_tag_columns: list = None):
        if isinstance(data_frame_tag_columns, dict):
            for tag, value in data_frame_tag_columns.items():
                df[tag] = value
            data_frame_tag_columns = list(data_frame_tag_columns.keys())
        # df.index = pd.to_datetime(df.index, unit='ns')

        self.api_write.write(bucket=bucket, org=org, record=df,
                             data_frame_measurement_name=data_frame_measurement_name,
                             data_frame_tag_columns=data_frame_tag_columns)
        print(f'Injected {df.shape} dataframe records')

    def write(self, record, **kwargs):
        """
        p = influxdb_client.Point("my_measurement").tag("location", "Prague").field("temperature", 25.3)
        write_api.write(bucket=bucket, org=org, record=p)
        data = "mem,host=host1 used_percent=23.43234543"
        """
        record = pd.DataFrame(record) if isinstance(record, pd.Series) else record
        if isinstance(record, pd.DataFrame):
            self.write_dataframe(df=record, **kwargs)
        else:
            self.api_write.write(bucket=bucket, org=org, record=record)

    def delete(self, predicate, start: datetime = datetime.datetime(1970, 1, 1), stop=datetime.datetime(2099, 12, 31)):
        self.api_delete.delete(start=start, stop=stop, predicate=predicate, bucket=bucket, org=org)

    def close(self):
        self.client.close()


influx = Influx()

if __name__ == '__main__':
    pass
    # influx.delete(predicate='''_measurement="trade bars" ''')
    # influx.delete(predicate='''_measurement="trade bars" and information="imbalance" and unit="ethusd" and unit_size=100 ''')
    influx.delete(predicate='''_measurement="order book" ''')
    # influx.query(query='''
    #                 from(bucket:"trading")
    #                 |> range(start: 2020-01-01T00:00:00Z, stop: 2020-01-05T00:00:00Z)
    #                 |> filter(fn:(r) => r._measurement == "trade bars" and
    #                                     r.weighting == "ewm" and
    #                                     r._field == "volatility"
    #                                     )
    #                 ''')
    #
    print('Done')
