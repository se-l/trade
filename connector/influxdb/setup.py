from influxdb import InfluxDBClient


client = InfluxDBClient('localhost', 8086, 'root', 'root', 'trade')
client.create_database('trade')

# example
# json_body = [
#     {
#         "measurement": "cpu_load_short",
#         "tags": {
#             "host": "server01",
#             "region": "us-west"
#         },
#         "time": "2019-12-14T12:00:00Z",
#         "fields": {
#             "value": 0.8
#         }
#     }
# ]
# client.write_points(json_body)
