import pandas as pd
from julia import Main

Main.eval('''
include("C://repos//trade//connector//ts2hdf5//client.jl")
using Dates
using PyCall
np = pyimport("numpy")

function send(meta, py_ts, vec)
    v_ts=Nanosecond.(py_ts.astype(np.int64)) + DateTime(1970)
    ClientTsHdf5.upsert(
        meta,
        hcat(v_ts, vec)
    )
end
''')


def upsert(meta: dict, data):
    if isinstance(data, pd.Series):
        Main.send(meta, data.index.values, data.values)
    elif isinstance(data, pd.DataFrame):
        for col, data in data.iteritems():
            Main.send({**meta, **{'col': col}}, data.index.values, data.values)
    else:
        raise ValueError("Type of df not clear")


def query(meta: dict, start="", stop="0"):
    return Main.ClientTsHdf5.query(meta, start=str(start), stop=str(stop))


if __name__ == '__main__':
    import datetime
    import pandas as pd
    upsert({'a': 3}, pd.Series([1], index=[datetime.datetime(2022, 1, 1)]))
    print(Main.ClientTsHdf5.query({'a': 3}))
    Main.ClientTsHdf5.drop({'a': 3})
