import pandas as pd
from julia import Main as Jl

Jl.eval('''include("C://repos//trade//connector//ts2hdf5//client.jl")''')


def upsert(meta: dict, data):
    if isinstance(data, pd.Series):
        Jl.ClientTsHdf5.py_upsert(meta, data.index.values, data.values)
    elif isinstance(data, pd.DataFrame):
        for col, data in data.iteritems():
            Jl.ClientTsHdf5.py_upsert({**meta, **{'col': col}}, data.index.values, data.values)
    else:
        raise ValueError("Type of df not clear")


def query(meta: dict, start="", stop="9") -> pd.DataFrame:
    cols, mat = Jl.ClientTsHdf5.py_query(meta, start=str(start), stop=str(stop))
    df = pd.DataFrame(mat, columns=cols)
    if 'ts' in cols:
        df = df.set_index("ts")
    return df


if __name__ == '__main__':
    import datetime
    import pandas as pd
    # upsert({'a': 3}, pd.Series([1], index=[datetime.datetime(2022, 1, 1)]))
    print(query(meta={
        "measurement_name" : "trade bars",
        "exchange" : "bitfinex",
        "asset" : "ethusd",
        "information" : "volume"
    }))
    # Jl.ClientTsHdf5.drop({'a': 3})
