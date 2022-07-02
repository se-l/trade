import os
import json
import hashlib
import numpy as np
import pandas as pd

from common.paths import Paths


class Client:
    """Each series a numpy binary"""
    def get(self, meta) -> np.ndarray:
        return np.load(os.path.join(Paths.data, 'npy', self.fname(meta)))

    def query(self, meta: dict, params: dict):
        """Given meta, get object. Given params return right time slice"""
        pass

    def upsert(self, meta, data: np.ndarray | pd.Series | pd.DataFrame):
        if isinstance(data, pd.DataFrame):
            for col, ps in data.iteritems():
                self.upsert(meta={**meta, **{'column': col}}, data=ps)
        else:
            p = self.path(meta)
            if os.path.exists(p):
                d0 = np.load(p)
                if isinstance(data, np.ndarray):
                    d1 = data
                elif isinstance(data, pd.Series):
                    d1 = np.array((data.index, data.to_numpy()))
                else:
                    raise
                _, ix1, ix2 = np.intersect1d(d0[0], d1[0], return_indices=True)
                d01 = np.concatenate((d0[np.setdiff1d(range(len(d0)), ix1)], d1))
            else:
                d01 = data
                self.register(meta)
            np.save(file=p, arr=d01)

    def delete(self, meta: dict | str):
        fname = self.fname(meta) if isinstance(meta, dict) else meta
        self.unregister(fname)
        os.remove(os.path.join(Paths.data, 'npy', fname))

    def fname(self, meta: dict) -> str:
        return hashlib.md5(json.dumps(meta).encode()).hexdigest()

    def path(self, meta: dict):
        return os.path.join(Paths.data, 'npy', self.fname(meta))

    def register(self, meta):
        p = os.path.join(Paths.data, 'npy', 'registry.json')
        if not os.path.exists(p):
            map = {}
        else:
            with open(p, 'r') as f:
                map = json.load(f)
        map[self.fname(meta)] = meta
        with open(p, 'w') as f:
            json.dump(map, f)

    def unregister(self, key: str):
        p = os.path.join(Paths.data, 'npy', 'registry.json')
        with open(p, 'r') as f:
            map = json.load(f)
        del map[key]
        with open(p, 'w') as f:
            json.dump(map, f)


npy_client = Client()

if __name__ == '__main__':
    client = Client()
    import datetime
    ts = [datetime.datetime.utcnow() + datetime.timedelta(days=i) for i in range(10)]
    ps = pd.Series(range(10), index=ts)
    client.upsert({'a': 2}, ps)

