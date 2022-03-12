from scipy import stats
import numpy as np

for p in range(99, 0, -1):
    d = list(range(100))
    zd = stats.zscore(d)
    r = [e for e in zd if e > np.percentile(zd, p)]

    d2 = list(range(1000))
    zd2 = stats.zscore(d)
    r2 = [e for e in zd2 if e > np.percentile(zd2, p)]

    if len(r) > 10 and len(r2) > 10:
        print(p)
        break
