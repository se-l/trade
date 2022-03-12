import pandas as pd

df = pd.DataFrame(range(3), columns=['a'])
df.loc[2, {}]