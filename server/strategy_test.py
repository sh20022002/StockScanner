import strategy
import polars as pl
# import os
# print(os.getcwd())
df = pl.read_csv('APPL1m.csv', has_header=True)


print(df.head(30))