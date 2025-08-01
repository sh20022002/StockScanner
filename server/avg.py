import polars as pl
import pandas as pd



def find_avg(df:pl.DataFrame) -> pl.DataFrame:
    '''
    finds the simple moving avrage the the stock price baunce on 20/50/100/150/200
    '''
    required_columns = ['Open', 'High', 'Low', 'Close', 'Volume', 'SMA150']

    df = df[required_columns].drop_nulls()
    
    df = df.with_columns([
        (((df['Close'] - df['SMA150'])/df['Close'])*100).alias('dis_from_150')])
    me = df['dis_from_150'].mean()

    return me


    
    
