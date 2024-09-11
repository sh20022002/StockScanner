import yfinance as yf
import pandas as pd
import pandas_ta as ta




ticker = yf.Ticker("AAPL")

for interval in ['1m', '1d', '5d', '1wk', '1mo', '3mo']:

    
    df = ticker.history(period="max", interval=interval)
    # print(df.head(5))


    if df.index.name == 'Date':

        df = df.rename_axis('Datetime')

        df.index = pd.to_datetime(df.index)

    # if 'Dividends' in df.columns and 'Stock Splits' in df.columns:
    #     df.drop(columns=['Dividends','Stock Splits'], inplace=True)  

    macd = df.ta.macd(fast=12, slow=26, signal=9)
    print(macd)
    df['MACD'] = macd['MACD_12_26_9']
    df['MACD_Signal'] = macd['MACDs_12_26_9']
    df['MACD_Hist'] = macd['MACDh_12_26_9']


    print(interval, df.tail(5))