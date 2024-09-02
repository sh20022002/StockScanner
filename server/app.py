import streamlit as st
import scraping, plots
import database
from datetime import datetime

if 'page' not in st.session_state:
        st.session_state['page'] = 'app'

def client_page():
    """
    This function displays the client page of the SmartTraid application.
    
    Parameters:
    - user: User object representing the current user
    - compenies: List of available stock companies
    
    Returns:
    None
    """
    
    names, symbols = scraping.get_stocks()
    # Check if df is None
    

    st.title("SmartTraid")
    st.title("The Future of Trading.")
    st.sidebar.title("Analyze Stock")  
    st.sidebar.title("s&p500")
    stock_name = st.sidebar.selectbox("stock", names)
    sindex = names.index(stock_name)
    stock_ticker = symbols[sindex]
    st.sidebar.write(f"Stock Ticker: {stock_ticker}")

    un_inicaitors = ['SMA20', 'SMA50', 'SMA100', 'SMA150', 'SMA200', 'EMA20', 'MACD', 'RSI']
    # bar = 50
    # if 'SMA100' in un_inicaitors:
    #     bar = 100
    # elif 'SMA150' in un_inicaitors:
    #     bar = 150
    # elif 'SMA200' in un_inicaitors:
    #     bar = 200
    intervals = ['1m', '1h', '1d', '5d', '1wk', '1mo', '3mo']
    interval = st.sidebar.selectbox('Interval', intervals, index=2)

    
    if interval != '1m' or interval != '1h':
        un_inicaitors = ['SMA20', 'SMA50', 'SMA100', 'SMA150', 'SMA200', 'EMA20', 'RSI']
    inicaitors= st.sidebar.multiselect('Inicaitors', un_inicaitors)
    max_bar = 1000
    min_bar = 1
   
        
    if interval == '1m':
        max_bar = 2
        min_bar = 1
    elif interval == '1h':
        max_bar = 7
        min_bar = 1
    median_bar = round((min_bar + max_bar) // 2)
    days = st.sidebar.slider('Days', min_bar, max_bar, median_bar)
    df, max_key, summery, divid, info = scraping.get_stock_data(stock_ticker,DAYS=days , interval=interval)
    # st.write(df.head())
    if df is None:
        st.error(f"No data available for {stock_ticker} at {interval} interval.")
        return

    # Ensure 'High', 'Low', and 'Close' columns are present in the DataFrame
    if not all(col in df.columns for col in ['High', 'Low', 'Close']):
        st.error(f"Required columns are missing in the data for {stock_ticker}.")
        return
    
    st.plotly_chart(plots.plot_stock(df, stock_ticker, inicaitors, show='all', interval=interval))


    
    st.write(f"Summary: {summery}")
    st.write(f"\n Recommendation: {max_key}     Dividend: {divid}")
    text = ''
    n = 0
    for key, value in info.items():
        
        if n == 2:
            st.text(text)
            text = ''
            n = 0
        elif n == 1:
            if len(text) < 45:
                f = 45 - len(text)
                text += ' '*f
                
        
        if key not in ['longBusinessSummary', 'recommendationKey', 'lastDividendDate', 'lastDividendValue', 'trailingPegRatio', 'financialCurrency', 'uuid', 'timeZoneShortName', 'timeZoneFullName', 'firstTradeDateEpochUtc',  'lastDividendDate', 'lastDividendValue', 'lastSplitDate', 'lastSplitFactor', 'irWebsite', 'compensationAsOfEpochDate', 'governanceEpochDate', 'companyOfficer', 'longBusinessSummary', 'sectorDisp', 'sectorKey', 'industryDisp', 'industryKey', 'address1', 'companyOfficers', 'currency', 'exchange', 'quoteType', 'symbol', 'underlyingSymbol', 'shortName', 'longNam', 'messageBoardId', 'gmtOffSetMilliseconds']:
            if key in ['mostRecentQuarter', 'nextFiscalYearEnd', 'lastFiscalYearEnd']:
                t = datetime.fromtimestamp(value)
                text += f"{key}: {value}    "
            else:
                text += f"{key}: {value}    "

            n += 1


def go_to_login():
    st.session_state['page'] = 'app'

if st.session_state['page'] == 'app':
    client_page()