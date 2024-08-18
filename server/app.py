import streamlit as st
import scraping, plots
import database

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

    un_inicaitors = ['SMA20', 'SMA50', 'SMA100', 'SMA150', 'SMA200', 'EMA20', 'MACD', 'RSI', 'ADX']
    bar = 50
    if 'SMA100' in un_inicaitors:
        bar = 100
    elif 'SMA150' in un_inicaitors:
        bar = 150
    elif 'SMA200' in un_inicaitors:
        bar = 200
    intervals = ['1m', '2m', '5m', '15m', '30m', '60m', '90m', '1h', '1d', '5d', '1wk', '1mo', '3mo']
    interval = st.sidebar.selectbox('Interval', intervals, index=8)

    
    if interval == '1m':
        un_inicaitors = ['SMA20', 'SMA50', 'SMA100', 'SMA150', 'SMA200', 'EMA20', 'RSI', 'ADX']
    inicaitors= st.sidebar.multiselect('Inicaitors', un_inicaitors)

    min_bar = 1
    if intervals.index(interval) < 7:
        min_bar = 30
    days = st.sidebar.slider('Days', min_bar, 1000, bar)
    df = scraping.get_stock_data(stock_ticker,DAYS=days , interval=interval)
    if df is None:
        st.error(f"No data available for {stock_ticker} at {interval} interval.")
        return

    # Ensure 'High', 'Low', and 'Close' columns are present in the DataFrame
    if not all(col in df.columns for col in ['High', 'Low', 'Close']):
        st.error(f"Required columns are missing in the data for {stock_ticker}.")
        return
    
    st.plotly_chart(plots.plot_stock(df, stock_ticker, inicaitors, show='all', interval=interval))

    compeny = database.get_compeny(stock_ticker)
    if compeny:
        st.write(f"Name: {compeny.compeny_name}")
        st.plotly_chart(compeny.show(interval, inicaitors))
        st.write(f"Location: {compeny.Location}")
        st.write(f"Founded: {compeny.Founded}")
        st.write(f"CIK: {compeny.CIK}")
        st.write(f"GICS Sector: {compeny.GICS_Sector}")
        st.write(f"GICS Sub-Industry: {compeny.GICS_Sub_Industry}")
        st.write(f"Price: {scraping.current_stock_price(stock_ticker)}")


def go_to_login():
    st.session_state['page'] = 'app'

if st.session_state['page'] == 'app':
    client_page()