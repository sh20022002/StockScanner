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

    inicaitors = st.sidebar.multiselect('Inicaitors', ['Volume', 'SMA20', 'SMA50', 'SMA100', 'SMA150', 'EMA', 'ADX', 'RSI', 'MCAD'])
    interval = st.sidebar.radio('Interval', ['Day', 'Hour'])
    days = st.sidebar.slider('Days', 1, 1000, 100)
    if interval == 'Day':
        interval = '1d'
    else:
        interval = '1h'
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