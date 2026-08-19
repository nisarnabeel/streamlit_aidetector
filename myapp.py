import datetime

import yfinance as yf
import streamlit as st

st.write("""
# Simple Stock Price App

Shown are the stock closing price and volume!

""")

# https://towardsdatascience.com/how-to-get-stock-data-using-python-c0de1df17e75
#define the ticker symbol
tickerSymbol = st.text_input("Enter a stock ticker symbol", "GOOGL")
start_date = st.date_input("Start date", datetime.date(2010, 5, 31))
end_date = st.date_input("End date", datetime.date(2020, 5, 31))

#get data on this ticker
tickerData = yf.Ticker(tickerSymbol)
#get the historical prices for this ticker
tickerDf = tickerData.history(start=start_date, end=end_date)
# Open	High	Low	Close	Volume	Dividends	Stock Splits

if tickerDf.empty:
    st.warning("No data found for this ticker/date range.")
else:
    st.line_chart(tickerDf.Close)
    st.line_chart(tickerDf.Volume)