#importing libraries
import streamlit as st
from datetime import date
import matplotlib.pyplot as plt
import pandas as pd
import yfinance as yf
from plotly import graph_objs as go

st.set_page_config(
    page_title="Hello",
    page_icon="👋",
)

#sidebar and title
st.sidebar.title('Navigation')
st.title("TSLA Price Prediction 🚗⚡🔋")

#Assigning start dates
START = '2020-06-26'
TODAY = date.today().strftime("%Y-%m-%d")

stocks = ('TSLA')

# n_days = st.slider("Days of prediction:", 1,30)

#Writing a function to download data
@st.cache
def load_data(ticker):
    price = yf.download(ticker,START,TODAY)
    price.reset_index(inplace = True)
    return price

price = load_data(stocks)


st.write('Ticker Data')
st.write(price.tail())

#plotting the open and close of TSLA 
def plot_tsla():
    fig = go.Figure()
    fig.add_trace(go.Scatter(x = price['Date'],y = price['Open'],name = 'TSLA Open'))
    fig.add_trace(go.Scatter(x = price['Date'],y = price['Close'],name = 'TSLA Close'))
    fig.layout.update(title_text = 'TSLA Movement',xaxis_rangeslider_visible=True)

    st.plotly_chart(fig)

plot_tsla()

#describing the data
if st.checkbox('Data Description'):
    st.write(price.describe())

def tsla_ma30():
    fig = go.Figure()
    fig.add_trace(go.Scatter(x = price['Date'],y = price['Close'].rolling(30).mean(),name = 'TSLA 30 day Moving Average'))
    fig.add_trace(go.Scatter(x = price['Date'],y = price['Close'].rolling(60).mean(),name = 'TSLA 60 day Moving Average'))
    fig.add_trace(go.Scatter(x = price['Date'],y = price['Close'],name = 'TSLA Close'))
    fig.layout.update(title_text = 'TSLA Moving Average',xaxis_rangeslider_visible=True)

    st.plotly_chart(fig)

tsla_ma30() 