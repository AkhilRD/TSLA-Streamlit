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

#describing the data
if st.checkbox('Data Description'):
    st.write(price.describe())

#plotting the open and close of TSLA 
def plot_tsla():
    fig = go.Figure()
    fig.add_trace(go.Scatter(x = price['Date'],y = price['Open'],name = 'TSLA Open'))
    fig.add_trace(go.Scatter(x = price['Date'],y = price['Close'],name = 'TSLA Close'))
    fig.layout.update(title_text = 'TSLA Movement',xaxis_rangeslider_visible=True)

    st.plotly_chart(fig)

plot_tsla()


def tsla_ma30():
    fig = go.Figure()
    fig.add_trace(go.Scatter(x = price['Date'],y = price['Close'].rolling(30).mean(),name = 'TSLA 30 day Moving Average'))
    fig.add_trace(go.Scatter(x = price['Date'],y = price['Close'].rolling(60).mean(),name = 'TSLA 60 day Moving Average'))
    fig.add_trace(go.Scatter(x = price['Date'],y = price['Close'],name = 'TSLA Close'))
    fig.layout.update(title_text = 'TSLA Moving Average',xaxis_rangeslider_visible=True)

    st.plotly_chart(fig)

tsla_ma30() 

df = pd.read_csv('https://raw.githubusercontent.com/AkhilRD/Foundational_Project/pre-mid-eval/TSLARATIOS.csv')
df['Date'] = df['Date'].astype('datetime64[ns]')
df = df.sort_values('Date',ascending=True)



def tesla_metrics():
    fig = go.Figure()
    fig.add_trace(go.Scatter(x = df['Date'],y = df['Gross Margin'],name = 'TSLA Gross Margin'))
    fig.add_trace(go.Scatter(x = df['Date'],y = df['Operating Margin'],name = 'TSLA Operating Margin'))
    fig.add_trace(go.Scatter(x = df['Date'],y = df['Net Profit Margin'],name = 'TSLA Net Profit Margin'))
    fig.layout.update(title_text = 'TSLA Profit Margins',xaxis_rangeslider_visible=True)

    st.plotly_chart(fig)

tesla_metrics()

def tesla_debt():
    fig = go.Figure()
    fig.add_trace(go.Scatter(x = df['Date'],y = df['Long-term Debt / Capital'],name = 'TSLA Long term debt to capital'))
    fig.add_trace(go.Scatter(x = df['Date'],y = df['Debt/Equity Ratio'],name = 'TSLA Debt to Equity'))
    fig.layout.update(title_text = 'TSLA Debt Ratios',xaxis_rangeslider_visible=True)

    st.plotly_chart(fig)

tesla_debt()

def tesla_equity():
    fig = go.Figure()
    fig.add_trace(go.Scatter(x = df['Date'],y = df['ROE - Return On Equity'],name = 'TSLA Return on Equity'))
    fig.add_trace(go.Scatter(x = df['Date'],y = df['ROA - Return On Assets'],name = 'TSLA Return on Assets '))
    fig.add_trace(go.Scatter(x = df['Date'],y = df['Debt/Equity Ratio'],name = 'TSLA Return On Investment'))
    fig.layout.update(title_text = 'TSLA Return Ratios',xaxis_rangeslider_visible=True)

    st.plotly_chart(fig)

tesla_equity()