from curses import erasechar
import streamlit as st
import yfinance as yf
import pandas as pd
from plotly import graph_objs as go
import math
from datetime import date
import pickle
from sklearn.metrics import mean_squared_error
from plotly import graph_objs as go
from statsmodels.tsa.stattools import adfuller
from pmdarima import auto_arima 
from statsmodels.tsa.arima_model import ARIMA,ARIMAResults


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
steps = 50
price_test  = price[-steps:]

st.header('Prediction Kernel')
y_true = price_test['Close']
x_test= pd.read_csv(r"https://raw.githubusercontent.com/AkhilRD/Foundational_Project/pre-mid-eval/x_test.csv",index_col='Date',parse_dates=True)

#OLS MODEL
st.subheader('Basic OLS Model')

# with open('model_OLS.pkl', 'rb') as f:
#     loaded_regressor = pickle.load(f)

loaded_regressor = pickle.load(open('model_OLS.pkl','rb'))

y_pred = loaded_regressor.predict(x_test)
error = math.sqrt(mean_squared_error(y_true,y_pred))
st.write('RMSE OLS:')
st.write(error)

#Voting Regressor
st.subheader('Voting Regressor')
st.write('Used Voting regressor to ensemble, hyper-tuned random forest, decision tree, cat boost model and K-fold cross validation for validation.')
loaded_regressor = pickle.load(open('modelv2.pkl','rb'))


y_pred = loaded_regressor.predict(x_test)
error = math.sqrt(mean_squared_error(y_true,y_pred))
st.write('RMSE')
st.write(error)

steps = 50
price_test  = price[-steps:]

VR_predict= pd.DataFrame(list(zip(y_true,y_pred)), columns = ['Actual', 'Predicted'], index =price_test["Date"]).reset_index()
VR_predict.head()
st.write(VR_predict)


#Chart
def plot_ts():
    fig = go.Figure()
    fig.add_trace(go.Scatter(x = VR_predict['Date'],y = VR_predict['Predicted'],name = 'TSLA Predicted'))
    fig.add_trace(go.Scatter(x = VR_predict['Date'],y = VR_predict['Actual'],name = 'TSLA Actual'))
    fig.layout.update(title_text = 'TSLA Prediction Chart',xaxis_rangeslider_visible=True)

    st.plotly_chart(fig)

plot_ts()

pred_y= VR_predict["Predicted"]
train_y= price["Close"]
combined = pd.concat([train_y, pred_y])
combined= pd.DataFrame(combined, columns =['Close'])
combined.head()

st.subheader('ARIMA Model Forecasts')
# st.write(auto_arima(combined['Close']).summary())

#Building Arima Model
arima_model = ARIMA(combined['Close'],order=(0,1,0))
model = arima_model.fit()
st.write(model.summary())

#Forecasting for 30 days in the future
forecast = model.predict(len(combined),len(combined)+30,typ='levels').rename('Forecast')
st.write(forecast)


