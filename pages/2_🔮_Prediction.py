from curses import erasechar
import streamlit as st
import pandas as pd
from plotly import graph_objs as go
import math
from datetime import date
import pickle
from sklearn.metrics import mean_squared_error
from plotly import graph_objs as go


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
with open('modelv2.pkl', 'rb') as f:
    loaded_regressor = pickle.load(f)

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



