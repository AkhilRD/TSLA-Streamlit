import streamlit as st
import pandas as pd
import pickle

st.header('Prediction Kernel')
loaded_model = pickle.load(open('/Users/akhil/Downloads/VR.pkl','rb'))

st.slider("Pick Days to Predict",0,30)



if st.button('Predict'):
    make_prediction = loaded_model.predict()
    output = round(make_prediction[0,2])
    st.success('You can buy/sell TSLA for {}'.format(output))