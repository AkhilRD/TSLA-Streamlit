import streamlit as st
import pandas as pd
import pickle
from datetime import datetime
from datetime import date
import snscrape.modules.twitter as sntwitter
from nltk.tokenize import sent_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import unicodedata
import nltk
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import re
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import yfinance as yf
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import TruncatedSVD as svd

st.header('Prediction Kernel')
# loaded_model = pickle.load(open('/Users/akhil/Downloads/Foundational Project 1/tree.pkl','rb'))


# st.write(loaded_model)



# @st.cache(suppress_st_warning=True)
# def crux():
#     START = '2020-06-27'
#     TODAY = date.today().strftime("%Y-%m-%d")

#     stocks = ('TSLA')


#     #Writing a function to download data
#     def load_data(ticker):
#         price = yf.download(ticker,START,TODAY)
#         price.reset_index(inplace = True)
#         return price

#     price = load_data(stocks)

#     #Reading past 2 years of tweets from Git

#     data = pd.read_csv('https://raw.githubusercontent.com/AkhilRD/Foundational_Project/pre-mid-eval/TSLATWEETS.csv',
#     usecols = ['Date','Tweet'])
#     data['Date']= data['Date'].str.split(" ").str[0].astype('datetime64[ns]') #Normalizing datetime column
#     data['Date'] = data['Date'].dt.date 


#     query = "tesla Tesla TSLA TESLA min_faves:100 lang:en since:2022-08-09"
#     tweets = []

#     for tweet in sntwitter.TwitterSearchScraper(query).get_items():  #Scraping additional tweets from 2022-08-09
#         try:
#             tweets.append([tweet.date,tweet.content])
#         except:
#             continue

#     df = pd.DataFrame(tweets,columns = ['Date','Tweet'])             #Normalizing datetime column
#     df['Date'] = df['Date'].dt.date

#     tesla = pd.concat([df, data[:]]).reset_index(drop = True)        # Concatenating Git csv and live scraped tweets
#     tesla['Date'] = tesla['Date'].astype('datetime64[ns]')


#     #Reading the last 1000 (updated on 10/08/2022) hosted on Git

#     reddit_df = pd.read_csv('https://raw.githubusercontent.com/AkhilRD/Foundational_Project/pre-mid-eval/reddit.csv',usecols = ['created','title','body'])
#     reddit_df['created']= reddit_df['created'].str.split(" ").str[0].astype('datetime64[ns]') #Normalizing datetime column
#     reddit_df['created'] = reddit_df['created'].dt.date
#     reddit_df['created'] = reddit_df['created'].astype('datetime64[ns]')
#     reddit_df.info()

#     #Combining title and body columns

#     reddit_df['full_text'] = reddit_df['title'].fillna('') + ' '+ reddit_df['body'].fillna('')
#     reddit_df.head()

#     ratios = pd.read_csv('https://raw.githubusercontent.com/AkhilRD/Foundational_Project/pre-mid-eval/tsla_ratios.csv')
#     ratios['Date'] = ratios['Date'].astype('datetime64[ns]')
#     ratios

#     ratios['quarter'] = ratios['Date'].dt.quarter
#     ratios['year'] = ratios['Date'].dt.year
#     ratios['quarter_year'] = ratios.quarter.astype(str) + ' ' + ratios.year.astype(str)
#     ratios.drop(['quarter','year'],axis = 1,inplace=True)

#     final = price.merge(tesla,on = 'Date',how = 'left')
#     final_reddit = final.merge(reddit_df,left_on='Date',right_on='created',how = 'left')
#     final_reddit.drop(['created','title','body'],axis = 1,inplace=True)
#     final = final_reddit
#     final['text'] = final['Tweet'].fillna('') +' '+ final['full_text'].fillna('')
#     final.drop(['Tweet','full_text'],axis = 1,inplace = True)

#     final['quarter'] = final['Date'].dt.quarter
#     final['year'] = final['Date'].dt.year
#     final['quarter_year'] = final.quarter.astype(str) + ' ' + final.year.astype(str)
#     final.drop(['quarter','year'],axis=1,inplace=True)

#     final_tesla = final.merge(ratios,on='quarter_year',how = 'left')
#     final_tesla.drop(['Date_y','quarter_year'],axis=1,inplace=True)
#     final_tesla.rename(columns={'Date_x':'Date'},inplace=True)

#     final_tesla['text'] = final_tesla.groupby(['Date'])['text'].transform(
#                                                 lambda x: ' '.join(x))
#     tesla = final_tesla.drop_duplicates() 
#     tesla= tesla.groupby('Date').first()

#     #Text pre-processing

#     def preprocessing(text):
#         stop = nltk.corpus.stopwords.words('english')
#         lem = WordNetLemmatizer()
#         text = (unicodedata.normalize('NFKD', text).encode('ascii', 'ignore')
#         .decode('utf-8', 'ignore')
#         .lower())
#         words = re.sub(r'[^\w\s]', '', text).split()
#         return [lem.lemmatize(w) for w in words if w not in stop]

#     tesla['text']=tesla.apply(lambda x: preprocessing(x['text']), axis=1)
#     def final(lem_col):
#         return (" ".join(lem_col))

#     tesla['text'] = tesla.apply(lambda x: final(x['text']),axis=1)

#     sent_analyzer = SentimentIntensityAnalyzer()
#     cs = []
#     def senti(text):
#         for row in range(len(text)):
#             cs.append(sent_analyzer.polarity_scores((text).iloc[row])['compound'])

#     senti(tesla['text'])
#     tesla['sentiment_score'] = cs
#     tesla = tesla[(tesla[['sentiment_score']] != 0).all(axis=1)]
 
# crux()


