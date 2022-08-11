import streamlit as st
import snscrape.modules.twitter as sntwitter
import pandas as pd
from datetime import date
from nltk.tokenize import sent_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import unicodedata
import nltk
from plotly import graph_objs as go
import re
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer



st.sidebar.title('Navigation')

st.title("TSLA Price Prediction 🚗⚡🔋")
st.subheader("Twitter Sentiment")

#reading the data and scraping using snscrape
data = pd.read_csv('https://raw.githubusercontent.com/AkhilRD/Foundational_Project/pre-mid-eval/TSLATWEETS.csv',
usecols = ['Date','Tweet'])

query = "tesla Tesla TSLA TESLA min_faves:100 lang:en since:2022-08-09"
tweets = []

for tweet in sntwitter.TwitterSearchScraper(query).get_items():
    tweets.append([tweet.date,tweet.content])

df = pd.DataFrame(tweets,columns = ['Date','Tweet'])             #Normalizing datetime column
df['Date'] = df['Date'].dt.date

tesla = pd.concat([df, data[:]]).reset_index(drop = True)        # Concatenating Git csv and live scraped tweets
tesla['Date'] = tesla['Date'].astype('datetime64[ns]')
tesla

# #editing the date time column to date format
# tesla['Date']= tesla['Date'].str.split(" ").str[0].astype('datetime64[ns]')
# tesla['Date'] = tesla['Date'].astype('datetime64[ns]')
# # tesla['Date'] = pd.to_datetime(tesla['Date'], format="%Y-%m-%d",utc=True)


# if st.checkbox('Display Tweets'):
#     st.write(tesla.head(50))

#pre-processing text
def preprocessing(text):
    stop = nltk.corpus.stopwords.words('english')
    lem = WordNetLemmatizer()
    text = (unicodedata.normalize('NFKD', text).encode('ascii', 'ignore')
    .decode('utf-8', 'ignore')
    .lower())
    words = re.sub(r'[^\w\s]', '', text).split()
    return [lem.lemmatize(w) for w in words if w not in stop]

tesla['text']=tesla.apply(lambda x: preprocessing(x['Tweet']), axis=1)
def final(lem_col):
    return (" ".join(lem_col))

tesla['text'] = tesla.apply(lambda x: final(x['text']),axis=1)
# st.write('Processed Tweets')
# tesla['text']


#converting text to sentiment scores
sent_analyzer = SentimentIntensityAnalyzer()
cs = []
def senti(text):
    for row in range(len(text)):
        cs.append(sent_analyzer.polarity_scores((text).iloc[row])['compound'])

senti(tesla['text'])
tesla['sentiment_score'] = cs
tesla = tesla[(tesla[['sentiment_score']] != 0).all(axis=1)]
# st.write('Polarity scores (-/+)')
# tesla['sentiment_score']

sent_score = tesla.groupby('Date')['sentiment_score'].mean().reset_index()

def tsla_sentiment():
    fig = go.Figure()
    fig.add_trace(go.Line(x = sent_score['Date'],y = sent_score['sentiment_score'],name = 'TSLA sentiment'))
    fig.layout.update(title_text = 'TSLA Sentiment Movement',xaxis_rangeslider_visible=True)

    st.plotly_chart(fig)

tsla_sentiment()

sent_score = sent_score.sort_values('Date',ascending=False)
st.write('Twitter Sentiment: 30 days')
st.write(sent_score.iloc[0:30,1].mean())

st.write('Twitter Sentiment: 7 days')
st.write(sent_score.iloc[0:7,1].mean())

    


