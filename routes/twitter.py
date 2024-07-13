import requests
import operator
from flask import Blueprint, request, render_template, redirect, url_for, session
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
from scipy.special import softmax
import json
import pandas as pd
import tweepy
import re
from nltk.tokenize import word_tokenize
import unicodedata
import matplotlib.pyplot as plt
import numpy as np
import nltk
from nltk.tokenize import TweetTokenizer
from nltk.corpus import stopwords
from nltk.sentiment.vader import SentimentIntensityAnalyzer
#nltk.download('vader_lexicon')
import string 
from colorama import Fore, Back, Style
import pandas as pd
import matplotlib.pyplot as plt
# from textblob import TextBlob
from nltk.sentiment.vader import SentimentIntensityAnalyzer
# import pyLDAvis.sklearn
# from wordcloud import WordCloud
import nltk
#nltk.download('punkt') 
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import math
import pandas as pd

twitter_routes = Blueprint('twitter_routes', __name__)

MODEL=f"cardiffnlp/twitter-roberta-base-sentiment"
tokenizer=AutoTokenizer.from_pretrained(MODEL)
model=AutoModelForSequenceClassification.from_pretrained(MODEL)

BEARER_TOKEN="AAAAAAAAAAAAAAAAAAAAADh5mwEAAAAABdHgxUgpqpVQaXVzudfI%2Fn0AXpk%3DW9uQ3vp4z8NqvtJpFN1De2ULYRhiLXfK5iGeghpEvEOTyvVNzn"
client = tweepy.Client(bearer_token=BEARER_TOKEN)

@twitter_routes.route('/twitter_search', methods=['GET','POST'])
def twitter_search():
    # if request.method == 'POST':
    #     session['keyword'] = request.form.get('keyword', '')
    #     return redirect(url_for('twitter_routes.twitter_search', page=1))

    keyword = session.get('final_keywords', '')
    print(f"Keyword from session: {keyword}")  # Debugging line

    if 'get_users' in request.form:
        # Handle the "Get Users" button click
        try:
            twitter_dict, tUsers = get_tUsers(keyword)
            print(f"User dict: {twitter_dict}")  # Debugging line
            return render_template('twitter_users.html', users=tUsers)
        except ValueError as e:
            res = 'fun1 not working'
            return res, 400
    
    if keyword:
        try:
            twitter_dict, tUsers = get_tUsers(keyword)
            print(f"Tweet dict: {twitter_dict}")  # Debugging line
            total_results = len(twitter_dict)
            analyzed_tweets = [{'username': k,'text': v[0], 'label': v[1]} for k, v in twitter_dict.items()]
            return render_template('twitter_search.html', tweets=analyzed_tweets, total_results=total_results)
        except ValueError as e:
            res = 'fun2 not working'
            return res, 400
    else:
        return render_template('twitter_search.html')

def get_tUsers(keywords):
    tUsers = []
    tweet_dict = {}
    print("keywords for audience generator with Twitter:::::::", keywords)
    
    # Process only the first 3 keywords
    for keyword in keywords[:2]:
        search_query = f'"{keyword}" -is:retweet lang:en'
        print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>", search_query)
        tweets_response = generate_tweets(search_query)
        author = get_tweets_authors(tweets_response)
        if tweets_response:
            tdf = pd.DataFrame(author)
            
            tdf['text'] = tdf['text'].apply(lambda x: x.lower())
            stopword = stopwords.words('english')
            stemmer = nltk.PorterStemmer()
            
            tdf['tweets_text'] = tdf['text'].replace(r'(@\w+|\"|\'|#\w+)', '', regex=True)
            tdf['tweets_text'] = tdf['tweets_text'].replace(r'https\S+|www\S+https\S+', '', regex=True)
            tdf['tweets_text'] = tdf['tweets_text'].replace(r'http\S+|www\S+https\S+', '', regex=True)
            
            tdf = tdf.drop_duplicates(subset='text', keep="first")
            tdf['filtered_tweets'] = tdf['tweets_text'].replace(r'[^\x00-\x7F]+', '', regex=True)
            tdf['final_tweets'] = tdf['filtered_tweets'].apply(remove_non_english)
            
            print("::::::::::::::::::::::::::::::::::::::::::")
            tdf['tokenized_tweets'] = tdf['final_tweets'].apply(data_processing)
            tdf['stemm_tweets'] = tdf['tokenized_tweets'].apply(lambda x: stemming(x))
            tdf['robertaSentimentScore'] = tdf['stemm_tweets'].apply(sentimentRoberta)
            
            pydf = tdf[tdf["robertaSentimentScore"] == "positive"]
            
            for index, row in tdf.iterrows():
                tweet_dict[row['username']] = [row['text'], row['robertaSentimentScore']]
            
            for index, row in pydf.iterrows():
                tUsers.append(row['username'])
        elif not tweets_response: # this line will skip the iteration if there are no tweets related to the keyword
            continue
    
    return tweet_dict, tUsers


def generate_tweets(search_query):
    if not search_query:
        raise ValueError("Search query cannot be empty")
    tweets_response = []
    for response in  tweepy.Paginator(client.search_recent_tweets,
                        query = search_query,
                        user_fields = ['username', 'public_metrics', 'description', 'location'],
                        tweet_fields = ['created_at', 'geo', 'public_metrics', 'text'],
                        expansions = "author_id",
                        max_results = 10, limit = 10):
        tweets_response.append(response)
    return tweets_response

def get_tweets_authors(tweets_response):
    result = []
    authors = {}
    try:
        for response in tweets_response:
            for user in response.includes["users"]:
                authors[user.id] = {'username': user.username,
                                    'followers': user.public_metrics["followers_count"],
                                    'tweets': user.public_metrics["tweet_count"],
                                    'description': user.description,
                                    'location': user.location
                                    }      
            for tweet in response.data:
                author_info = authors[tweet.author_id]
                result.append({
                                "author_id": tweet.author_id,
                                "username": author_info['username'],
                                "author_followers": author_info['followers'],
                                "author_tweets": author_info['tweets'],
                                "author_description": author_info['description'],
                                "author_location": author_info['location'],
                                "text": tweet.text,
                                "created_at": tweet.created_at,
                                "retweets": tweet.public_metrics['retweet_count'],
                                "replies": tweet.public_metrics['reply_count'],
                                "likes": tweet.public_metrics['like_count'],
                                "quote_count": tweet.public_metrics['quote_count']
                            })
        return result
    except KeyError:
        return result

def remove_non_english(text):
    return ''.join(c for c in text if unicodedata.category(c) == 'Lu' or unicodedata.category(c) == 'Ll')

def data_processing(arr):
    tokens = word_tokenize(arr)
    stopWords = stopwords.words('english')
    # remove puntuations
    table = str.maketrans('', '', string.punctuation)
    puncStripped = [w.translate(table) for w in tokens]

    nonAlpha = [word for word in puncStripped if word.isalpha()] # remove non alphabetic characters
    cleanArr = [word for word in nonAlpha if word not in stopWords] # remove stopwords
    
    return ' '.join(cleanArr)

def stemming(arr):
    stemmer = nltk.stem.PorterStemmer()
    stemmedArr = [stemmer.stem(word) for word in arr]
    return arr

def sentimentRoberta(arr):
    encodedText=tokenizer(arr, add_special_tokens=True, max_length=512, truncation=True, return_tensors='pt')
    result=model(**encodedText)
    polarScores=result[0][0].detach().numpy()
    polarScores=softmax(polarScores)
    scoresDict={
        'negative':polarScores[0],
        'neutral':polarScores[1],
        'positive':polarScores[2]
    }
    finalResult=max(scoresDict.items(), key=operator.itemgetter(1))[0]
    return finalResult