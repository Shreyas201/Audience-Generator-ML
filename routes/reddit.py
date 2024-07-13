## importing all necessary libraries
from flask import Blueprint, request, render_template, redirect, url_for, session
import requests
from math import ceil
import re
from math import ceil
from datetime import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import string
import nltk
nltk.download('punkt')
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import requests
import unicodedata
from nltk.stem import PorterStemmer
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
from scipy.special import softmax
import operator
import praw

reddit_routes = Blueprint('reddit_routes', __name__)

MODEL=f"cardiffnlp/twitter-roberta-base-sentiment"
tokenizer=AutoTokenizer.from_pretrained(MODEL)
model=AutoModelForSequenceClassification.from_pretrained(MODEL)

# Initialize PRAW
reddit = praw.Reddit(
    client_id='X4FiAmYWrTMYh75irHpzDg',
    client_secret='-ydtKNqrxp2opc5M-K6vaQ9Jh3lojg',
    user_agent='AMWTech'
)

@reddit_routes.route('/reddit_search', methods=['GET','POST'])
def reddit_search():
    # if request.method == 'POST':
    #     session['keyword'] = request.form.get('keyword', '')
    #     return redirect(url_for('twitter_routes.twitter_search', page=1))

    keyword = session.get('final_keywords', '')
    print(f"Keyword from session: {keyword}")  # Debugging line

    if 'get_users' in request.form:
        # Handle the "Get Users" button click
        try:
            reddit_dict, rUsers = get_rUsers(keyword)
            print(f"User dict: {reddit_dict}")  # Debugging line
            return render_template('reddit_users.html', users=rUsers)
        except ValueError as e:
            return str(e), 400

    if keyword:
        try:
            reddit_dict,rUsers = get_rUsers(keyword)
            print(f"Tweet dict: {reddit_dict}")  # Debugging line
            total_results = len(reddit_dict)
            analyzed_comments = [{'username': k,'text': v[0], 'label': v[1]} for k, v in reddit_dict.items()]
            return render_template('reddit_search.html', tweets=analyzed_comments, total_results=total_results)
        except ValueError as e:
            return str(e), 400
    else:
        return render_template('reddit_search.html')


def get_rUsers(keywords):
    rUsers = []
    tweet_dict = {}
    print("keywords for audience generator with Reddit:::::::", keywords)

    # Process only the first 5 keywords
    for keyword in keywords[:5]:
        search_query = keyword
        print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>", search_query)
        rthreads_response = generate_threads(search_query)


        if rthreads_response:
            rdf = pd.DataFrame(rthreads_response, columns=['title', 'url', 'author', 'upvotes', 'num_comments', 'created'])

            rdf['title'] = rdf['title'].apply(lambda x: x.lower())
            stopword = stopwords.words('english')
            stemmer = nltk.PorterStemmer()

            rdf['rThreads_title'] = rdf['title'].replace(r'(@\w+|\"|\'|#\w+)', '', regex=True)
            rdf['rThreads_title'] = rdf['rThreads_title'].replace(r'https\S+|www\S+https\S+', '', regex=True)
            rdf['rThreads_title'] = rdf['rThreads_title'].replace(r'http\S+|www\S+https\S+', '', regex=True)

            rdf = rdf.drop_duplicates(subset='rThreads_title', keep="first")
            rdf['final_rThreads'] = rdf['rThreads_title'].replace(r'[^\x00-\x7F]+', '', regex=True)
            rdf['final_rThreads'] = rdf['final_rThreads'].apply(remove_non_english)

            print("::::::::::::::::::::::::::::::::::::::::::")
            rdf['tokenized_rThreads'] = rdf['final_rThreads'].apply(data_processing)
            rdf['stemm_rThreads'] = rdf['tokenized_rThreads'].apply(lambda a: stemming(a))
            rdf['robertaSentimentScore'] = rdf['stemm_rThreads'].apply(sentimentRoberta)

            pydf = rdf[rdf["robertaSentimentScore"] == "positive"]

            for index, row in rdf.iterrows():
                tweet_dict[row['author']] = [row['title'], row['robertaSentimentScore']]

            for index, row in pydf.iterrows():
                rUsers.append(row['author'])
        elif not rthreads_response: # this line will skip the iteration if there are no threads related to the keyword
            continue

    return tweet_dict, rUsers

def generate_threads(search_query):
    reddit = praw.Reddit(client_id='X4FiAmYWrTMYh75irHpzDg', client_secret='-ydtKNqrxp2opc5M-K6vaQ9Jh3lojg', user_agent='AMWTech')
    results = list(reddit.subreddit('all').search(search_query, time_filter='all', limit=None))
    rthreads_response = []
    try:
        for idx, submission in enumerate(results):
            post_info = {
                'title': submission.title,
                'url': submission.url,
                'author': str(submission.author),
                'upvotes': submission.ups,
                'num_comments': submission.num_comments,
                'created': datetime.fromtimestamp(submission.created).strftime('%Y-%m-%d %H:%M:%S')
            }
            rthreads_response.append(post_info)
        return rthreads_response
    except KeyError:
        return rthreads_response

def remove_non_english(arr):
    return ''.join(i for i in arr if unicodedata.category(i) == 'Lu' or unicodedata.category(i) == 'Ll')

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
