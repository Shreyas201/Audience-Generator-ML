## importing all necessary libraries
import nltk
import praw
import tweepy
import string
import requests
import operator
import unicodedata
import numpy as np
import pandas as pd
from math import ceil
from datetime import datetime
import matplotlib.pyplot as plt
from scipy.special import softmax
from colorama import Fore, Back, Style
from nltk.tokenize import word_tokenize
from nltk.tokenize import TweetTokenizer
from nltk.corpus import stopwords
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from flask import Blueprint, request, render_template, redirect, url_for, session
#nltk.download('punkt')

all_routes = Blueprint('all_routes', __name__)

MODEL=f"cardiffnlp/twitter-roberta-base-sentiment"
tokenizer=AutoTokenizer.from_pretrained(MODEL)
model=AutoModelForSequenceClassification.from_pretrained(MODEL)

# for twitter
BEARER_TOKEN="AAAAAAAAAAAAAAAAAAAAADh5mwEAAAAABdHgxUgpqpVQaXVzudfI%2Fn0AXpk%3DW9uQ3vp4z8NqvtJpFN1De2ULYRhiLXfK5iGeghpEvEOTyvVNzn"
client = tweepy.Client(bearer_token=BEARER_TOKEN)

# for youtube
api_key = 'AIzaSyCYXRjbqEwAjtWH3W_WJmOFj0Xe2ZxvzOs'

# Initialize PRAW
reddit = praw.Reddit(
    client_id='X4FiAmYWrTMYh75irHpzDg',
    client_secret='-ydtKNqrxp2opc5M-K6vaQ9Jh3lojg',
    user_agent='AMWTech'
)

@all_routes.route('/all_search', methods=['GET','POST'])
def all_search():
    # if request.method == 'POST':
    #     session['keyword'] = request.form.get('keyword', '')
    #     return redirect(url_for('twitter_routes.twitter_search', page=1))

    keyword = session.get('final_keywords', '')
    print(f"Keyword from session: {keyword}")  # Debugging line

    if 'get_users' in request.form:
        # Handle the "Get Users" button click
        try:
            all_dict = {}
            allUsers = []

            # calling dictionaries and users list from their respectve python file
            reddit_dict, rUsers = get_rUsers(keyword)
            youtube_dict, yUsers = get_yUsers(keyword)
            twitter_dict, tUsers = get_tUsers(keyword)

            # merging all 3 dictionaries into 1
            all_dict.update(reddit_dict)
            all_dict.update(youtube_dict)
            all_dict.update(twitter_dict)

            # merging users into single lists
            allUsers = rUsers + yUsers + tUsers

            print(f"User dict: {all_dict}")  # Debugging line
            return render_template('all_users.html', user1=yUsers, user2=rUsers, user3=tUsers)
        
        except ValueError as e:
            return str(e), 400

    if keyword:
        try:
            all_dict = {}
            allUsers = []

            # calling dictionaries and users list from their respectve python file
            reddit_dict, rUsers = get_rUsers(keyword)
            youtube_dict, yUsers = get_yUsers(keyword)
            twitter_dict, tUsers = get_tUsers(keyword)

            # merging all 3 dictionaries into 1
            all_dict.update(reddit_dict)
            all_dict.update(youtube_dict)
            all_dict.update(twitter_dict)

            print(f"Tweet dict: {all_dict}")  # Debugging line
            total_results = len(all_dict)
            analyzed_comments = [{'username': k,'text': v[0], 'label': v[1]} for k, v in all_dict.items()]
            return render_template('all_search.html', tweets=analyzed_comments, total_results=total_results)
        
        except ValueError as e:
            return str(e), 400
    else:
        return render_template('all_search.html')

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

def get_yUsers(keywords):
    yUsers = []
    tweet_dict={}
    print("keywords for audience generator with Youtube Comments:::::::",keywords)

    # Process only the first 5 keywords
    for keyword in keywords[:5]:
        search_query = keyword
        search_query=search_query+" -is:youtube comment lang:en"
        print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>",search_query)
        youtube_comments_response=generate_youtube_comments(search_query)


        if youtube_comments_response:
            ydf = pd.DataFrame(youtube_comments_response,columns=['Username', 'Comment', 'Video_Link'])

            ydf['Comment']= ydf['Comment'].apply(lambda x: x.lower())
            stopword = stopwords.words('english')
            stemmer = nltk.PorterStemmer()

            ydf['comment_title'] = ydf['Comment'].replace(r'(@\w+|\"|\'|#\w+)','', regex=True) # removes handlers(like '@') and Hashtags (like '#')
            ydf['comment_title']=ydf['comment_title'].replace(r'https\S+|www\S+https\S+','',regex=True) # removes URLs
            ydf['comment_title']=ydf['comment_title'].replace(r'http\S+|www\S+https\S+','',regex=True) # removes URLs

            ydf = ydf.drop_duplicates(subset='Comment', keep="first")  # remove duplicates
            ydf['filtered_comment'] = ydf['comment_title'].replace(r'[^\x00-\x7F]+', '', regex=True) # removes non-english words
            ydf['final_comment'] = ydf['filtered_comment'].apply(remove_non_english) # removes threads with no english words

            print("::::::::::::::::::::::::::::::::::::::::::")
            ydf['tokenized_comment']=ydf['final_comment'].apply(data_processing) # removes punctuations, stop words, and non-aplhanumeric characters
            ydf['stemm_comment'] = ydf['tokenized_comment'].apply(lambda a: stemming(a))
            ydf['robertaSentimentScore'] = ydf['stemm_comment'].apply(sentimentRoberta)

            pydf = ydf[ydf["robertaSentimentScore"]=="positive"]

            for index, row in ydf.iterrows():
                tweet_dict[row['Username']]=[row['Comment'],row['robertaSentimentScore']]

            for index, row in pydf.iterrows():
                yUsers.append(row['Username'])
        elif not youtube_comments_response: # this line will skip the iteration if there are no youutube comments related to the keyword
            continue

    return tweet_dict, yUsers 


def generate_youtube_comments(search_query):
    api_key = 'AIzaSyCYXRjbqEwAjtWH3W_WJmOFj0Xe2ZxvzOs'
    search_url = 'https://www.googleapis.com/youtube/v3/search'
    search_params = {
    'part': 'snippet',
    'q': search_query,
    'key': api_key,
    'maxResults': 10
    }
    search_response = requests.get(search_url, params=search_params)
    if search_response.status_code != 200:
        print(f'Error: {search_response.status_code}')
        return []


    search_data = search_response.json()
    youtube_comments = []
    try:
        for item in search_data['items']:
            if item['id'].get('videoId'):
                video_id = item['id']['videoId']
                video_link = f'https://www.youtube.com/watch?v={video_id}'
                comments_url = 'https://www.googleapis.com/youtube/v3/commentThreads'
                comments_params = {
                    'part': 'snippet',
                    'videoId': video_id,
                    'key': api_key,
                    'maxResults': 50
                }

                comments_response = requests.get(comments_url, params=comments_params)
                comments_data = comments_response.json()

                    # Process the comments data
                for item in comments_data['items']:
                    comment = item['snippet']['topLevelComment']['snippet']['textDisplay']
                    username = item['snippet']['topLevelComment']['snippet']['authorDisplayName']
                    post_info = {'Username': username, 'Comment': comment,'Video_Link': video_link}
                    youtube_comments.append(post_info)
        return youtube_comments
    except KeyError:
        return youtube_comments

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
