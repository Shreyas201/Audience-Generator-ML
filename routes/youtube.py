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

MODEL=f"cardiffnlp/twitter-roberta-base-sentiment"
tokenizer=AutoTokenizer.from_pretrained(MODEL)
model=AutoModelForSequenceClassification.from_pretrained(MODEL)

youtube_routes = Blueprint('youtube_routes', __name__)

api_key = 'AIzaSyCYXRjbqEwAjtWH3W_WJmOFj0Xe2ZxvzOs' 


@youtube_routes.route('/youtube_search', methods=['GET','POST'])
def youtube_search():
    # if request.method == 'POST':
    #     session['keyword'] = request.form.get('keyword', '')
    #     return redirect(url_for('twitter_routes.twitter_search', page=1))

    keyword = session.get('final_keywords', '')
    print(f"Keyword from session: {keyword}")  # Debugging line

    if 'get_users' in request.form:
        # Handle the "Get Users" button click
        try:
            youtube_dict, yUsers = get_yUsers(keyword)
            print(f"User dict: {youtube_dict}")  # Debugging line
            return render_template('youtube_users.html', users=yUsers)
        except ValueError as e:
            return str(e), 400

    if keyword:
        try:
            youtube_dict,yUsers = get_yUsers(keyword)
            print(f"Tweet dict: {youtube_dict}")  # Debugging line
            total_results = len(youtube_dict)
            analyzed_comments = [{'username': k,'text': v[0], 'label': v[1]} for k, v in youtube_dict.items()]
            return render_template('youtube_search.html', tweets=analyzed_comments, total_results=total_results)
        except ValueError as e:
            return str(e), 400
    else:
        return render_template('youtube_search.html')

def get_yUsers(keywords):
    yUsers = []
    tweet_dict={}
    print("keywords for audience generator with Youtube Comments:::::::",keywords)

    # Process only the first 5 keywords
    for keyword in keywords[:5]:
        search_query = keyword
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
        youtube_comments = []
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

