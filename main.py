from flask import Flask, render_template, redirect, request,jsonify,session
from routes.all import all_routes
from routes.reddit import reddit_routes
from routes.youtube import youtube_routes
from routes.twitter import twitter_routes
from gensim.models import KeyedVectors
import random
import pandas as pd
# Load the saved model from disk
preTW2v_wv = KeyedVectors.load('preTW2v_wv.model')

app = Flask(__name__)
app.config['SECRET_KEY'] = '123456789'
app.register_blueprint(all_routes, url_prefix='/all')
app.register_blueprint(reddit_routes, url_prefix='/reddit')
app.register_blueprint(youtube_routes, url_prefix='/youtube')
app.register_blueprint(twitter_routes, url_prefix='/twitter')


@app.route('/')
def login_page():
    return render_template('login.html')

@app.route('/', methods=['POST'])
def login():
    username = request.form['username']
    password = request.form['password']
    if username == 'admin' and password == 'password':
        return redirect('/results')
    else:
        return 'Invalid login'

@app.route('/results')
def results_page():
    
    return render_template('results.html')

@app.route('/results', methods=['POST'])
def results():
    keywords = request.form['keywords']
    negative_keywords = request.form['neg_keywords']
    # print(keywords)
    # Generate similar keywords
    similar_keywords = get_similar_keywords(keywords, negative_keywords)
    # Save keywords to db
    initial_keywords = [value.strip() for value in keywords.split(',')]
    
    temp_final_keywords = list(set(similar_keywords) - set(initial_keywords))
    
    final_keywords = initial_keywords + temp_final_keywords
    # final_keywords = list(set(final_keywords))
    #negative_keywords = [value.strip() for value in negative_keywords.split(',')]
    session['final_keywords'] = final_keywords
    return redirect('/finalize_keywords')

@app.route('/finalize_keywords', methods=['GET'])
def finalize_keywords():
    final_keywords = session.get('final_keywords', [])  # Retrieve final_keywords from the session
    return render_template('keywords.html', final_keywords=final_keywords)  # Pass final_keywords to the template



@app.route('/home', methods=['GET'])
def index():
    return render_template('index.html')

def get_similar_keywords(keywords, neg_keywords):
    #Extract the data form previous query into structured data format of pandas dataframe
    
    keywords = [value.strip() for value in keywords.split(',')]
    neg_keywords = [value.strip() for value in neg_keywords.split(',')]

    # Remove any keywords that are not found in the KeyedVectors model
    keywords_found = [keyword for keyword in keywords if keyword in preTW2v_wv.key_to_index]
    neg_keywords_found = [keyword for keyword in neg_keywords if keyword in preTW2v_wv.key_to_index]


    #handel error when keywords are not in vocab
    similar_keywords_and_scores = preTW2v_wv.most_similar(positive=keywords_found,
                              negative=neg_keywords_found,topn=100)
    
    similar_keywords_df = pd.DataFrame.from_dict(similar_keywords_and_scores, orient='columns')
    similar_keywords_df.columns=['keywords', 'similarity_score']
    
    #convert the simialr keywords to a lsit
    similar_keywords = similar_keywords_df['keywords']
    # similar_keywords = similar_keywords.tolist()
    similar_keywords = [keyword.lower() for keyword in similar_keywords]
    similar_keywords = list(set(similar_keywords))



    return similar_keywords


if __name__ == '__main__':
    app.run(debug=True)