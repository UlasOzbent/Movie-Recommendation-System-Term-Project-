from flask import Flask, flash, redirect, render_template,request,url_for
from flask_wtf import FlaskForm
from wtforms import StringField, SubmitField, PasswordField, TextAreaField
from wtforms.validators import DataRequired
from werkzeug.utils import secure_filename
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors
from keras.models import load_model
import h5py
import json

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import numpy as np
import tensorflow as tf
import keras




app = Flask(__name__, template_folder='templates', static_folder='static')
app.secret_key = "secret_key"


class MovieForm(FlaskForm):
	movie_name = StringField("Movie Name", validators=[DataRequired()])
	submit = SubmitField("Submit")





df_movies = pd.read_csv('movies.csv')
df_ratings = pd.read_csv('ratings.csv')
df_links = pd.read_csv('links.csv')

df_links['imdbId'] = df_links['imdbId'].apply(lambda x: f"https://www.imdb.com/title/tt{x}/?ref_=nv_sr_srsg_0")



#seperate genre for each movie and count genres
sss = df_movies['genres'].str.split(pat='|', expand=True).fillna(0)
sss.columns = ['genre1', 'genre2', 'genre3', 'genre4', 'genre5', 'genre6', 'genre7', 'genre8', 'genre9', 'genre10']
cols = sss.columns
sss[cols] = sss[cols].astype('category')
ss1 = sss.copy()
cat_columns = ss1.select_dtypes(['category']).columns

#count genres (non zeros)
ss1[cat_columns] = ss1[cat_columns].apply(lambda x: x.cat.codes)
ss1['genre_count'] = ss1[cols].gt(0).sum(axis=1) #count greater than 0 values for less than: df[cols].lt(0).sum(axis=1), for equal==0: df[cols].eq(0).sum(axis=1)

#assigning everything to same dataframe
df_movies['genre_count'] = ss1['genre_count']
df_movies[cols] = sss[cols]

data = pd.merge(df_movies,df_ratings)
data.drop(["genres","timestamp"], axis = 1,inplace = True)

data_pivot2 = pd.pivot(index = "movieId", columns = "userId", data = df_ratings, values = "rating")
data_pivot2.fillna(0, inplace = True)

csr_data = csr_matrix(data_pivot2.values)
data_pivot2.reset_index(inplace=True)

knn = NearestNeighbors(metric='cosine', algorithm='brute', n_neighbors=20, n_jobs=-1)
knn.fit(csr_data)

rating_avg = data.groupby('movieId')['rating'].mean().reset_index()
rating_avg = pd.DataFrame(rating_avg)

def movie_recommendation(movie_name):
    
    n_movies_to_reccomend = 10
    movie_list = data[data['title'].str.contains(movie_name, case=False)] # Fint movie name
    
    if len(movie_list):
        
        movie_idx= movie_list.iloc[0]['movieId']
        movie_idx = data_pivot2[data_pivot2['movieId'] == movie_idx].index[0]
        
        distances , indices = knn.kneighbors(csr_data[movie_idx],n_neighbors=n_movies_to_reccomend+1)    
        rec_movie_indices = sorted(list(zip(indices.squeeze().tolist(),distances.squeeze().tolist())),key=lambda x: x[1])[:0:-1]
        recommend_frame = []
        
        for val in rec_movie_indices:
            movie_idx = data_pivot2.iloc[val[0]]['movieId']
            #print(data.loc[data['title']==movie_idx])
            idx = data[data['movieId'] == movie_idx].index
            
            #print(data.iloc[idx]['rating'].values[0])
            

            #df_deneme = data.loc[data['title']==idx]

            recommend_frame.append({'Title':data.iloc[idx]['title'].values[0], 'Rating':round(data.iloc[idx]['rating'].mean(),1), 'Distance':val[1]})
        df = pd.DataFrame(recommend_frame,index=range(1,n_movies_to_reccomend+1)).iloc[::-1].reset_index()
        df.index = df.index + 1
        df.drop(["index"], axis = 1,inplace = True)
        return df
    
    else:
        return "No movies found. Please check your input"

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/recommend', methods=['POST'])

def recommend():
    
    df_flask = df_movies.copy()
    
    features = [str(x) for x in request.form.values()]
    print(features)
    movie_name = str(features[0])

    print(movie_name)
    
    if movie_name != df_flask['title'].any():
        print('yess')
    
    a = df_flask['title'].any()

    output = movie_recommendation(movie_name)
    #table_html = output.to_html()

    return render_template('index.html', table=output, movie_name=movie_name, a=a, df_flask=df_flask)

"""

with open('final.json') as f:
  final_data = json.load(f)

# Create a DataFrame from the data
df_final = pd.DataFrame(final_data)


@app.route('/refined_dataset')
def display_data():
  return render_template('refined.html', data=df_final.to_html())
"""
"""
@app.route('/run-cell1')
def run_cell1():
    cell1()
    return "Cell 1 ran successfully"
    """
"""
@app.route('/input')
def input_form():
  return render_template('input.html')



@app.route('/predict', methods=['POST'])
def predict():
  # Get the input data from the request
  data = request.form['input']
  
  # Convert the data to an integer
  data = int(data)
  
  # Save the data to a file
  with open('input.txt', 'w') as f:
    f.write(str(data))
    
  return render_template('predict_value.html')
"""

if __name__ == '__main__':
	app.run(host='127.0.0.1',debug=True)