#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd


# In[3]:


df_movies = pd.read_csv('movies.csv')
df_ratings = pd.read_csv('ratings.csv')
df_links = pd.read_csv('links.csv')


# In[4]:


df_movies


# In[5]:


df_movies['genres'].value_counts().nlargest(100)


# In[6]:


df_movies.tail()


# In[7]:


df_ratings.head()


# In[8]:


df_links.tail()


# In[9]:


df_links['imdbId'] = df_links['imdbId'].apply(lambda x: f"https://www.imdb.com/title/tt{x}/?ref_=nv_sr_srsg_0")


# In[10]:


#"https://www.imdb.com/title/tt0114709/?ref_=nv_sr_srsg_0"


# In[11]:


df_movies.head()


# In[12]:


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


# In[13]:


(df_movies['genre10'] != 0).value_counts()


# In[14]:


df_movies.shape


# In[15]:


df_movies.tail()


# In[16]:


data = pd.merge(df_movies,df_ratings)


# In[17]:


data.head()


# In[18]:


data.drop(["genres","timestamp"], axis = 1,inplace = True)


# In[19]:


data.tail()


# In[20]:


df_deneme = data.loc[data['title']=='Batman Begins (2005)']

print(df_deneme['rating'].mean())


# In[21]:


df_deneme = data.loc[data['title']=='Oh, God! (1977)']

print(df_deneme['rating'].mean())


# In[22]:


df_deneme = data.loc[data['title']=='Inception (2010)']

print(df_deneme['rating'].mean())


# In[23]:


data_pivot2 = pd.pivot(index = "movieId", columns = "userId", data = df_ratings, values = "rating")
data_pivot2


# In[24]:


data_pivot2.fillna(0, inplace = True)
data_pivot2


# In[25]:


from scipy.sparse import csr_matrix
csr_data = csr_matrix(data_pivot2.values)
data_pivot2.reset_index(inplace=True)


# In[26]:


csr_data


# In[27]:


from sklearn.neighbors import NearestNeighbors
knn = NearestNeighbors(metric='cosine', algorithm='brute', n_neighbors=20, n_jobs=-1)
knn.fit(csr_data)


# In[28]:


rating_avg = data.groupby('movieId')['rating'].mean().reset_index()
rating_avg = pd.DataFrame(rating_avg)


# In[ ]:





# In[29]:


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
            
            print(data.iloc[idx]['rating'].values[0])
            

            #df_deneme = data.loc[data['title']==idx]

            recommend_frame.append({'Title':data.iloc[idx]['title'].values[0], 'Rating':round(data.iloc[idx]['rating'].mean(),1), 'Distance':val[1]})
        df = pd.DataFrame(recommend_frame,index=range(1,n_movies_to_reccomend+1)).iloc[::-1].reset_index()
        df.index = df.index + 1
        df.drop(["index"], axis = 1,inplace = True)
        return df
    
    else:
        return "No movies found. Please check your input"


# In[30]:


movie_recommendation('harry potter')


# In[31]:


genre = data.genre1.value_counts()
genre = pd.DataFrame(genre)
genre = genre.reset_index()
genre.rename({'index': 'genre', 'genre1':'Count'}, axis=1, inplace=True)


# In[32]:


import seaborn as sns
import matplotlib.pyplot as plt


# In[33]:


sns.barplot(x = genre.genre, y=genre.Count)
plt.xticks(rotation=90)
plt.show()


# In[34]:


import plotly.express as px


# In[35]:


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


# In[36]:


import tensorflow as tf
import keras
from pprint import pprint


# In[37]:


df_movies2 = pd.read_csv('movies.csv')


# In[38]:


merged_dataset = pd.merge(df_ratings, df_movies2, how='inner', on='movieId')
merged_dataset.head()


# In[39]:


refined_dataset = merged_dataset.groupby(by=['userId','title'], as_index=False).agg({"rating":"mean"})

refined_dataset.head()


# In[40]:


user_enc = LabelEncoder()
refined_dataset['user'] = user_enc.fit_transform(refined_dataset['userId'].values)
n_users = refined_dataset['user'].nunique()


# In[41]:


item_enc = LabelEncoder()
refined_dataset['movie'] = item_enc.fit_transform(refined_dataset['title'].values)
n_movies = refined_dataset['movie'].nunique()


# In[42]:


import numpy as np


# In[43]:


refined_dataset['rating'] = refined_dataset['rating'].values.astype(np.float32)
min_rating = min(refined_dataset['rating'])
max_rating = max(refined_dataset['rating'])
n_users, n_movies, min_rating, max_rating


# In[44]:


refined_dataset.head()


# In[45]:


X = refined_dataset[['user', 'movie']].values
y = refined_dataset['rating'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=50)
X_train.shape, X_test.shape, y_train.shape, y_test.shape


# In[46]:


n_factors = 150


# In[47]:


X_train_array = [X_train[:, 0], X_train[:, 1]]
X_test_array = [X_test[:, 0], X_test[:, 1]]


# In[48]:


X_train, X_train_array, X_train_array[0].shape


# In[49]:


y_train = (y_train - min_rating)/(max_rating - min_rating)
y_test = (y_test - min_rating)/(max_rating - min_rating)


# In[50]:


from tensorflow.python.keras.layers import Input, Dense, Embedding


# In[51]:


## Initializing a input layer for users
user = tf.keras.layers.Input(shape = (1,))

## Embedding layer for n_factors of users
u = tf.compat.v1.keras.layers.Embedding(n_users, n_factors, embeddings_initializer = 'he_normal', embeddings_regularizer = tf.keras.regularizers.l2(1e-6))(user)
u = tf.keras.layers.Reshape((n_factors,))(u)

## Initializing a input layer for movies
movie = tf.keras.layers.Input(shape = (1,))

## Embedding layer for n_factors of movies
m = tf.compat.v1.keras.layers.Embedding(n_movies, n_factors, embeddings_initializer = 'he_normal', embeddings_regularizer=tf.keras.regularizers.l2(1e-6))(movie)
m = tf.keras.layers.Reshape((n_factors,))(m)

## stacking up both user and movie embeddings
x = tf.keras.layers.Concatenate()([u,m])
x = tf.keras.layers.Dropout(0.05)(x)

## Adding a Dense layer to the architecture
x = tf.keras.layers.Dense(32, kernel_initializer='he_normal')(x)
x = tf.keras.layers.Activation(activation='relu')(x)
x = tf.keras.layers.Dropout(0.05)(x)

x = tf.keras.layers.Dense(16, kernel_initializer='he_normal')(x)
x = tf.keras.layers.Activation(activation='relu')(x)
x = tf.keras.layers.Dropout(0.05)(x)

## Adding an Output layer with Sigmoid activation funtion which gives output between 0 and 1
x = tf.keras.layers.Dense(9)(x)
x = tf.keras.layers.Activation(activation='softmax')(x)

## Adding a Lambda layer to convert the output to rating by scaling it with the help of available rating information
# x = tf.keras.layers.Lambda(lambda x: x*(max_rating - min_rating) + min_rating)(x)

## Defining the model
model = tf.keras.models.Model(inputs=[user,movie], outputs=x)
# optimizer = tf.keras.optimizers.Adam(lr=0.001)
# optimizer = tf.keras.optimizers.RMSprop(learning_rate=0.005,
    # rho=0.9, momentum=0.01, epsilon=1e-07)

## Compiling the model
# model.compile(loss='binary_crossentropy', optimizer = optimizer)
# model.compile(loss='mean_squared_error', optimizer = optimizer,metrics=['accuracy'])
model.compile(optimizer='sgd', loss=tf.keras.losses.SparseCategoricalCrossentropy(), metrics=['accuracy'])


# In[52]:


model.summary()


# In[53]:


reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.75, patience=3, min_lr=0.000001, verbose=1)

history = model.fit(x = X_train_array, y = y_train, batch_size=128, epochs=70, verbose=1, validation_data=(X_test_array, y_test)
,shuffle=True,callbacks=[reduce_lr])


# In[54]:


plt.plot(history.history["loss"][5:])
plt.plot(history.history["val_loss"][5:])
plt.title("model loss")
plt.ylabel("loss")
plt.xlabel("epoch")
plt.legend(["train", "test"], loc="upper left")
plt.show()


# In[55]:


refined_dataset.head()


# In[56]:


refined_dataset.loc[refined_dataset['userId'] == 75]


# In[57]:


X_train_array


# In[58]:


user_id = [75]
encoded_user_id = user_enc.transform(user_id)

seen_movies = list(refined_dataset[refined_dataset['userId'] == user_id[0]]['movie'])
print(seen_movies)
     


# In[59]:


len(refined_dataset['movie'].unique()), min(refined_dataset['movie']), max(refined_dataset['movie'])


# In[60]:


unseen_movies = [i for i in range(min(refined_dataset['movie']), max(refined_dataset['movie'])+1) if i not in seen_movies]
print(unseen_movies)


# In[61]:


len(unseen_movies) + len(seen_movies)


# In[62]:


model_input = [np.asarray(list(encoded_user_id)*len(unseen_movies)), np.asarray(unseen_movies)]
len(model_input), len(model_input[0])


# In[63]:


predicted_ratings = model.predict(model_input)


# In[64]:


print(predicted_ratings.shape)


# In[65]:


print(predicted_ratings)


# In[66]:


predicted_ratings = np.max(predicted_ratings, axis=1)
predicted_ratings


# In[67]:


predicted_ratings.shape


# In[68]:


sorted_index = np.argsort(predicted_ratings)[::-1]
print(sorted_index)


# In[69]:


recommended_movies = item_enc.inverse_transform(sorted_index)
recommended_movies


# In[70]:


from pprint import pprint
pprint(list(recommended_movies[:20]))


# In[71]:


df_recom = pd.DataFrame(recommended_movies, columns=['movie'])


# In[72]:


df_recom


# In[73]:


data.head()


# In[74]:


refined_dataset2 = refined_dataset.copy()


# In[75]:


with open('input.txt', 'r') as f:
  input_data = int(f.read())
  
# Use the input data as the input to your notebook
print(input_data)


# In[76]:


type(input_data)


# In[77]:


refined_dataset2 = refined_dataset2[refined_dataset2['userId'] == input_data]


# In[78]:


refined_dataset2.rename(columns = {'title':'movie', 'movie':'movieId'}, inplace = True)


# In[79]:


refined_dataset2


# In[80]:


merged_dataset2 = pd.merge(df_recom, refined_dataset2, how='inner', on='movie')


# In[81]:


merged_dataset2.head()


# In[82]:


np.mean(merged_dataset2['rating'].value_counts().index)


# In[83]:


data3 = data.copy()


# In[84]:


data3.head()


# In[85]:


df_ortalamalar = pd.DataFrame(columns=['Film', 'Ortalama Oy'])

# Film kolonundaki tüm farklı değerleri alın
for film in data3['title'].unique():
  # Film kolonunda o değere sahip satırları seçin
  film_satirlari = data3[data3['title'] == film]
  # o film için kullanıcı oylarının ortalamasını alın
  ortalama_oy = film_satirlari['rating'].mean()
  # yeni veri çerçevesine o film ve ortalama oy değerlerini ekleyin
  df_ortalamalar = df_ortalamalar.append({'Film': film, 'Ortalama Oy': ortalama_oy}, ignore_index=True)


# In[86]:


df_ortalamalar


# In[87]:


df_ortalamalar.rename(columns = {'Film':'movie'}, inplace = True)


# In[88]:


df_ortalamalar


# In[89]:


merged_dataset5 = pd.merge(df_recom, df_ortalamalar, how='inner', on='movie')


# In[90]:


merged_dataset5


# In[91]:


merged_dataset5['Ortalama Oy'] = round(merged_dataset5['Ortalama Oy'],1)


# In[92]:


merged_dataset5[:20]


# In[93]:


final = merged_dataset5[:20]
final


# In[94]:


data.loc[data['title'] == 'Inception (2010)']


# In[95]:


data.loc[data['title'] == 'What Men Talk About (2010)']  #rating'in bu kadar yüksek çıkmasının sebebi bir kullanıcının giriş yapmasıymış.


# In[96]:


#Kodu düzenle. Merged5'in round'Unu almak yerine df_ortalamalarda hallet, isimlendirmeleleri düzelt.


# In[97]:


model.save('softmax.h5')


# In[98]:


refined_dataset.to_json('refined_dataset.json', orient='records')


# In[99]:


data.to_json('data.json', orient='records')


# In[100]:


final.to_json('final.json', orient='records')


# In[101]:


final


# In[104]:

def last_cell():
  df_recom2 = pd.DataFrame(recommended_movies, columns=['movie'])
  refined_dataset3 = refined_dataset.copy()
  with open('input.txt', 'r') as f:
    input_data = int(f.read())
    
  # Use the input data as the input to your notebook
  print(input_data)

  refined_dataset3 = refined_dataset3[refined_dataset3['userId'] == input_data]

  data4 = data.copy()

  refined_dataset3.rename(columns = {'title':'movie', 'movie':'movieId'}, inplace = True)

  df_ortalamalar2 = pd.DataFrame(columns=['Film', 'Ortalama Oy'])

  # Film kolonundaki tüm farklı değerleri alın
  for film in data4['title'].unique():
    # Film kolonunda o değere sahip satırları seçin
    film_satirlari = data4[data4['title'] == film]
    # o film için kullanıcı oylarının ortalamasını alın
    ortalama_oy = film_satirlari['rating'].mean()
    # yeni veri çerçevesine o film ve ortalama oy değerlerini ekleyin
    df_ortalamalar2 = df_ortalamalar2.append({'Film': film, 'Ortalama Oy': ortalama_oy}, ignore_index=True)

  df_ortalamalar2.rename(columns = {'Film':'movie'}, inplace = True)

  merged_dataset6 = pd.merge(df_recom2, df_ortalamalar2, how='inner', on='movie')

  merged_dataset6['Ortalama Oy'] = round(merged_dataset6['Ortalama Oy'],1)
  final2 = merged_dataset6[:20]

  final2

if __name__ == '__main__':
    last_cell()

