# -*- coding: utf-8 -*-

# Import the libraires
from typing import Dict, Text

import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds

import tensorflow_recommenders as tfrs
import pprint
from matplotlib import pyplot as plt

    ############################

# Load data for votes
votes = tfds.load("movielens/100k-ratings", split="train")
# Load data for movies
movies = tfds.load("movielens/100k-movies", split="train")

# We need to map the necessary attributes
 
votes = votes.map(lambda x: {"movie_title": x["movie_title"], "user_id": x["user_id"],})
movies = movies.map(lambda x: x['movie_title'])      

#View the data

for x in votes.take(1).as_numpy_iterator():
    pprint.pprint(x)

# We separate our data set into training/testing
# Assign a seed=73 for consistency of results and permtuez the data to keep no particular order
seed = 73
    ## Dataset length
l = len(votes)
tf.random.set_seed(seed)
shuffled = votes.shuffle(l, seed=seed, reshuffle_each_iteration=False)     

#Save 75% of the data for training and 25% for testing
train_ = int(0.75 * l)
test_ = int(0.25 * l)
train = shuffled.take(train_)
test = shuffled.skip(train_).take(test_) 

# We check how many users and unique movies     
movies_titles = movies.batch(l)
user_ids = votes.batch(l).map(lambda x: x['user_id'])

#films uniques
'''titles = iter(films_titres)
titles = list(set(next(titles).numpy()))'''
titles = np.unique(np.concatenate(list(films_titres)))
len_films = len(titles)
print(len_films) #1682

    #users unique
'''ids = iter(user_ids)
ids_ = list(set(next(ids).numpy()))'''
ids = np.unique(np.concatenate(list(user_ids)))
len_users = len(ids)
print(len_users) #943

embedding_dimension = 32

#We define the embedding on the user side, we must transform the user ids into a vector representation
user_model = tf.keras.Sequential([tf.keras.layers.experimental.preprocessing.StringLookup(vocabulary=ids_, mask_token=None),
                                  tf.keras.layers.Embedding(len_users + 1,
                                                            embedding_dimension)])
                                                                                        
# We now define the embedding of the film portion 

film_model = tf.keras.Sequential([tf.keras.layers.experimental.preprocessing.StringLookup(vocabulary=titles ,mask_token=None),
                                   tf.keras.layers.Embedding(len_films + 1, 
                                                             embedding_dimension)])                              
                                                           
#We define the desired metrics : FactorizedTopK
metrics = tfrs.metrics.FactorizedTopK(candidates=films.batch(128).map(film_model))

#The Retrieval task is defined according to the FactorizedTopK metrics. 
task = tfrs.tasks.Retrieval(metrics=metrics)

class MovieLensModel(tfrs.Model):

  def __init__(self, user_model, film_model):
    super().__init__()
    self.film_model: tf.keras.Model = film_model
    self.user_model: tf.keras.Model = user_model
    self.task: tf.keras.layers.Layer = task

  def compute_loss(self, features: Dict[Text, tf.Tensor], training=False) -> tf.Tensor:
    
    user_embeddings = self.user_model(features['user_id'])

    positive_film_embeddings = self.film_model(features['movie_title'])

    # La task calcule les métriques et le loss
    return self.task(user_embeddings, positive_film_embeddings)

model = MovieLensModel(user_model, film_model)
model.compile(optimizer=tf.keras.optimizers.Adagrad(learning_rate=0.1))

#Segmenter les batchs de manière à ce que le modèle roule 10 batch d'entraînement et 13 batchs de test par epoch, tout en ayant un batch size qui est un multiple de 2^n.  
cached_train = train.shuffle(l).batch(8192).cache()
cached_test = test.batch(2048).cache()

## Fit the model 
history_train = model.fit(cached_train, validation_data = cached_test, epochs=32)

plt.plot(history_train.history['total_loss'] )
plt.title("Total Loss over epochs", fontsize=14)
plt.ylabel('Loss Total')
plt.xlabel('epochs')
plt.show()

plt.plot(history_train.history['factorized_top_k/top_100_categorical_accuracy'],color='green', alpha=0.8, label='Train' )
plt.plot(history_train.history['val_factorized_top_k/top_100_categorical_accuracy'],color='red', alpha=0.8, label='Test' )
plt.title("Accuracy over epochs", fontsize=14)
plt.ylabel('Accuracy')
plt.xlabel('epochs')
plt.legend(loc='upper left')
plt.show()

##Evaluate the model
model.evaluate(cached_test, return_dict=True)

# Recommend the 5 best movies for user 25

index = tfrs.layers.factorized_top_k.BruteForce(model.user_model)

index.index(films.batch(100).map(model.film_model), films)

print("the top 5 recommended movies for user 25 are : " )

_, titles = index(tf.constant(['25']))
titles[0, :5]