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