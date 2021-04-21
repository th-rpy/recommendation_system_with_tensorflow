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