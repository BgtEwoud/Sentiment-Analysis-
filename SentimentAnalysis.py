# -*- coding: utf-8 -*-
"""
Created on Mon Dec 30 10:29:13 2019

@author: BU573YA
"""

#Data Analysis
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#Data Preprocessing and Feature Engineering
from textblob import TextBlob
import re
import os
import nltk
from nltk.corpus import wordnet
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

#Model Selection and Validation
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix, classification_report,accuracy_score

#Change your working directory to where your raw data is located
os.chdir('C:/Users/BU573YA/Documents/1_Work/Business Development/Projects/Sentiment Analysis/Source')
train_tweets = pd.read_csv('train_tweets.csv')
test_tweets = pd.read_csv('test_tweets.csv')

#Perform some data exploration
sns.countplot(x = 'label' , data = train_tweets)

#Define function so that we first identify the individual words of the tweet and then we append all the words in the original sentence
def form_sentence(tweet):
    tweet_blob = TextBlob(tweet)
    return ' '.join(tweet_blob.words)

print(form_sentence(train_tweets['tweet'].iloc[10]))
print(train_tweets['tweet'].iloc[10])

'''
â ireland consumer price index mom climbed from previous 0.2 to 0.5 in may blog silver gold forex
â #ireland consumer price index (mom) climbed from previous 0.2% to 0.5% in may   #blog #silver #gold #forex
'''


#Create function to find the approriate part of speech tag based on the context in which the word is used
def get_wordnet_pos(word):
    """Map POS tag to first character lemmatize() accepts"""
    tag = nltk.pos_tag([word])[0][1][0].upper()
    tag_dict = {"J": wordnet.ADJ,
                "N": wordnet.NOUN,
                "V": wordnet.VERB,
                "R": wordnet.ADV}
    return tag_dict.get(tag, wordnet.NOUN)

#Perform lexicon normalisation so we only have one representation of each word in a sentence eg. Play, played, playing => play

def normalization(tweet_list):
    lem = WordNetLemmatizer()
    normalized_tweet = []
    for word in tweet_list:
        normalized_text = lem.lemmatize(word, get_wordnet_pos(word))
        normalized_tweet.append(normalized_text)
    return normalized_tweet

#Create pipeline model which uses a naive bayes classifier on a vectorized version of our preprocessed data. The pipeline model will save much time and computational power.
pipeline = Pipeline([
    ('bow',CountVectorizer()),  # strings to token integer counts
    ('tfidf', TfidfTransformer()),  # integer counts to weighted TF-IDF scores
    ('classifier', MultinomialNB()),  # train on TF-IDF vectors w/ Naive Bayes classifier
])

    
#Model validation
    
msg_train, msg_test, label_train, label_test = train_test_split(train_tweets['tweet'], train_tweets['label'], test_size=0.2)
pipeline.fit(msg_train,label_train)
predictions = pipeline.predict(msg_test)
print(classification_report(predictions, label_test))
print(accuracy_score(predictions, label_test))
