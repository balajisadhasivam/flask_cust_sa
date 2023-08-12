import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import spacy
nlp = spacy.load('en_core_web_sm')

# Assign column names
columan_name = ['Reviews', 'Sentiment']

df = pd.read_csv('dataset\imdb_labelled.txt',sep='\t',header=None)
df.head()

df.columns = columan_name

df.head()

df.shape

# check distribution of sentiments

df['Sentiment'].value_counts()

# check for null values
df.isnull().sum()
# no null values in the data

x = df['Reviews']
y = df['Sentiment']

# 2) Data Cleaning

# here we will remove stopwords, punctuations
# as well as we will apply lemmatization

# Create a list comprehension to get Word Count, Uppercase Char Count, Special Char Count

import string
df['Word Count'] = [len(review.split()) for review in df['Reviews']]

df['Uppercase Char Count'] = [sum(char.isupper() for char in review) \
                              for review in df['Reviews']]

df['Special Char Count'] = [sum(char in string.punctuation for char in review) \
                            for review in df['Reviews']]

df.head()

punct = string.punctuation

punct

from spacy.lang.en.stop_words import STOP_WORDS

stopwords = list(STOP_WORDS) # list of stopwords

"""## Creating a function for data cleaning"""

def text_data_cleaning(sentence):
  doc = nlp(sentence)

  tokens = [] # list of tokens
  for token in doc:
    if token.lemma_ != "-PRON-":
      temp = token.lemma_.lower().strip()
    else:
      temp = token.lower_
    tokens.append(temp)

  cleaned_tokens = []
  for token in tokens:
    if token not in stopwords and token not in punct:
      cleaned_tokens.append(token)
  return cleaned_tokens

# if root form of that word is not pronoun then it is going to convert that into lower form
# and if that word is a proper noun, then we are directly taking lower form, because there is no lemma for proper noun

text_data_cleaning("Hello all, It's a beautiful day outside there!")
# stopwords and punctuations removed

## Vectorization Feature Engineering (TF-IDF)

from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline

tfidf = TfidfVectorizer(tokenizer=text_data_cleaning)
# tokenizer=text_data_cleaning, tokenization will be done according to this function

classifier = LinearSVC()

# 3) Train the model

## Splitting the dataset into the Train and Test set


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)

x_train.shape, x_test.shape
# 2198 samples in training dataset and 550 in test dataset

x_train.head()

## Fit the x_train and y_train

clf = Pipeline([('tfidf',tfidf), ('clf',classifier)])
# it will first do vectorization and then it will do classification

clf.fit(x_train, y_train)

# save the model in pickle file
with open('cust_review_senti.pkl', 'wb') as file:
    pickle.dump((clf, text_data_cleaning), file)

# in this we don't need to prepare the dataset for testing(x_test)

# 4) Predict the Test set results

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

y_pred = clf.predict(x_test)

# confusion_matrix
confusion_matrix(y_test, y_pred)

# classification_report
print(classification_report(y_test, y_pred))


accuracy_score(y_test, y_pred)

clf.predict(["While informative, the film struggled to capture the excitement of scientific discovery."])

clf.predict(["Blue Beetle's potential was not fully realized, resulting in a less-than-thrilling adventure."])

