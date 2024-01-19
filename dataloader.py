import os
import re
import readline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from nltk.stem.porter import *

def preprocess (text, stemmer):
    text = text.lower ()

    text = re.sub ('(\\d|\\W)+', ' ', text)
    text = re.sub ('\d+(\.\d*)?', ' ', text)
    split_text = text.split ()
    split_text = [stemmer.stem (word) for word in split_text]

    split_text = [word for word in split_text if len (word) > 3]
    text = ' '.join (split_text)
    return text

def extract_topn (feature_names, text, topn = 10):
    text = text [:topn]
    scores = []
    features = []

    for index, score in text:
        scores.append (round (score, 3))
        features.append (feature_names [index])

    results = []

    for index in range (len (features)):
        results.append (features [index])
    
    return results

def data_loader (path):
    documents = []

    for document in os.listdir (path):
        if document.endswith ('.txt.final'):
            documents.append (document)

    data = []
    for document in documents:
        with open (path+document, 'r') as file:
            text = file.read ()
            text = text.strip ()
            data.append (text)
    return data

def evaluate (test_type, y_hat):
    y_test = []

    with open (f'test_answer/test.{test_type}.stem.final') as file:
        y_test = file.read ()

    y_test = y_test.split ('\n')
    y_test = y_test [:-1]

    y_test = [ele.split (':') [-1].strip () for ele in y_test]

    y_test = [ele.split (',') for ele in y_test]

    cnt = 0

    for i in range (len (y_test)):
        for word in y_test [i]:
            if word in y_hat [i]:
                print (word)
                cnt += 1

    print (f"Exact keyword matches {test_type}", cnt)

stemmer = PorterStemmer ()

train_data = data_loader ('train/')
train_data = [preprocess (text, stemmer) for text in train_data]

vocab = CountVectorizer (max_df = 100, min_df = 1, stop_words = 'english', max_features = 10000, ngram_range = (1, 3))

word_list = vocab.fit_transform (train_data)

tfidftrans = TfidfTransformer (smooth_idf = True, use_idf = True)
tfidftrans.fit (word_list)

feature_names = vocab.get_feature_names_out ()

test_data = data_loader ('test/')
test_data = [preprocess (text, stemmer) for text in test_data]

y_hat = []
for test_doc in test_data:
    tfidfVector = tfidftrans.transform (vocab.transform ([test_doc]))

    coo_matrix = tfidfVector.tocoo ()

    tuples = zip (coo_matrix.col, coo_matrix.data)
    sorted_data = sorted (tuples, key = lambda x: (x[1], x[0]), reverse = True)

    keywords = extract_topn (feature_names, sorted_data, 15)

    y_hat.append (keywords)

evaluate ('combined', y_hat)
evaluate ('reader', y_hat)
evaluate ('author', y_hat)