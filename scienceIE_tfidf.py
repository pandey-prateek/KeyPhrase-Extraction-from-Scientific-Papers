import os
from collections import defaultdict
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
import re

def preprocess (text):
    text = text.lower ()

    text = re.sub ('(\\d|\\W)+', ' ', text)
    text = re.sub ('\d+(\.\d*)?', ' ', text)
    split_text = text.split ()

    split_text = [word for word in split_text if len (word) > 3]
    text = ' '.join (split_text)
    return text

def extract_topn (feature_names, text, topn = 15):
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

# Folder Path

# Read text File
def read_text_file(file_path):
    with open(file_path, 'r') as f:
        return f.read()

def read_ann_file(file_path):
    with open(file_path, 'r') as f:
        l=[]
        for line in f.readlines():
            l.append(line[:-1].split('\t') [-1])
        return l
        # data[name.split('.')[0]].append(l)
  
  
# iterate through all file

def load_data (path):  
    data=defaultdict(dict)  

    for file in os.listdir(path):
        # Check whether file is in text format or not
        
        name=file.split('.')[0]
        if file.endswith(".txt"):
            file_path = path + file
            data[name]['text']=read_text_file(file_path)
        if file.endswith(".ann"):
            file_path = path + file
            # call read text file function
            data[name]['ann']=read_ann_file(file_path)
    return data

train_data = load_data ('./scienceie2017_train/train2/')
train_text = []
for i in train_data.keys ():
    train_text.append (train_data [i]['text'])

train_text = [text.strip ('\n') for text in train_text]

test_text = []
y_test = []

test_data = load_data ('./semeval_articles_test/')

for i in test_data.keys ():
    test_text.append (test_data [i]['text'])
    y_test.append (test_data [i]['ann'])

test_text = [text.strip ('\n') for text in test_text]

train_text = [preprocess (text) for text in train_text]
test_text = [preprocess (text) for text in test_text]


vocab = CountVectorizer (max_df = 100, min_df = 1, stop_words = 'english', max_features = 10000, ngram_range = (1, 3))

word_list = vocab.fit_transform (train_text)

tfidftrans = TfidfTransformer (smooth_idf = True, use_idf = True)
tfidftrans.fit (word_list)

feature_names = vocab.get_feature_names_out ()

y_hat = []
for test_doc in test_text:
    tfidfVector = tfidftrans.transform (vocab.transform ([test_doc]))

    coo_matrix = tfidfVector.tocoo ()

    tuples = zip (coo_matrix.col, coo_matrix.data)
    sorted_data = sorted (tuples, key = lambda x: (x[1], x[0]), reverse = True)

    keywords = extract_topn (feature_names, sorted_data, 15)

    y_hat.append (keywords)

cnt = 0

for i in range (len (y_test)):
    for word in y_test [i]:
        if word in y_hat [i]:
            print (word)
            cnt += 1

print ("Exact keyword matches ", cnt)