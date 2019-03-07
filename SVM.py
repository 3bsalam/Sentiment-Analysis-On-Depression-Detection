import csv

import nltk
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
from scipy import sparse
from gensim import corpora
from gensim.models import Word2Vec
import re
from collections import Counter
import numpy as np
from numpy import asarray
import itertools
from sklearn import svm
import csv
import time
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler

def del_Punctutation(s):
    return re.sub("[\.\t\,\:;\(\)\_\.!\@\?\&\--]", "", s, 0, 0)

def filter(Sentences,labels):
    nSentences=[]
    nLabels=[]
    for Sentence in Sentences:
        if Sentence != []:
            nSentences.append(Sentence)
            nLabels.append(labels)
    return nSentences,nLabels

def get_tokens_list(Data):

    stemmer = SnowballStemmer("english")
    tokensList = []
    stopWords = set(stopwords.words('english'))

    for i in range(1, len(Data)):
        #### to get the words from every sentences
        t = ""
        tokens = nltk.word_tokenize(del_Punctutation(Data[i]))
        for token in tokens:
            if token not in stopWords:
                token = stemmer.stem(token)
                t = t + token + " "
        #### add all tokens of the tweets to list
        tokensList.append(t)

    return tokensList

TrainingSentences = []
TrainingLabels = []
TestingSentences = []
TestingLabels = []
t0 = time.time()

print("1) Reading Data ...")

with open('SentimentAnalysisDataset100000.csv', 'r', encoding="latin-1") as csvfile:
    readCSV = csv.reader(csvfile, delimiter=",")
    for row in readCSV:
        TrainingSentences.append(row[2])
        TrainingLabels.append(row[1])
TestingSentences = TrainingSentences[99000:]
TestingLabels = TrainingLabels[99900:]

TrainingSentences = get_tokens_list(TrainingSentences)
#TrainingSentences,TrainingLabels = filter(TrainingSentences,TrainingLabels)

originalSentences = TrainingSentences
originalLabels = TrainingLabels
TrainingSentences = originalSentences[1:1000]
TrainingLabels = originalLabels[1:1000]
print("2) Feature Extracting ...")
vectorizer = TfidfVectorizer()

TrainingSentences = vectorizer.fit_transform(TrainingSentences)

print("type:",type(TrainingSentences))
clf = svm.SVC(kernel='rbf')
print("3) Training and Fitting ...")
clf.fit(TrainingSentences, TrainingLabels)


print("4) Predicting ...")


t = vectorizer.transform(["I love you"])

correct = 0

for i in range(0, len(TestingSentences)):
    print(TrainingSentences[i],i,TestingSentences[i])
    currentCorpse = [TestingSentences[i]]
    currentCorpse = vectorizer.transform(currentCorpse)
    predict = clf.predict(currentCorpse)
    if predict == TestingLabels[i]:
        correct = correct + 1

accuracy = correct/len(TestingLabels)

print("accuracy=",accuracy*100,"%")

print(time.time() - t0)
