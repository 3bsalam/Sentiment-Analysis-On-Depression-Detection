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



def del_Punctutation(s):
    return re.sub("[\.\t\,\:;\(\)\_\.!\@\?\&\--]", "", s, 0, 0)

def get_tokens_list(Data):

    stemmer = SnowballStemmer("english")
    tokensList = []
    stopWords = set(stopwords.words('english'))

    for i in range(1, len(Data)):
        #### to get the words from every sentences
        t = []
        tokens = nltk.word_tokenize(del_Punctutation(Data[i]).lower())
        for token in tokens:
            if token not in stopWords:
                token = stemmer.stem(token)
                t.append(token)
        tokensList.append(t)
        #### add all tokens of the tweets to list


    return tokensList

def LoadWord2VecModel():

    try:
        print("Loading Word2Vec Model ...")
        model = Word2Vec.load('\Word2Vec_Model')
    except:
        wordsData = []
        print("Training Word2Vec Model ...")
        with open('DatasetForword2vec.tsv', 'r',
                  encoding="latin-1") as csvfile:
            readCSV = csv.reader(csvfile, delimiter='\t')
            for row in readCSV:
                wordsData.append(row[1])

        wordsData = get_tokens_list(wordsData)

        model = Word2Vec(wordsData)
        model.save('\Word2Vec_Model')

    return model



def convert2vec(Sentences,word2vModel):
    TrainingVectors = []
    for i in range(0, len(Sentences)):
        sentence = Sentences[i]
        vectors = []
        for j in range(0, len(sentence)):
            word = sentence[j]
            if not word in word2vModel.wv.vocab:
                continue
            vector = word2vModel[word]
            vectors.append(vector)
        TrainingVectors.append(vectors)
    return TrainingVectors
###################
def padding(datavector):
    for i in range(0,len(datavector)):
        vectorsize =  100
        for j in range (len(datavector[i]),20):
            datavector[i].append(np.zeros((1,vectorsize)))
    return datavector


def filter(Sentences,labels):
    nSentences=[]
    nLabels=[]
    for i in range (len(Sentences)):
        if Sentences[i] != []:
            nSentences.append(Sentences[i])
            nLabels.append(labels[i])
    return nSentences,nLabels




def main():
 TrainingSentences = []
 TrainingLabels = []
 with open('SentimentAnalysisDataset100000.csv', 'r', encoding="latin-1") as csvfile:
    readCSV = csv.reader(csvfile, delimiter=",")
    for row in readCSV:
        TrainingSentences.append(row[2])
        TrainingLabels.append(row[1])
####### Pre Processing to clean The sentences
##print(TrainingSentences[5])
 TestingVectors = TrainingSentences[99800:]
 TestingLabels = TrainingLabels[99800:]
 TrainingSentences = TrainingSentences[1:1000]
 TrainingLabels = TrainingLabels[1:1000]

 TrainingSentences = get_tokens_list(TrainingSentences)
 #TrainingSentences,TrainingLabels = filter(TrainingSentences,TrainingLabels)
 print("2) Feature Extracting ...")
 word2v = LoadWord2VecModel()

 TrainingVectors = convert2vec(TrainingSentences,word2v)
 TrainingVectors = padding(TrainingVectors)
# TrainingVectors =  TrainingVectors[1:1000]

 #TrainingLabels =  TrainingLabels[1:1000]



 clf = svm.SVC(kernel='rbf',max_iter = 1 )
 print("3) Training and Fitting ...")
 clf.fit(TrainingVectors, TrainingLabels)
 print("4) Predicting ...")
 ##print('\n'.join([TrainingSentences[0], TrainingVectors[0]]))


 correct = 0
 for i in range(0, len(TestingVectors)):
     predict = clf.predict(TestingVectors[i])
     if predict == TestingLabels[i]:
         correct = correct + 1

 accuracy = correct / len(TestingLabels)

 print("accuracy =",accuracy*100,"%")
 #########











if __name__ == '__main__':
    main()
