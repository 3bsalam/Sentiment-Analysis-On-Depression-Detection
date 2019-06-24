import csv

import cv2
import numpy as np

from analyzer import *
from tweets_cleaner import *

svm = cv2.ml.SVM_create()
svm.setType(cv2.ml.SVM_C_SVC)
svm.setKernel(cv2.ml.SVM_RBF)

knn = cv2.ml.KNearest_create()
word2v = word2vec.KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300-SLIM.bin', binary=True)


def read_dataset(file, min=1, max=-1):
    TrainingSentences = []
    TrainingLabels = []

    with open(file, 'r', encoding="latin-1") as csvfile:
        readCSV = csv.reader(csvfile, delimiter=",")
        if max == -1:
            max = len(readCSV)

        for row in readCSV:
            TrainingSentences.append(row[3])
            TrainingLabels.append(row[1])

    TrainingSentences = TrainingSentences[min:max]
    TrainingLabels = [int(x) for x in TrainingLabels[min:max]]

    return TrainingSentences, TrainingLabels


def svm_train(vectors, labels):
    svm.train(vectors, cv2.ml.ROW_SAMPLE, labels)


def knn_train(vectors, labels):
    knn.train(vectors, cv2.ml.ROW_SAMPLE, labels)


def svm_test(vectors, labels,svm=svm):
    predict = svm.predict(vectors)
    correct = 0
    for i in range(0, len(vectors)):
        if predict[1][i] == labels[i]:
            correct = correct + 1

    accuracy = correct / len(labels)
    accuracy = accuracy * 100
    return accuracy


def knn_test(vectors, labels,knn=knn):
    ret, predict, neighbours, dist = knn.findNearest(vectors, k=1)
    correct = 0
    for i in range(0, len(vectors)):
        if predict[i] == labels[i]:
            correct = correct + 1

    accuracy = correct / len(labels)
    accuracy = accuracy * 100
    return accuracy


def train():
    TrainingSentences = []
    TrainingLabels = []

    print("1) Reading Training Dataset ...")

    TrainingSentences, TrainingLabels = read_dataset('SentimentAnalysisDataset100000.csv', 1, 50000)

    TrainingVector = []

    print("2) Cleaning The Dataset and Getting feature vectors ...")
    for i in range(0, len(TrainingSentences)):
        sentence = get_tokens_list(TrainingSentences[i])
        vector = analyze(sentence, word2v)
        TrainingVector.append(vector)

    TrainingVector = np.array(TrainingVector)
    TrainingVector = np.array(TrainingVector, np.float32)
    TrainingLabels = np.array([TrainingLabels]).T

    print("3) SVM Training...")
    svm_train(TrainingVector, TrainingLabels)
    svm.save('SVMModel.dat')

    print("4) KNN Training...")
    knn_train(TrainingVector, TrainingLabels)
    knn.save('KNNModel.dat')


def test(svm=svm,knn=knn):
    print("5) Reading Testing Dataset...")

    TestingSentences, TestingLabels = read_dataset('SentimentAnalysisDataset100000.csv', 50000, 55000)

    TestingVector = []
    print("6) Cleaning Testing Dataset and Getting FeatureVector...")
    for i in range(0, len(TestingSentences)):
        sentence = get_tokens_list(TestingSentences[i])
        vector = analyze(sentence, word2v)
        TestingVector.append(vector)

    TestingVector = np.array(TestingVector, np.float32)
    TestingLabels = np.array([TestingLabels]).T

    print("7) Testing SVM...")
    accuracy_svm = svm_test(TestingVector, TestingLabels,svm)
    print("accuracy of SVM = ", accuracy_svm, "%")
    print("8) Testing KNN...")
    accuracy_knn = knn_test(TestingVector, TestingLabels,knn)
    print("accuracy of KNN = ", accuracy_knn, "%")

    return accuracy_svm, accuracy_knn


def init(svm=svm, knn=knn):
    try:
        print("Loading SVM and KNN Models... ")

        svm = cv2.ml.SVM_load("SVMModel.dat")

        # fs = cv2.FileStorage('KNNModel.dat', cv2.FILE_STORAGE_READ)
        # knn_yml = fs.getNode('opencv_ml_knn')
        #
        # knn_format = knn_yml.getNode('format').real()
        # is_classifier = knn_yml.getNode('is_classifier').real()
        # default_k = knn_yml.getNode('default_k').real()
        # samples = knn_yml.getNode('samples').mat()
        # responses = knn_yml.getNode('responses').mat()
        # fs.release()
        # knn.train(samples, cv2.ml.ROW_SAMPLE, responses)
        print("Loaded Models Successfully!!")

    except:
        print("Failed to find Models")
        print("Training new models....")
        train()

   # test(svm,knn)
    return svm, knn


def predict(sentence, svm, knn, mode='svm'):
    result = 0
    sentence = get_tokens_list(sentence)
    vector = analyze(sentence, word2v)
    vector = np.array([vector])
    vector = np.array(vector, np.float32)
    if mode == 'svm':

        result = svm.predict(vector)

        emoition = ''
        if result[1][0] == 0:
            emoition = 'depressed'
        elif result[1][0]== 1 :
            emoition = 'not depressed'

        return emoition

    elif mode == 'knn':
        ret, result, neighbours, dist = knn.findNearest(vector, k=1)
        emoition = ''
        if result[0] == 0:
            emoition = 'depressed'
        elif result[0] == 1:
            emoition = 'not depressed'
        return emoition

    return 0


if __name__ == '__main__':
    svm, knn = init()
    print(predict('i love my life i am so happy', svm, knn, mode='knn'))
