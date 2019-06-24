import gensim.models.keyedvectors as word2vec
import numpy as np

def LoadWord2VecModel():

    model = word2vec.KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True)
    # try:
    #     print("Loading Word2Vec Model ...")
    #     #model = Word2Vec.load('\Word2Vec_Model')

    #
    # except:
    #     wordsData = []
    #     print("Training Word2Vec Model ...")
    #     with open('DatasetForword2vec.tsv', 'r',
    #               encoding="latin-1") as csvfile:
    #         readCSV = csv.reader(csvfile, delimiter='\t')
    #         for row in readCSV:
    #             wordsData.append(row[1])
    #
    #     wordsData = get_tokens_list(wordsData)
    #
    #     model = Word2Vec(wordsData)
    #     model.save('\Word2Vec_Model')

    return model

def convert2vec(sentence,word2vModel):

    TrainingVectors = []
    vectors = []
    for j in range(0, len(sentence)):
        word = sentence[j]
        vectorsize = 300
        if not word in word2vModel.wv.vocab:
           vector = (np.zeros(vectorsize))
        else:
            vector = word2vModel[word]
        vectors.append(vector)
    TrainingVectors = vectors

    return TrainingVectors

def padding(datavector):

    if len(datavector)>=10:
        datavector = datavector[0:10]
    else:

        vectorsize = 300
        for j in range (len(datavector),10):

            datavector.append(np.zeros(vectorsize))
    return datavector

def flat_vectors(tv):
    ntv=[]
    row= []
    for j in range (0,len(tv)):
        for k in range (0,len(tv[j])):

            row.append(tv[j][k])
    ntv = row
    return ntv

def analyze(sentence,word2vecModel):
    """
    @Input : List ( full cleaned sentence) - Model (word2vec converter model)

    @Output :  List ( feature vector).

    @Intent :  takes sentence and convert to vec , padding, flatting

    @Assumptions (The less assumptions, the less coupling in the code) :
    - List of strings
    - Data must be preprocessed using tweets cleaner first
    - no null data inserted
    """
    sentenceVec = convert2vec(sentence, word2vecModel)
    sentenceVec = padding(sentenceVec)
    sentenceVec = flat_vectors(sentenceVec)

    return sentenceVec