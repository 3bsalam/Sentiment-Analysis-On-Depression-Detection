from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
import re
import nltk


def del_Punctutation(s):
    """
    @Input : String ( un-cleaned sentence)

    @Output :  String ( sentence without punctutation).

    @Intent :  takes sentence and delete any punctutation inside it

    @Assumptions (The less assumptions, the less coupling in the code) :
    - string is passed
    - no null data inserted
    """
    return re.sub("[\.\t\,\:;\(\)\_\.!\@\?\&\--]", "", s, 0, 0)


def get_tokens_list(Data):
    """
    @Input : String ( full sentence)

    @Output :  List ( List of token words (no (stopping words, ing , ed .. etc)).

    @Intent :  takes sentence and delete any stopping words or additional characters

    @Assumptions (The less assumptions, the less coupling in the code) :
    - string is passed
    - string is sentence (eg. I love real madrid)0000000000000000
    - no null data inserted
    """

    stemmer = SnowballStemmer("english")
    tokensList = []
    stopWords = set(stopwords.words('english'))

    # Loop through sentence get the words

    t = []
    tokens = nltk.word_tokenize(del_Punctutation(Data).lower())
    for token in tokens:
        if token not in stopWords:
            token = stemmer.stem(token)
            t.append(token)

    # add all tokens of the sentence to list

    tokensList = t

    return tokensList


