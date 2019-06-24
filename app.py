#   Imports are places here, and only here, please don't add it anywhere else
import os
import cv2
import base64
import numpy as np
import smtplib
import json
from engine import init, predict
from datetime import datetime
from flask import Flask, render_template, jsonify, redirect, url_for, request, Response, json
from flask_cors import CORS
import time
import requests

#from license.License import isValid
#from Digified.OCR.Utility.error_handler import ErrorHandler
#Functions,classes, variables and so on are places here for the production ready
app = Flask(__name__)
app.config.from_object(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['PROPAGATE_EXCEPTIONS'] = True
svm, knn = init()
CORS(app)

def analyze_tweets(tweetsList):
    result = []

    for tweet in tweetsList:
        emotion = {}
        emotion['tweet'] = tweet
        emotion['svm'] = predict(tweet, svm, knn, mode='svm')
        #emotion['knn'] = predict(tweet, svm, knn, mode='knn')
        result.append(emotion)

    return result


def analyze_tweets_percentage(tweetsList):

    depressed = 0
    notDepressed = 0

    for tweet in tweetsList:
        result = predict(tweet, svm, knn, mode='svm')
        if result == 'depressed':
            depressed = depressed + 1
        elif result == 'not depressed':
            notDepressed = notDepressed + 1

    percentage = {}
    percentage['depressed'] = depressed / len(tweetsList)
    percentage['nondepressed'] = notDepressed / len(tweetsList)

    return percentage



@app.errorhandler(500)
def internal_error(error):
    return "ERR_500 : Internal Server Error"



# API Routes #
@app.route('/')
def main():
    return '<center><h1>You\'ve Reached The API, Congratulations !</h1><hr/></center>'

@app.route('/analyze', methods=['POST'])
def get_tweets():
    """
    @Input : POST Request with a JSON inside it with certain format

    @Output :  HTTP Response as JSON and Status Code
    
    @Intent : API Endpoint, used to interface with the outer world and run the engine remotely
    
    @Assumptions (The less assumptions, the less coupling in the code) : 
    - NO ASSUMPTIONS 
    
    @Additional Data :
    - HTTP 200 == Full Success 
    - HTTP 207 == Partial Success
    - HTTP 400 == Bad JSON Format 
    - HTTP 409 == Error  in processing
    """

    if request.method == 'POST':
        #Check for format
        if not request.is_json:
            return Response(
                json.dumps("{ 'error': 'bad input format' }"),
                status=400,
                mimetype='application/json')

        content = request.get_json()
        Rlist = analyze_tweets(content['tweets'])


        return Response(json.dumps(Rlist),status=200,mimetype='application/json')

@app.route('/percentage', methods=['POST'])
def get_depression_precentage():

    if request.method == 'POST':
        #Check for format
        if not request.is_json:
            return Response(
                json.dumps("{ 'error': 'bad input format' }"),
                status=400,
                mimetype='application/json')

        content = request.get_json()
        Rlist = analyze_tweets_percentage(content['tweets'])

        return Response(json.dumps(Rlist),status=200,mimetype='application/json')


if __name__ == "__main__":
    #app.run(host="0.0.0.0", ssl_context=('/etc/letsencrypt/live/digified.ml/fullchain.pem', '/etc/letsencrypt/live/digified.ml/privkey.pem'))
    port = int(os.environ.get('PORT', 5000))
    app.run(host='127.0.0.1', port=port)

