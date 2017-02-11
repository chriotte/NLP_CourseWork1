# coding: utf-8

import unicodecsv			# csv reader
from sklearn.svm import LinearSVC
from nltk.classify import SklearnClassifier
from random import shuffle

# DATA LOADING AND PARSING

# convert line from input file into a datetime/string pair
def parseTweet(tweetLine):
    # should return a pair of a datetime object and a string containing the tweet message
    text = tweetLine[4]
    date = tweetLine[1]
    return (date, text)

# load data from a file and append it to the tweetData
def loadData(path, label, tweet=None):
    with open(path, 'rb') as f:
        reader = unicodecsv.reader(f, encoding='utf-8')
        for line in reader:
            if(line[0] != 'id'):
                (dt,tweet) = parseTweet(line)
                tweetData.append((dt,preProcess(tweet),label))
                trainData.append((toFeatureVector(preProcess(tweet)),label))

# load application data
def loadApplicationData(path):
    with open(path, 'rb') as f:
        reader = unicodecsv.reader(f, encoding='utf-8')
        for line in reader:
            londonTweetData.append(parseTweet(line))

# TEXT PREPROCESSING AND FEATURE VECTORIZATION
# input: a string of one tweet
def preProcess(text):
    # should return a list of tokens
    
    list1 = text.split()  
    return list1

# input: a tokenised sequence
featureDict = []
#################################
## CHECK THE NEW COURSEWORK STUFF
#################################
def toFeatureVector(words):
    for word in words:
        print("Checking word: ",word)
        if word in featureDict:
            print(word, " is already there")
        else:
            featureDict.append(word)
            print("added word: ", word)
    # return a dictionary 'featureDict' where the keys are indices (one for every word) and values are numerical feature representations
    return 

# TRAINING AND VALIDATING OUR CLASSIFIER

def trainClassifier(trainData):
    print("Training Classifier...")
    return SklearnClassifier(LinearSVC()).train(trainData)

def crossValidate(dataset, folds):
    shuffle(dataset)
    results = []
    foldSize = int(len(dataset)/folds)
    for i in range(0,len(dataset),foldSize):
        pass
        # insert code here that trains and tests on the 10 folds of data in the dataset
    return results

# PREDICTING LABELS GIVEN A CLASSIFIER

def predictLabels(tweetData, classifier):
    return classifier.classify_many(map(lambda t: toFeatureVector(preProcess(t[1])), tweetData))

def predictLabel(text, classifier):
    return classifier.classify(toFeatureVector(preProcess(text)))

# COMPUTING ANGER LEVEL ON A SET OF TWEETS

def findAngerLevels(tweetData, classifier):
    # should return a list of anger level for a period of time
    return

# MAIN

# loading tweets
# tweets will have one feature, which is going to be a numerical value

tweetData = []
trainData = []
londonTweetData = []

# the output classes
angryLabel = 'angry'
happyLabel = 'happy'

path = "/Volumes/Lagring/GitHub/University/NLP/"

path = ""

# references to the data files
angryPath = path + 'angry_tweets.csv'
happyPath = path + 'happy_tweets.csv'
londonPath = path + 'london_2017_tweets.csv'

angryPath = path + 'angry_tweets_1000.csv'
happyPath = path + 'happy_tweets_1000.csv'
londonPath = path + 'london_2017_tweets_1000.csv'

angryPath = path + 'angry_tweets_100.csv'
happyPath = path + 'happy_tweets_100.csv'
londonPath = path + 'london_2017_tweets_100.csv'

# do the actual stuff
print("Loading happy tweets...")
loadData(happyPath, happyLabel)

print("Loading angry tweets...")
loadData(angryPath, angryLabel)

print('number of words: ' + str(len(featureDict)))

cv_results = crossValidate(trainData, 10)
print(cv_results)

classifier = trainClassifier(trainData)

print("Loading London data")
loadApplicationData(londonPath)

print("Computing anger levels!")
angerLevels = findAngerLevels(londonTweetData, classifier)
anger_peaks = None

print(anger_peaks)
