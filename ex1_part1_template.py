# coding: utf-8

import unicodecsv			# csv reader
from sklearn.svm import LinearSVC
from nltk.classify import SklearnClassifier
from random import shuffle


# DATA LOADING AND PARSING

# convert line from input file into a datetime/string pair
def parseTweet(tweetLine):
	# should return a pair of a datetime object and a string containing the tweet message
	return

# load data from a file and append it to the tweetData
def loadData(path, label, tweet=None):
	with open(path, 'rb') as f:
		reader = unicodecsv.reader(f, encoding='utf-8')
		reader.next()
		for line in reader:
			(dt,tweet) = parseTweet(line)
			tweetData.append((dt,preProcess(tweet),label))
			trainData.append((toFeatureVector(preProcess(tweet)),label))

# load application data
def loadApplicationData(path):
	with open(path, 'rb') as f:
		reader = unicodecsv.reader(f, encoding='utf-8')
		reader.next()
		for line in reader:
			londonTweetData.append(parseTweet(line))

# TEXT PREPROCESSING AND FEATURE VECTORIZATION

# input: a string of one tweet
def preProcess(text):
	# should return a list of tokens
    return

# input: a tokenised sequence
# you can optionally keep track of a global feature list that is a list of all the features (or words) that you encounter while going through the dataset
# featureList = {} 
def toFeatureVector(words):
    featureVector = {}
	# return a dictionary 'featureVect' where the keys are the tokens in 'words' and the values are the number of occurrences of the tokens
    return featureVector

# TRAINING AND VALIDATING OUR CLASSIFIER

def trainClassifier(trainData):
	print "Training Classifier..."
	return SklearnClassifier(LinearSVC()).train(trainData)

def crossValidate(dataset, folds):
	shuffle(dataset)
	results = []
	foldSize = len(dataset)/folds
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

# references to the data files
angryPath = 'angry_tweets.csv'
happyPath = 'happy_tweets.csv'
londonPath = 'london_2017_tweets.csv'


## In order to test the code in this
##
# do the actual stuff
print "Loading happy tweets..."
loadData(happyPath, happyLabel)
print "Loading angry tweets..."
loadData(angryPath, angryLabel)
print 'number of words: ' + str(len(featureDict))
cv_results = crossValidate(trainData, 10)
print cv_results
classifier = trainClassifier(trainData)
print "Loading London data"
loadApplicationData(londonPath)
print "Computing anger levels!"
angerLevels = findAngerLevels(londonTweetData, classifier)
anger_peaks = None
print anger_peaks
