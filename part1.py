#=============================================================================#
# Part I
#=============================================================================#
# coding: utf-8
import unicodecsv			# csv reader
import time
import re
import matplotlib.pyplot as plt
import operator

from datetime import datetime
from random import shuffle

from sklearn.svm         import LinearSVC
from sklearn.metrics     import precision_recall_fscore_support
from sklearn.naive_bayes import MultinomialNB, BernoulliNB

import nltk
from nltk.classify       import SklearnClassifier
from nltk.tokenize       import word_tokenize
from nltk.stem           import WordNetLemmatizer
from nltk.corpus         import stopwords
from nltk.tokenize       import TweetTokenizer

timeStart = time.time() # Start timer
#=============================================================================#
# Load Data
#=============================================================================#
def loadData(path, label):
    with open(path, 'rb') as f:
        reader = unicodecsv.reader(f, encoding='utf-8')
        next(reader, None)  
        for line in reader:
            (date,tweet) = parseTweet(line)
            if tweet:   #if tweet is an empty string python reads it as False 
                tweet = re.sub("[^a-zA-Z0-9 #]","",tweet)
                tweet = re.sub(r'^https?:\/\/.*[\r\n]*', '__URL__', tweet, flags=re.MULTILINE)
            
                tokenizedTweets     = preProcess(tweet)
                tweetFeatureVector  = toFeatureVector(tokenizedTweets)
                tweetData.append((date,tokenizedTweets,label))      # (date, [word1, word2, word3], label)
                trainData.append((tweetFeatureVector,label))        # ({Word1: count, Word2: count}, label)
             
def loadApplicationData(path):
    with open(path, 'rb') as f:
        reader = unicodecsv.reader(f, encoding='utf-8')
        next(reader, None)  
        for line in reader:
            countTweets("Tweet")
            (date,tweet) = parseTweet(line)
            if tweet:   #if tweet is an empty string python reads it as False
                if date.day == 9:
                    countTweets("lol")
                    tokenizedTweets = preProcess(tweet)
                    tweetFeatureVector = toFeatureVector(tokenizedTweets)
                    londonTweetData.append([tweetFeatureVector, date])

#=============================================================================#
# TEXT PREPROCESSING AND FEATURE VECTORIZATION
#=============================================================================#

def preProcess(text): 
    tknzr = TweetTokenizer()
    tokens = tknzr.tokenize(text)
    return tokens
    
def parseTweet(tweetLine):
    tweet = tweetLine[4]
    date  = datetime.strptime(tweetLine[1], "%Y-%m-%d %H:%M:%S")
    return (date, tweet)
        
def countTweets(tweetType):
    if tweetType == "Tweet":
        tubeTweetCount.append(1)
    else:
        septTweetCount.append(1)

def timeElapsed():
    print("   Done...")
    timeEnd = time.time()
    elapsed = timeEnd-timeStart
    print("   Elapsed seconds:", int(elapsed))

#==============================================================================
# Feature vector
#==============================================================================
def toFeatureVector(words):
    featureDict = {}
    for word in words:
        unimportant_words = [':', 'http', '.', ',', '?', '...', "'s", "n't", 'RT', ';', '&', ')', '``', 'u', '(', "''", '|',]
        word = WordNetLemmatizer().lemmatize(word)

# Uncommment for testing -- Faster, but less accurate
        unimportant_words = []

        if word not in unimportant_words:
            if word[0:2] != '//':
                if isinstance(word, str):
                    if word not in featureDict:
                        featureDict[word] = 1
                    else:
                        featureDict[word] += 1
                    if word not in compleateFeatureDic:
                        compleateFeatureDic[word]    = 1
                    else:
                        compleateFeatureDic[word]    += 1    
    return featureDict

#==============================================================================
# Train Classifier & cross-validation
#==============================================================================
def trainClassifier(trainData):
    classifier = SklearnClassifier(LinearSVC())

    result = classifier.train(trainData)
    return result

# TRAINING AND VALIDATING OUR CLASSIFIER
def crossValidate(dataset, folds):
   # print(dataset)
    shuffle(dataset)
    subset_size = int(len(dataset)/folds)
       
    for i in range(folds):
        print("*************************")
        print("Cross validation fold: ",i+1)
        testing_this_round  = dataset[i*subset_size:][:subset_size]
        training_this_round = dataset[:i*subset_size] + dataset[(i+1)*subset_size:]
        classifier = trainClassifier(training_this_round)
        prediction = predictLabels(testing_this_round,classifier)
        
###METRICS
        trueLabels  = [x[1] for x in testing_this_round]
        metrics     = precision_recall_fscore_support(trueLabels, prediction, average='macro')
        accuracy    = nltk.classify.accuracy(classifier, testing_this_round)
        precision   = metrics[0]
        recall      = metrics[1]
        fscore      = metrics[2]
        print(" Accuracy:  ", "{:.4}".format(accuracy), "\n Precision: ", "{:.4}".format(precision))
        print(" Recall:    ", "{:.4}".format(recall),   "\n Fscore:    ", "{:.4}".format(fscore),"\n")
        cv_accuracy.append(accuracy)
        cv_precision.append(precision)
        cv_recall.append(recall)
        cv_fscore.append(fscore)
    print("***************************")
    print("***** AVERAGE METRICS *****")
    print("***************************")
    print(" Accuracy:  ",  "{:.4}".format(sum(cv_accuracy)/len(cv_accuracy)))
    print(" Precision: ",  "{:.4}".format(sum(cv_precision)/len(cv_precision)))
    print(" Recall:    ",  "{:.4}".format(sum(cv_recall)/len(cv_recall)))
    print(" Fscore:    ",  "{:.4}".format(sum(cv_fscore)/len(cv_fscore)))
        # find mean accuracy over all rounds
        
#==============================================================================
# PREDICTING LABELS GIVEN A CLASSIFIER
#==============================================================================
def predictLabels(tweetData, classifier):
	return classifier.classify_many(map(lambda t: t[0], tweetData))

#Use later, if we need to evaluate single words
def predictLabel(text, classifier):
	return classifier.classify(text)

#==============================================================================
# COMPUTING ANGER LEVEL ON A SET OF TWEETS
#==============================================================================
def findAngerLevels(tweetData, classifier):    
    for tweet in tweetData:
        prediction = predictLabel(tweet[0],classifier)
        try:# if its not a proper date
            newDate = tweet[1]
            if newDate.hour in angryLevel:
                angryIncrement = 0
                if prediction == "angry":
                    angryIncrement = 1
                angryLevel[newDate.hour][0] += 1
                angryLevel[newDate.hour][1] += angryIncrement 
            else:
                angryIncrement = 0
                if prediction == "angry":
                    angryIncrement = 1
                angryLevel[newDate.hour] = [1,angryIncrement]
        except:
            pass
    for hour in angryLevel:
        angerRatio = angryLevel[hour][1] / angryLevel[hour][0]
        angryLevel[hour] = angerRatio
    return angryLevel

def findAngerPeaks(angerLevels):
    sorted_x = sorted(angerLevels.items(), key=operator.itemgetter(1), reverse = True)
    sorted_x = sorted_x[0:10]
    return sorted_x
    
#=============================================================================#
# Define paths 
#=============================================================================#
angryLabel  =    'angry'
happyLabel  =    'happy'
path        =    'Data/'
angryPath   = path + 'angry_tweets.csv'
happyPath   = path + 'happy_tweets.csv'
londonPath  = path + 'london_2017_tweets.csv'

## Use the below for testing 
angryPath   = path + 'angry_tweets_500.csv'
happyPath   = path + 'happy_tweets_500.csv'
#londonPath  = path + 'london_2017_tweets_NEW.csv'    # 48.000 lines
#londonPath  = path + 'london_2017_tweets_SMALL.csv'  # 10.000 lines
londonPath  = path + 'london_2017_tweets_TINY.csv'  # 5.000 lines

#==============================================================================
# Initialize variables
#==============================================================================
tweetData           = []
trainData           = []                  # [({TweetWords: count}, label)]
londonTweetData     = []            # Tweets from the london tube strike
tubeTweetCount      = []
septTweetCount      = []
compleateFeatureDic = {}
angryLevel          = {}
anger_peaks         = {}
cv_accuracy         = []
cv_precision        = []
cv_recall           = []
cv_fscore           = []

#==============================================================================
# Load functions
#==============================================================================
print ("Loading happy")
loadData(happyPath, happyLabel)

print("Loading angry")
loadData(angryPath, angryLabel)

print ("Loading London data...")
loadApplicationData(londonPath) 

print("All data loaded successfully")

#==============================================================================
#==============================================================================
#==============================================================================
#                                 MAIN
#==============================================================================
#==============================================================================
#==============================================================================

#==============================================================================
#  CV & Classifier
#==============================================================================
cv_results = crossValidate(trainData, 10)
classifier = trainClassifier(trainData)

#==============================================================================
#  Anger levels
#==============================================================================
print("************************")
print ("Computing anger levels!")
angerLevels = findAngerLevels(londonTweetData, classifier)
anger_peaks = findAngerPeaks(angerLevels)
for x in angerLevels:
    print(x, ":", "{:.4}".format(angerLevels[x]) )
print("************************")
print ("Peaks")
for x in anger_peaks:
    print(x[0], ":","{:.4}".format(x[1]))
print("************************")
print("Tube tweets loaded: ", sum(tubeTweetCount))
print("Tube tweets on 9. Sept: ", sum(septTweetCount))

#==============================================================================
# PLOTTING
#==============================================================================
plt.bar(range(len(angerLevels)), angerLevels.values(), align='center')
plt.xticks(range(len(angerLevels)), list(angerLevels.keys()))
plt.ylim( (0.5, 1))  
plt.show() 

#==============================================================================
# DONE
#==============================================================================
print("**************************************************")
timeEnd = time.time()
elapsed = timeEnd-timeStart
print("Elapsed seconds:", int(elapsed))