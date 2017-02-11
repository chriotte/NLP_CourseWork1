# coding: utf-8
import unicodecsv			# csv reader
import heapq
from datetime import datetime
from random import shuffle

from sklearn.svm         import LinearSVC
from sklearn.metrics     import precision_recall_fscore_support
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn             import svm

import nltk
from nltk.classify       import SklearnClassifier
from nltk.tokenize       import word_tokenize
from nltk.stem           import WordNetLemmatizer
from nltk.corpus         import stopwords

##################################
######### DATA LOADING AND PARSING
##################################
def loadData(path, label):
    with open(path, 'rb') as f:
        reader = unicodecsv.reader(f, encoding='utf-8')
        next(reader, None)  
#        reader.next()
        for line in reader:
            (date,tweet) = parseTweet(line)
            if tweet:   #if tweet is an empty string python reads it as False 
                tokenizedTweets     = preProcess(tweet)
                tweetFeatureVector = toFeatureVector(tokenizedTweets)
                tweetData.append((date,tokenizedTweets,label))      # (date, [word1, word2, word3], label)
                trainData.append((tweetFeatureVector,label))        # ({Word1: count, Word2: count}, label)
             
# load application data
def loadApplicationData(path):
    with open(path, 'rb') as f:
        reader = unicodecsv.reader(f, encoding='utf-8')
        next(reader, None)  
#        reader.next()
        for line in reader:
            countTweets("Tweet")
            (date,tweet) = parseTweet(line)
            if tweet:   #if tweet is an empty string python reads it as False
                if date.day == 9:
                    countTweets("lol")
                    tokenizedTweets = preProcess(tweet)
                    tweetFeatureVector = toFeatureVector(tokenizedTweets)
                    londonTweetData.append([tweetFeatureVector, date])
                    
# TEXT PREPROCESSING AND FEATURE VECTORIZATION
def preProcess(text): 
    tokens = word_tokenize(text)
    return tokens
    
def parseTweet(tweetLine):
    tweet = tweetLine[4]
    date  = datetime.strptime(tweetLine[1], "%Y-%m-%d %H:%M:%S")

#    if tweetLine[1] == 'created_at':
#        date = datetime.strptime("01/01/2017 00:00:00", "%d/%m/%Y %H:%M:%S")
#    elif tweetLine[1]:
#        date  = datetime.strptime(tweetLine[1], "%d/%m/%Y %H:%M:%S")
#    else:
#        date = tweetLine[1]
    return (date, tweet)
        
def countTweets(tweetType):
    if tweetType == "Tweet":
        tubeTweetCount.append(1)
    else:
        septTweetCount.append(1)
          
def toFeatureVector(words):
    featureDict = {}
    for word in words:
        unimportant_words = [':', 'http', '.', ',', '?', '...', "'s", "n't", 'RT', ';', '&', ')', '``', 'u', '(', "''", '|',]
        ENGstopwords = stopwords.words('english')
        word = WordNetLemmatizer().lemmatize(word)

#        unimportant_words = []
#        ENGstopwords = []

        if word not in ENGstopwords:
            if word not in unimportant_words:
                if word[0:2] != '//':
                    if isinstance(word, str):
################################################################
                        if word not in featureDict:
                            featureDict[word] = 1
                        else:
                            featureDict[word] += 1
                        if word not in compleateFeatureDic:
                            compleateFeatureDic[word]    = 1
                        else:
                            compleateFeatureDic[word]    += 1    
################################################################
    return featureDict

## This should be a complete function 
def trainClassifier(trainData):
    classifier = SklearnClassifier(LinearSVC())
#    classifier = SklearnClassifier(BernoulliNB())
#    classifier = SklearnClassifier(svm.SVC())
#    classifier = SklearnClassifier(MultinomialNB())

    result = classifier.train(trainData)
    return result

# TRAINING AND VALIDATING OUR CLASSIFIER
def crossValidate(dataset, folds):
   # print(dataset)
    shuffle(dataset)
    subset_size = int(len(dataset)/folds)

#    cv_angryTestTotal   = 0
#    cv_happyTestTotal   = 0
#    cv_angryPredTotal   = 0
#    cv_happyPredTotal   = 0
#    cv_count            = 0
    
    cv_accuracy   = []
    cv_precision  = []
    cv_recall     = []
    cv_fscore     = []
    
    for i in range(folds):
        print("*************************")
        print("Cross validation fold: ",i+1)
        testing_this_round  = dataset[i*subset_size:][:subset_size]
        training_this_round = dataset[:i*subset_size] + dataset[(i+1)*subset_size:]
        classifier = trainClassifier(training_this_round)
        prediction = predictLabels(testing_this_round,classifier)
        
###METRICS
#        cv_angryTest        = 0
#        cv_happyTest        = 0
#        cv_angryPred        = 0
#        cv_happyPred        = 0
#        cv_count            = 0
#        for testLabel in testing_this_round:
#            cv_count += 1
#            if testLabel[1] == 'angry':
#                cv_angryTest        +=1
#                cv_angryTestTotal   +=1
#            else:
#                cv_happyTest        +=1
#                cv_happyTestTotal   +=1
#    
#        for x in prediction:
#            cv_count += 1
#            if x == 'angry':
#                cv_angryPred        +=1
#                cv_angryPredTotal   +=1
#            else: 
#                cv_happyPred        +=1
#                cv_happyPredTotal   +=1
        #print("prediction: ",prediction)
        trueLabels  = [x[1] for x in testing_this_round]
        metrics     = precision_recall_fscore_support(trueLabels, prediction, average='macro')
        accuracy    = nltk.classify.accuracy(classifier, testing_this_round)
        precision   = metrics[0]
        recall      = metrics[1]
        fscore      = metrics[2]
#        print("cv_happyTest:    ", cv_happyTest)
#        print("cv_happyPred:    ", cv_happyPred)
#        print("cv_angryTest:    ", cv_angryTest)
#        print("cv_angryPred:    ", cv_angryPred)
        print("accuracy:  ", "{:.4}".format(accuracy))       
        print("precision: ", "{:.4}".format(precision))
        print("recall:    ", "{:.4}".format(recall))
        print("fscore:    ", "{:.4}".format(fscore))
        print()
        cv_accuracy.append(accuracy)
        cv_precision.append(precision)
        cv_recall.append(recall)
        cv_fscore.append(fscore)
    print("***************************")
    print("****** TOTAL METRICS ******")
    print("***************************")
#    print("cv_happyTestTotal:    ",cv_happyTestTotal)
#    print("cv_happyPredTotal:    ",cv_happyPredTotal)
#    print("cv_angryTestTotal:    ",cv_angryTestTotal)
#    print("cv_angryPredTotal:    ",cv_angryPredTotal)
    print("Accuracy:  ",  "{:.4}".format(sum(cv_accuracy)/len(cv_accuracy)))
    print("Precision: ",  "{:.4}".format(sum(cv_precision)/len(cv_precision)))
    print("Recall:    ",  "{:.4}".format(sum(cv_recall)/len(cv_recall)))
    print("Fscore:    ",  "{:.4}".format(sum(cv_fscore)/len(cv_fscore)))
        # find mean accuracy over all rounds
        
# PREDICTING LABELS GIVEN A CLASSIFIER
def predictLabels(tweetData, classifier):
	return classifier.classify_many(map(lambda t: t[0], tweetData))

#Use later, if we need to evaluate single words
def predictLabel(text, classifier):
	return classifier.classify(text)

# COMPUTING ANGER LEVEL ON A SET OF TWEETS

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
    tenHighest = heapq.nlargest(10, angerLevels, key=angerLevels.get)
    return tenHighest
    

# tweets will have one feature, which is going to be a numerical value
angryLabel  =    'angry'
happyLabel  =    'happy'
path        =    'Data/'
angryPath   = path + 'angry_tweets.csv'
happyPath   = path + 'happy_tweets.csv'
londonPath  = path + 'london_2017_tweets.csv'

# Use the below for testing 
angryPath   = path + 'angry_tweets_500.csv'
happyPath   = path + 'happy_tweets_500.csv'
#londonPath  = path + 'london_2017_tweets_500.csv'
londonPath  = path + 'london_2017_tweets_NEW.csv'


########################################################
######################### MAIN #########################
########################################################
tweetData           = []
trainData           = []                  # [({TweetWords: count}, label)]
londonTweetData     = []            # Tweets from the london tube strike
tubeTweetCount      = []
septTweetCount      = []
compleateFeatureDic = {}
angryLevel          = {}

####### Load data
print ("Loading happy")
loadData(happyPath, happyLabel)
print("Loading angry")
loadData(angryPath, angryLabel)
print ("Loading London data...")
loadApplicationData(londonPath) 
print("All data loaded successfully")
print("**************************************************")
print ('Number of words in tweetData: ' + str(len(tweetData)))
print ('Number of words in trainData: ' + str(len(trainData)))
print ('Number of words in londonTweetData: ' + str(len(londonTweetData)))
print("**************************************************")

####### Cross validation
cv_results = crossValidate(trainData, 10)
print("************************")
print("CrossValidation complete")
print("************************")

####### Train classifier to use on London Data
print("Training Classifier on all trainData")
classifier = trainClassifier(trainData)

####### Compute anger from london data
print ("Computing anger levels!")
angerLevels = findAngerLevels(londonTweetData, classifier)
anger_peaks = None

print("angerLevels: ")

for x in angerLevels:
    print(x, ":", "{:.4}".format(angerLevels[x]) )

print (anger_peaks)
print("Tube tweets loaded: ", sum(tubeTweetCount))
print("Tube tweets on 9. Sept: ", sum(septTweetCount))
print("**************************************************")

"""
datetime.strptime("2016-01-05 12:02:03", "%Y-%m-%d %H:%M:%S")
Out[7]: datetime.datetime(2016, 1, 5, 12, 2, 3)

a = datetime.strptime("2016-01-05 12:02:03", "%Y-%m-%d %H:%M:%S")

a.hour
Out[9]: 12

a.second
Out[10]: 3
"""