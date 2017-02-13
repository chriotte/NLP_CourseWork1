#=============================================================================#
# Part II
#=============================================================================#
# coding: utf-8
import unicodecsv			# csv reader
import time
import re
from datetime import datetime
from nltk.probability import ConditionalFreqDist
from nltk.probability import ConditionalProbDist, LaplaceProbDist, MLEProbDist
from nltk.tokenize    import TweetTokenizer
from nltk.stem import PorterStemmer
import operator
from nltk.corpus import stopwords

timeStart = time.time()
#=============================================================================#
# lOAD DATA
#=============================================================================#
def loadApplicationData(path):
     with open(path, 'rb') as f:
         reader = unicodecsv.reader(f, encoding='utf-8')
         next(reader, None)  
         for line in reader:
             (date,tweet) = parseTweet(line)
             if tweet:
                 tokenizedTweets = preProcess(tweet)
                 tempString = []
                 date = simplifyDate(date)
                 for items in tokenizedTweets:
                     items = cleanTweet(items)
                     tempString.append(items)
                 londonTweetData.append([tempString, date])
                 
def cleanTweet(items):
     items = items.lower()
     items = items.strip('?!.,/|\;:% ')
     items = items.replace(" ", "")
     items = re.sub(r'^https?:\/\/.*[\r\n]*', '__URL__', items, flags=re.MULTILINE)
     items = re.sub("[^a-z#@][\s]","",items)
     ps = PorterStemmer()
     items = ps.stem(items)
     if items:
         return items

def parseTweet(tweetLine):
     tweet = tweetLine[4]
     date  = datetime.strptime(tweetLine[1], "%Y-%m-%d %H:%M:%S")
     return (date, tweet)
     
def preProcess(text): 
     tknzr = TweetTokenizer()
     tokens = tknzr.tokenize(text)
     return tokens

def getTweets(inputTweets):
    tweets = []
    for tweet in inputTweets:
        tweets.append(tweet[0])
    return tweets

def simplifyDate(date):
    newDate = date.replace(minute=0, second=0)
    return newDate

def fiveAndNineJan(londonTweetData):
    for tweet in londonTweetData:
        if tweet[1].day == 5:
            londonTweetData5.append(tweet)
        elif tweet[1].day == 9:
            londonTweetData9.append(tweet)

def findUniqueBigrams(bigramData):
    uniqueBigrams = {}
    for bigram in bigramData:
        if bigram not in uniqueBigrams:
            uniqueBigrams[bigram] = 1
        else:
            uniqueBigrams[bigram] += 1
    return uniqueBigrams

def getBigrams(tweets):
    bigrams = []
    for tweet in tweets:# if there is more than one element in the list
        for word in range(len(tweet)-1):
            bigrams.append((tweet[word], tweet[word+1]))
    return bigrams

# Count number of occurrences of bigram (x,y), in a given dataset
def countBigrams(x, y, dataset):
    bigram = (x,y)
    dataset = getBigrams(getTweets(dataset))
    count = 0
    for items in dataset:
        if items == bigram:
            count += 1
    print(count, "bigrams found that match(", x, ",", y, ")")

def timeElapsed():
    timeNow         = time.time()
    elapsedStart    = timeNow - timeStart
    print("   >> Elapsed time: ", "{}".format(elapsedStart))
    print("   >> Elapsed time: ", int(elapsedStart))


def findXCommonBigrams(bigramData, x):
    sorted_x = sortDic(bigramData)
    sorted_x = sorted_x[0:int((len(bigramData)*(x/100)))]
    return sorted_x

def sortDic(dictionary):
    sorted_x = sorted(dictionary.items(), key=operator.itemgetter(1), reverse = True)
    return sorted_x

def getBigramsFromDic(dictionary):
    bigramList = []
    for item in dictionary:
        bigramList.append(item)
    return bigramList
        
#=============================================================================#
# A bigram model using the NLTK built-in functions
#=============================================================================#
# conditionalProbDist will return a probability distribution over a list of
# bigrams, together with a specified probability distribution constructor
def conditionalProbDist(probDist, bigrams):
	cfDist = ConditionalFreqDist(bigrams)
	cpDist = ConditionalProbDist(cfDist, probDist, bins=len(bigrams))
	return cpDist
 
def quickMLE(tweets):
    bigrams = getBigrams(getTweets(tweets)) 
#    condProbDist = conditionalProbDist(MLEProbDist, bigrams)
    condProbDist = conditionalProbDist(LaplaceProbDist, bigrams)
    return condProbDist

def getRatio():
    MLElondon  = quickMLE(londonTweetData)
    MLElondon9 = quickMLE(londonTweetData9)
    MLElondon5 = quickMLE(londonTweetData5)  
    print("FOR KATE:")
    print(MLElondon['tube'].prob('strike'))
    
    
    # Get ratios jan9 and store in variable Ratio
    for bigrams in uniqueBigrams9:
        prob        = MLElondon9[bigrams[0]].prob(bigrams[1])
        probFULL    = MLElondon[bigrams[0]].prob(bigrams[1])
        
        ratio = prob - probFULL
        ratio9[bigrams] = ratio
    
    # Get ratios jan5
    for bigrams in uniqueBigrams5:
        prob = MLElondon5[bigrams[0]].prob(bigrams[1])
        probFULL = MLElondon[bigrams[0]].prob(bigrams[1])
        
        ratio = prob - probFULL
        ratio5[bigrams] = ratio
    
#=============================================================================#
# Intialize variables
#=============================================================================#
londonTweetData     = []
londonTweetData5    = []
londonTweetData9    = []
lastTime = time.time()
ratio5 = {}
ratio9 = {}

path                = 'Data/'
londonPath          = path + 'london_2017_tweets_TINY.csv'  # 5.000 lines
londonPath          = path + 'london_2017_tweets.csv'       # full dataset

print("Loading data")
loadApplicationData(londonPath)  
timeElapsed()

print("Isolating 5th & 9th") 
fiveAndNineJan(londonTweetData)
timeElapsed()

print("Finding unique bigrams")
uniqueBigramsData5  = findUniqueBigrams(getBigrams(getTweets(londonTweetData5)))
uniqueBigramsData9  = findUniqueBigrams(getBigrams(getTweets(londonTweetData9)))
uniqueBigrams5      = getBigramsFromDic(uniqueBigramsData5)
uniqueBigrams9      = getBigramsFromDic(uniqueBigramsData9)
timeElapsed()

#=============================================================================#
# Main Function
#=============================================================================#

def mainScript():
    getRatio()
     
#=============================================================================#
# Comment out to disable 
#=============================================================================#
print("Running main function")
mainScript()

sortedRatio5 = sortDic(ratio5)
sortedRatio9 = sortDic(ratio9)


#=============================================================================#
# Track time
#=============================================================================#
print("**************************************************")
print("Script finished:")
timeElapsed()