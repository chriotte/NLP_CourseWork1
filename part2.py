
# coding: utf-8
import unicodecsv			# csv reader
import time
from datetime import datetime
from nltk.probability import ConditionalFreqDist
from nltk.probability import ConditionalProbDist, LaplaceProbDist, MLEProbDist
from nltk.tokenize       import TweetTokenizer

timeStart = time.time()
#=============================================================================#
# Load and process 
#=============================================================================#
def loadApplicationData(path):
     with open(path, 'rb') as f:
         reader = unicodecsv.reader(f, encoding='utf-8')
         next(reader, None)  
         for line in reader:
             (date,tweet) = parseTweet(line)
             if tweet:  
                 tokenizedTweets = preProcess(tweet)
                 date = simplifyDate(date)
                 londonTweetData.append([tokenizedTweets, date])
                     
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
        
londonTweetData     = []
londonTweetData5    = []
londonTweetData9    = []

path                = 'Data/'
londonPath          = path + 'london_2017_tweets_TINY.csv'  # 5.000 lines
londonPath          = path + 'london_2017_tweets.csv'       # full dataset

loadApplicationData(londonPath)   
fiveAndNineJan(londonTweetData)
#=============================================================================#
# A bigram model using the NLTK built-in functions
#=============================================================================#
def getBigrams(tweets):
    bigrams = []
    for tweet in tweets:# if there is more than one element in the list
        for word in range(len(tweet)-1):
            bigrams.append((tweet[word], tweet[word+1]))
    return bigrams

# conditionalProbDist will return a probability distribution over a list of
# bigrams, together with a specified probability distribution constructor
def conditionalProbDist(probDist, bigrams):
	cfDist = ConditionalFreqDist(bigrams)
	cpDist = ConditionalProbDist(cfDist, probDist, bins=len(bigrams))
	return cpDist
 
def quickMLE(tweets):
    bigrams = getBigrams(getTweets(tweets)) 
    condProbDist = conditionalProbDist(MLEProbDist, bigrams)
    return condProbDist

def countBigrams(x, y, dataset):
    bigram = (x,y)
    dataset = getBigrams(getTweets(dataset))
    count = 0
    for items in dataset:
        if items == bigram:
            count += 1
    print(count, "bigrams found that match", x, "and", y)
            
# this is the function where you can put your main script, which you can then
# toggle if for test purposes
def mainScript():
    
    print("Probabilities of bigram: ('Tube','strike')")
    londonTweet  = quickMLE(londonTweetData)
    londonTweet5 = quickMLE(londonTweetData5)
    londonTweet9 = quickMLE(londonTweetData9)
    countBigrams('Tube','strike',londonTweetData)
    countBigrams('Tube','strike',londonTweetData5)
    countBigrams('Tube','strike',londonTweetData9)
    print("londonTweet :",londonTweet["tube"].prob("strike"))
    print("5 Jan       :",londonTweet5["tube"].prob("strike"))
    print("9 Jan       :",londonTweet9["tube"].prob("strike"))
     
#=============================================================================#
# Comment out to disable main function. 
#=============================================================================#

mainScript()

#=============================================================================#
# Track time
#=============================================================================#
print("**************************************************")
timeEnd = time.time()
elapsed = timeEnd-timeStart
print("Elapsed seconds:", int(elapsed))