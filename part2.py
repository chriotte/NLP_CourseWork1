
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
                 if (date.day == 9 or date.day == 5) and date.year == 2017 and date.month == 1 :
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
        tweets.append(inputTweets[0])
    print(tweets)
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

# this is the function where you can put your main script, which you can then
# toggle if for test purposes
def mainScript():
    bigrams = getBigrams(getTweets(londonTweetData)) 
    condProbDist = conditionalProbDist(MLEProbDist, bigrams)
    print(condProbDist)
    testProb = quickMLE(londonTweetData)
    print(testProb)
    
    
    print("Probabilities of bigram: ('Tube','strike')")
    testProb = condProbDist["tube"].prob("strike")
    print(testProb)
     

# The line below can be toggled as a comment to toggle execution of the main script
mainScript()


















print("**************************************************")
timeEnd = time.time()
elapsed = timeEnd-timeStart
print("Elapsed seconds:", int(elapsed))