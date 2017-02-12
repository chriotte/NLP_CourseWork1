
# coding: utf-8
import unicodecsv			# csv reader
from datetime import datetime
from nltk.probability import ConditionalFreqDist
from nltk.probability import ConditionalProbDist, LaplaceProbDist, MLEProbDist
from nltk.tokenize       import TweetTokenizer

#=============================================================================#
def loadApplicationData(path):
     with open(path, 'rb') as f:
         reader = unicodecsv.reader(f, encoding='utf-8')
         next(reader, None)  
         for line in reader:
             (date,tweet) = parseTweet(line)
             if tweet:  
                 if date.day == 9:
                     tokenizedTweets = preProcess(tweet)
                     londonTweetData.append([tokenizedTweets, date])
def parseTweet(tweetLine):
     tweet = tweetLine[4]
     date  = datetime.strptime(tweetLine[1], "%Y-%m-%d %H:%M:%S")
     return (date, tweet)
def preProcess(text): 
     tknzr = TweetTokenizer()
     tokens = tknzr.tokenize(text)
     return tokens

londonTweetData = []

path        = 'Data/'
londonPath  = path + 'london_2017_tweets_TINY.csv'  # 5.000 lines
loadApplicationData(londonPath)

londonOnlyTweets = []
def isolateTweets(londonTweets):
    for tweets in londonTweets:
        londonOnlyTweets.append(tweets[0])
        
isolateTweets(londonOnlyTweets)
#=============================================================================#

####
## A bigram model using the NLTK built-in functions
####

# given a list of lists of preprocessed tweets,
# getBigrams should return a list of pairs containing all the bigrams that
# are observed in the list.

def getBigrams(tweets):
    bigrams = []
    for tweet in tweets:# if there is more than one element in the list
        bigram = []
        for word in range(len(tweet)-1):
            bigram.append((tweet[word], tweet[word+1]))
        bigrams.append(bigram)
    return bigrams

# conditionalProbDist will return a probability distribution over a list of
# bigrams, together with a specified probability distribution constructor
def conditionalProbDist(probDist, bigrams):
	cfDist = ConditionalFreqDist(bigrams)
	cpDist = ConditionalProbDist(cfDist, probDist, bins=len(bigrams))
	return cpDist

# this is the function where you can put your main script, which you can then
# toggle if for test purposes
def mainScript():
	return "TODO"

# The line below can be toggled as a comment to toggle execution of the main script
# results = mainScript()


inputList = [["this","is","fun"],["london","is","great"]]

tweets = londonOnlyTweets
bigrams = getBigrams(tweets)
probDists = MLEProbDist()

condProbDist = conditionalProbDist(probDists, bigrams)

print(condProbDist)