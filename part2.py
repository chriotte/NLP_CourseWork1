
# coding: utf-8

from nltk.probability import ConditionalFreqDist
from nltk.probability import ConditionalProbDist, LaplaceProbDist, MLEProbDist

##############
### PART B ###
##############

####
## A bigram model using the NLTK built-in functions
####

# given a list of lists of preprocessed tweets,
# getBigrams should return a list of pairs containing all the bigrams that
# are observed in the list.

def getBigrams(tweets):
    bigrams = []
    for tweet in tweets:# if there is more than one element in the list
        curTweetBigram_list = []
        for word in range(len(tweet)-1):
            curTweetBigram_list.append((tweet[word], tweet[word+1]))
        bigrams.append(curTweetBigram_list)
        
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
bigrams = getBigrams(inputList)
print(bigrams)