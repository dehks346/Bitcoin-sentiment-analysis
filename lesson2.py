import json
from datetime import datetime, timedelta
from textblob import TextBlob
import statistics
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from serpapi import GoogleSearch
import matplotlib.pyplot as plt
import numpy as np


params = {
    "api_key": "e7cce04dc81f518b1b49a4b778a0c71ca7956e011710ed7ce06155f8765185c0",
    "engine": "google_news",
    "hl": "en",
    "q": "trump"
}

SIA = SentimentIntensityAnalyzer()
search = GoogleSearch(params)
results = search.get_dict()


stories = []
averagePolarity = []

#adding data to a dictionary 
for story in results['news_results']:
    if 'date' in story:
        date = story['date']
        title = (story['title'])
        sentiment = SIA.polarity_scores(title)
        stories.append({'title': title, 'date': date[:10], 'sentiment': sentiment})
        averagePolarity.append(sentiment['compound'])


#sorting data by date
stories = (sorted(stories, key=lambda x: x['date']))
stories.reverse()

relevanceScore = 1
relevanceDivider = len(stories) / (10 ** len(str(abs(len(stories))))) #shifts number to decimal place
averageRelativePolarity = []
negativeSentiments, positiveSentiments, neutralSentiments = 0,0,0

#add a relevance score based on time
for each_story in stories:
    relevanceScore = 1
    relevanceDivider *= 0.998
    relevanceScore *= relevanceDivider
    relevanceScore = round(relevanceScore,4)
    if relevanceScore >= 0: relevanceScore +=1
    else: relevanceScore -= 1
    each_story['relevance_score'] = relevanceScore
    relativePolarity = each_story['sentiment']['compound'] * relevanceScore
    each_story['relative_polarity'] = relativePolarity
    if relativePolarity >= 0.05:
        print('Positive Sentiment')
        positiveSentiments +=1
    elif -0.05 < relativePolarity < 0.05:
        print('Neutral sentiment')
        neutralSentiments += 1
    elif relativePolarity <= -0.05: 
        print('Negative sentiment')
        negativeSentiments +=1
    else: print('Sentiment Analysis Ran into an Error')
    averageRelativePolarity.append(relativePolarity)
    
    
    

averageRelativePolarity = sum(averageRelativePolarity) / len(averageRelativePolarity)

print("Positive: " + str(positiveSentiments))
print("Neutral: " + str(neutralSentiments))
print("Negatives: " + str(neutralSentiments))
x,y=[],[]
for d in stories:
    x.append(d['relative_polarity'])
    y.append(d['relevance_score'])

plt.scatter(x,y)
plt.show()

#x = np.array(stories[])
#y = np.array([100,105,84,105,90,99,90,95,94,100,79,112,91,80,85])
#plt.scatter(x, y)

#plt.show()