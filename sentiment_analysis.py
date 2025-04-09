from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from serpapi import GoogleSearch
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
from sqlalchemy.orm import Session


params = {
    "api_key": "e7cce04dc81f518b1b49a4b778a0c71ca7956e011710ed7ce06155f8765185c0",
    "engine": "google_news",
    "hl": "en",
    "q": "trump"
}
#get the news data
def get_data(parameters):
    search = GoogleSearch(parameters)
    results = search.get_dict()
    return results

#append data to dictionary and sort by date
def data_to_dict(data):
    stories = []
    for story in data['news_results']:
        if 'date' in story:
            stories.append({'title': story['title'], 'date': str(story['date'])})
    stories = (sorted(stories, key=lambda x: x['date']))
    stories.reverse()
    return stories

#calculate relevance
def calculate_relevance(data):
    divider = 1
    for story in data:
        relevanceScore = 1 #reset relevance score
        divider *= 0.995
        relevanceScore *= divider
        if relevanceScore >= 0: relevanceScore +=1
        else: relevanceScore -= 1
        #relevanceScore = round(relevanceScore,4)    
        story['relevance'] = relevanceScore
    return data

#VADER sentiment analysis
def vader_sentiment_analysis(data):
    SIA = SentimentIntensityAnalyzer()
    for x in data:
        sentiment = SIA.polarity_scores(x['title'])
        relativeCompound = sentiment['compound'] * x['relevance']
        x['sentiment'] = sentiment
        x['relative_compound'] = relativeCompound
    return data

def vader_sentiment_array(data):
    sentimentArray, positiveSentimentsArray, neutralSentimentsArray, negativeSentimentsArray = [],[],[], []
    for x in data:
        if x['relative_compound'] >= 0.05: 
            positiveSentimentsArray.append(x['relative_compound'])
        elif -0.05 < x['relative_compound'] < 0.05:
            neutralSentimentsArray.append(x['relative_compound'])
        else:
            negativeSentimentsArray.append(x['relative_compound'])
        sentimentArray.append(x['relative_compound'])
    return sentimentArray, positiveSentimentsArray, neutralSentimentsArray, negativeSentimentsArray
    


#TEXTBLOLB sentiment analysis



#function getting the news data
data = get_data(params)
data_dictionary = data_to_dict(data)
data_dictionary = calculate_relevance(data_dictionary)
data_dictionary = vader_sentiment_analysis(data_dictionary)
sentimentArray, positiveSentimentArray, neutralSentimentArray, negativeSentimentArray = vader_sentiment_array(data_dictionary)

print("Positive: " + str(len(positiveSentimentArray)))
print("Neutral: " + str(len(neutralSentimentArray)))
print("Negatives: " + str(len(neutralSentimentArray)))

datesArray = []
for j in data_dictionary:
    datesArray.append(j['date'])

plt.plot(datesArray, sentimentArray)
plt.axhline(y=0, color='gray', linestyle='--')  # Neutral line
plt.title("Sentiment Over Time: Trump")
plt.show()


#what i want to do, now store the sentiment, have a sentiment TREND tracker, alert on spikes in the sentiment trend, 
