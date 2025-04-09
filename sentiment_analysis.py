from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from serpapi import GoogleSearch
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
from sqlalchemy.orm import Session
from database import SessionLocal
from models import NewsArticle
from datetime import datetime

db: Session = SessionLocal()

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
            stories.append({'title': story['title'], 'date': str(story['date'][:10])})
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