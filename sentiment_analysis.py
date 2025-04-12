from serpapi import GoogleSearch
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sqlalchemy.orm import Session
from database import SessionLocal
from textblob import TextBlob

db: Session = SessionLocal()

# Get the news data
def get_data(parameters):
    search = GoogleSearch(parameters)
    results = search.get_dict()
    return results


#append data to dictionary and sort by date
def data_to_dict(data):
    stories = []
    for story in data.get('news_results', []):
        if 'date' in story:
            stories.append({'title': story['title'], 'date': str(story['date'])})
    stories = sorted(stories, key=lambda x: x['date'])
    stories.reverse()
    return stories

#VADER sentiment analysis
def sentiment_analysis(data):
    SIA = SentimentIntensityAnalyzer()    
    for x in data:
        sentimentTextBlob = TextBlob(x['title'])
        sentimentVader = SIA.polarity_scores(x['title'])
        textblob_dict = {
            'polarity': sentimentTextBlob.sentiment.polarity,
            'subjectivity': sentimentTextBlob.sentiment.subjectivity
        }
        x['vader_sentiment'] = str(sentimentVader)
        x['vader_compound'] = sentimentVader['compound']
        x['textblob_sentiment'] = str(textblob_dict)
        x['textblob_polarity'] = textblob_dict['polarity']
        x['combined_sentiment'] = (0.7 * float(sentimentVader['compound'])) + (0.3 * float(textblob_dict['polarity']))
    return data


