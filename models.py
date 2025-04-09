from sqlalchemy import Column, Integer, String, Float, Date
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()

class NewsArticle(Base):
    __tablename__ = "news_articles"
    
    id = Column(Integer, primary_key=True, index=True)
    title = Column(String)
    date = Column(Date)
    sentiment = Column(Float)
    relevance = Column(Float)
    relative_sentiment = Column(Float)


class SentimentStats(Base):
    __tablename__ = "SentimentStats"
    
    id = Column(Integer, primary_key=True, index=True)
    sentimentArray = Column(String)
    positiveSentimentsArray = Column(String)
    neutralSentimentsArray = Column(String)
    negativeSentimentsArray=  Column(String)
