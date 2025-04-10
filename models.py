from sqlalchemy import Column, Integer, String, Float, DateTime
from sqlalchemy.ext.declarative import declarative_base
from database import Base



class NewsArticle(Base):
    __tablename__ = "news_articles"
    
    id = Column(Integer, primary_key=True, index=True)
    title = Column(String, unique=True, nullable=False)
    date = Column(DateTime)
    vader_sentiment = Column(String)
    vader_compound = Column(Float)
    textblob_sentiment = Column(String)
    textblob_polarity = Column(Float)
    combined_sentiment = Column(Float)


