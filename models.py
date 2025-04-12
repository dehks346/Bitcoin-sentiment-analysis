from sqlalchemy import Column, Integer, String, Float, DateTime, Boolean
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


class TrainingData(Base):
    __tablename__ = 'training_data'
    
    id = Column(Integer, primary_key=True, index=True)
    date = Column(DateTime)
    vader_score = Column(Float)
    textblob_score = Column(Float)
    combined_sentiment = Column(Float)
    sentiment_momentum = Column(Float)
    
    
    btc_price = Column(Float)
    btc_volume = Column(Float)
    price_volatility = Column(Float)
    
    next_day_prediction = Column(Boolean)
    
    total_articles = Column(Integer, default=0)  

