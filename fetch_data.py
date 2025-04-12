#!/usr/bin/env python3
import logging
from datetime import datetime, timedelta
from sqlalchemy import func
from database import SessionLocal
from models import NewsArticle, TrainingData
from sentiment_analysis import get_data, data_to_dict, sentiment_analysis
import yfinance as yf
import asyncio


logging.basicConfig(
    level=logging.INFO,
    filename='/Users/henry/Documents/datasi/fetch_data.log',
    filemode='a',
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


print("test")

def get_unproccessed_data():
    params = {"api_key": "e7cce04dc81f518b1b49a4b778a0c71ca7956e011710ed7ce06155f8765185c0", "engine": "google_news", "hl": "en", "q": "bitcoin"}
    try:
        data = get_data(params)
        data = data_to_dict(data)
        return data
    except Exception as e:
        logger.error(f'Error fetching google news data: {e}')
        return[]


async def fetch_and_store_data():
    try:
        data = get_unproccessed_data()
        if not data:
            logger.info('No data fetched from google news')
            return 0
        titles = [item['title'] for item in data]
        
        with SessionLocal() as db:
            existing_titles = {row[0] for row in db.query(NewsArticle.title).filter(NewsArticle.title.in_(titles)).all()}        
            new_articles = [item for item in data if item['title'] not in existing_titles]
        
            seen_titles = set()
            deduped_new_articles = []
            for item in new_articles:
                if item['title'] not in seen_titles:
                    deduped_new_articles.append(item)
                    seen_titles.add(item['title'])
                else:
                    print(f"Skipping duplicate:  {item['title']}")
                    
            processed_new_articles = sentiment_analysis(deduped_new_articles)
            
            stored_count = 0
            for x in processed_new_articles:
                try:
                    pub_date = datetime.strptime(x['date'], "%m/%d/%Y, %I:%M %p, +0000 UTC")
                except ValueError as e:
                    print(f"Skipping article '{x['title']}' due to invalid date: {x['date']} - {e}")
                    continue
                    
                article = NewsArticle(
                    title = x['title'],
                    date = pub_date,
                    vader_compound = x['vader_compound'],
                    vader_sentiment = x['vader_sentiment'],
                    textblob_sentiment = x['textblob_sentiment'],
                    textblob_polarity = x['textblob_polarity'],
                    combined_sentiment = x['combined_sentiment']
                )
                db.add(article)
                db.commit()
                db.refresh(article)
                stored_count += 1
        logger.info(f"Stored {stored_count} new articles out of {len(titles)} fetched")
        return stored_count
    except Exception as e:
        logger.error(f'Error in fetching and storing google news data: {e}')
        return 0




async def fetch_and_store_training_data():
    try:
        with SessionLocal() as db:
            today = datetime.now().date()
            articles = db.query(NewsArticle).filter(func.date(NewsArticle.date) == today).all()
            
            if not articles: return {"messages": 'no articles found for today'}
            
            average_vader = sum(a.vader_compound for a in articles) / len(articles)
            average_textblob = sum(a.textblob_polarity for a in articles) / len(articles)
            average_combined_sentiment = sum(a.combined_sentiment for a in articles) / len(articles)
            
            btc_data = yf.Ticker('BTC-GBP').history(period='2d')
            today_close = btc_data['Close'].iloc[-1]
            yesterday_close = btc_data['Close'].iloc[-2]
            daily_return = (today_close - yesterday_close) / yesterday_close
            
            record = db.query(TrainingData).filter(func.date(TrainingData.date) == today).first()
            
            if not record:
                record = TrainingData(
                    date=datetime.now(),
                    vader_score = average_vader,
                    textblob_score = average_textblob,
                    combined_sentiment = average_combined_sentiment,
                    sentiment_momentum = 0,
                    
                    btc_price = today_close,
                    btc_volume= btc_data['Volume'].iloc[-1],
                    price_volatility = btc_data['Close'].std(),
                    next_day_prediction=None,
                    total_articles=len(articles)
                )
                db.add(record)
            else:
                record.vader_score = average_vader
                record.textblob_score = average_textblob
                record.combined_sentiment = average_combined_sentiment
                record.btc_price = today_close
                record.btc_volume = btc_data['Volume'].iloc[-1]
                record.total_articles = len(articles)
            db.commit()
            db.refresh(record)
            logger.info('Training data stored successfully')
            return True
    except Exception as e:
        logger.error(f'Error in fetching daily summary: {e}')
        return False


async def main():
    logger.info('Starting data fetch and store process')
    articles_stored = await fetch_and_store_data()
    training_stored = await fetch_and_store_training_data()
    logger.info(f"Process completed: {articles_stored} articles stored, training data {'stored' if training_stored else 'not stored'}")
    
if __name__ == '__main__':
    asyncio.run(main())
