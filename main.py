from fastapi import FastAPI, Depends, Request
from database import SessionLocal, engine
from models import NewsArticle, Base
from sentiment_analysis import get_data, data_to_dict, calculate_relevance, vader_sentiment_analysis
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
import plotly.express as px
from jinja2 import Template
from sqlalchemy import func
from datetime import datetime





app = FastAPI()

templates = Jinja2Templates(directory='templates')

#Base.metadata.drop_all(bind=engine)
#Base.metadata.create_all(bind=engine)
#print("Database reset successfully!")




@app.get('/fetch-and-store-data')
async def fetch_and_store_data():
    
    data = get_unproccessed_data()
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
    
        processed_new_articles = calculate_relevance(deduped_new_articles) 
        processed_new_articles = vader_sentiment_analysis(processed_new_articles)
        
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
                sentiment = x.get('sentiment', {}).get('compound', 0),
                relevance = x['relevance'],
                relative_sentiment = x['relative_compound'],
            )
            db.add(article)
            db.commit()
            db.refresh(article)
            stored_count += 1
    return {"message": f"Stored {stored_count} new articles out of {len(titles)} fetched"}
@app.get('/display-data', response_class=HTMLResponse)
async def display_data(request: Request):
    
    with SessionLocal() as db:
        articles = db.query(NewsArticle).all()
        
        
    return templates.TemplateResponse('data.html', {'request': request, 'articles': articles})

@app.get('/graph', response_class=HTMLResponse)
async def graph(request: Request):
    with SessionLocal() as db:
        # Query articles ordered by date
        data = db.query(NewsArticle.date, NewsArticle.sentiment).order_by(NewsArticle.date).all()

    if not data:
        return templates.TemplateResponse(
            "graph.html",
            {"request": request, "graph": "<p>No data available</p>"}
        )

    # Extract dates and sentiment scores
    dates = [row[0] for row in data]
    sentiments = [row[1] for row in data]

    # Create a line plot
    fig = px.line(
        x=dates,
        y=sentiments,
        labels={"x": "Publication Date", "y": "Sentiment Score"},
        title="Sentiment Over Time"
    )
    # Add a horizontal line at y=0 for neutral sentiment
    fig.add_hline(y=0, line_dash="dash", line_color="gray")

    # Calculate relative sentiment (e.g., rolling average)
    # For simplicity, we'll use a basic moving average here
    window_size = 5
    rolling_sentiment = [
        sum(sentiments[max(0, i - window_size + 1):i + 1]) / min(i + 1, window_size)
        for i in range(len(sentiments))
    ]
    fig.add_scatter(
        x=dates,
        y=rolling_sentiment,
        mode="lines",
        name="Rolling Average Sentiment",
        line=dict(color="red")
    )

    graph_html = fig.to_html(full_html=False)

    return templates.TemplateResponse(
        "graph.html",
        {"request": request, "graph": graph_html}
    )


def get_unproccessed_data():
    params = {
    "api_key": "e7cce04dc81f518b1b49a4b778a0c71ca7956e011710ed7ce06155f8765185c0",
    "engine": "google_news",
    "hl": "en",
    "q": "bitcoin"
    }
    data = get_data(params)
    data = data_to_dict(data)
    return data
