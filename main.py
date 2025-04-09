from sqlalchemy.orm import Session
from datetime import datetime
from database import SessionLocal
from models import NewsArticle
from database import init_db
from fastapi import FastAPI, Depends, Request
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from sentiment_analysis import get_data, data_to_dict, calculate_relevance, vader_sentiment_analysis, vader_sentiment_array
import io
import base64
from bokeh.plotting import figure, show, save, output_file
from bokeh.io import output_file
from bokeh.models import HoverTool
from bokeh.embed import components

app = FastAPI()
templates = Jinja2Templates(directory='templates')

#provides a database session for each request and ensures
#its closed afterwards
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

#initialises the database when the application starts
@app.on_event("startup")
def startup():
    init_db()

#when a HTTP GEt request is made to the endpoint "/articles"
#the decorator tells FASTAPI to execute the code below
@app.get('/articles', response_class=HTMLResponse)
#Fetches all articles from the database and renders
#them as a HTML page
def get_articles_html(request: Request, db: Session = Depends(get_db)):
    articles = db.query(NewsArticle).all()
    
    return templates.TemplateResponse('articles.html', {'request': request, 'articles': articles})


def get_sentiment_data(db: Session):
    articles = db.query(NewsArticle).all()
    
    dates, sentiment_values, relative_sentiment_values = [], [], []
    
    for article in articles:
        dates.append(article.date)
        sentiment_values.append(article.sentiment)
        relative_sentiment_values.append(article.relative_sentiment)
        
    return dates, sentiment_values, relative_sentiment_values


@app.get("/chart", response_class=HTMLResponse)
def chart(request: Request, db: Session = Depends(get_db)):
    # Fetch the data from the database
    dates, sentiment_values, relative_sentiment_values = get_sentiment_data(db)

    # Create the Bokeh plot
    p = figure(
        title="Sentiment and Relative Sentiment Over Time",
        x_axis_label='Date',
        y_axis_label='Sentiment Score',
        x_axis_type='datetime',
        height=400,
        width=800
    )

    # Convert dates to datetime format if they are strings
    dates = [datetime.strptime(date, "%Y-%m-%d") for date in dates]

    # Plot sentiment values
    p.line(dates, sentiment_values, legend_label="Sentiment", line_width=2, color="blue")
    
    # Plot relative sentiment values
    p.line(dates, relative_sentiment_values, legend_label="Relative Sentiment", line_width=2, color="green")

    # Add hover tool to show values when hovering over points
    hover = HoverTool()
    hover.tooltips = [("Date", "@x{%F}"), ("Sentiment", "@y")]
    hover.formatters = {"@x": "datetime"}  # Use datetime formatter for the x-axis
    p.add_tools(hover)

    # Format the axis for dates
    p.xaxis.major_label_orientation = 1  # Rotate the x-axis labels for better visibility

    # Use Bokeh's `components` function to get the HTML and JavaScript for the plot
    script, div = components(p)

    # Return the HTML and JavaScript as part of the response
    return HTMLResponse(content=f"{script}\n{div}")



@app.get("/fetch_and_store_data")
#Fetches, Processes, and stores new data in the database
def fetch_and_store_data(db: Session = Depends(get_db)):
    params = {
    "api_key": "e7cce04dc81f518b1b49a4b778a0c71ca7956e011710ed7ce06155f8765185c0",
    "engine": "google_news",
    "hl": "en",
    "q": "meta"
    }
    
    data = get_data(params)
    data_dict = data_to_dict(data)
    data_dict = calculate_relevance(data_dict)
    data_dict = vader_sentiment_analysis(data_dict)
        
    process_and_store_data(data_dict, db)
    
    return {"message": "Data fetched and stored successfully"}

#processes and stores multiple news articles in the database
def process_and_store_data(results, db: Session):
    for story in results:
        if 'date' in story:
            title = story['title']
            date = story['date'][:10]
            sentiment = story.get('sentiment', {}).get('compound', 0)
            relevance = story.get('relevance', 1)
            relative_sentiment = story['relative_compound']
            
            add_article_to_db(db, title, date, sentiment, relevance, relative_sentiment)

#adds a news article with its detials to the database
def add_article_to_db(db: Session, title: str, date: str, sentiment: float, relevance: float, relative_sentiment: float):
    existing_article = db.query(NewsArticle).filter(NewsArticle.title == title).first()
    if existing_article: return
    article = NewsArticle(
        title=title,
        date = datetime.strptime(date, "%m/%d/%Y").date(),
        sentiment=sentiment,
        relevance=relevance,
        relative_sentiment=relative_sentiment,
    )
    db.add(article)
    db.commit()
    db.refresh(article)
    print(f"article '{title}' added to the db ")

#returns on the root URL
@app.get("/")
def read_root():
    return{"message": "Hello world"}
