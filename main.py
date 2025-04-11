from fastapi import FastAPI, Depends, Request
from database import SessionLocal, engine
from models import NewsArticle, Base
from sentiment_analysis import get_data, data_to_dict, sentiment_analysis
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
import plotly.express as px
from jinja2 import Template
from sqlalchemy import func
from datetime import datetime
import json
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
from fastapi import Header
from requests import Session
import pprint


# Shared configuration for both charts
CHART_CONFIG = {
    "width": 1100,
    "height": 650,
    "font_family": "Arial",
    "title_font_size": 20,
    "axis_title_font_size": 14,
    "axis_tick_font_size": 12,
    "legend_font_size": 12,
    "hover_font_size": 12,
    "colors": {
        "vader": "#4285F4",
        "textblob": "#EA4335",
        "combined": "#34A853",
        "trend_line": "#9D40C5",
        "neutral_line": "#5F6368",
        "background": "white",
        "grid": "rgba(0,0,0,0.05)"
    },
    "range_selector": {
        "buttons": [
            {"count": 7, "label": "1w", "step": "day", "stepmode": "backward"},
            {"count": 1, "label": "1m", "step": "month", "stepmode": "backward"},
            {"count": 3, "label": "3m", "step": "month", "stepmode": "backward"},
            {"step": "all", "label": "Full Range"}
        ],
        "font_size": 11,
        "bgcolor": "rgba(255,255,255,0.8)",
        "bordercolor": "rgba(0,0,0,0.1)"
    }
}





app = FastAPI()

templates = Jinja2Templates(directory='templates')

#Base.metadata.drop_all(bind=engine)
#Base.metadata.create_all(bind=engine)
#print("Database reset successfully!")

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

def get_btc_data():
    url = 'https://pro-api.coinmarketcap.com/v2/cryptocurrency/quotes/latest'
    api = 'bf36015e-b032-48c9-b401-c03d5235e3c3'
    
    parameters = {
        'slug':'bitcoin',
        'convert':'GBP'
    }
    
    headers = {
        'Accepts': 'application/json',
        'X-CMC_PRO_API_KEY': api
    }
    
    session = Session()
    session.headers.update(headers)
    
    response = session.get(url, params=parameters)
    
    info = json.loads(response.text)['data']['1']['quote']['GBP']['price']
    
    return info

@app.get('/fetch-and-store-training-data')
async def fetch_and_store_training_data():
    with SessionLocal() as db:
        today = datetime.now().date()
        articles = db.query(NewsArticle).filter(
            func.date(NewsArticle.date) == today
        ).all()
        
        if not articles:
            return {"messages": 'no articles found for today'}
        
        average_vader = sum(a.vader_compound for a in articles) / len(articles)
        average_textblob = sum(a.textblob_polarity for a in articles) / len(articles)
        average_combined_sentiment = sum(a.combined_sentiment for a in articles) / len(articles)
    
    


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
    return {"message": f"Stored {stored_count} new articles out of {len(titles)} fetched"}


@app.get('/display-data', response_class=HTMLResponse)
async def display_data(request: Request):
    
    with SessionLocal() as db:
        articles = db.query(NewsArticle).all()
        
        
    return templates.TemplateResponse('data.html', {'request': request, 'articles': articles})


@app.get('/graph', response_class=HTMLResponse)
async def combined_graph(
    request: Request,
    cache_control: str = Header(default="max-age=3600")  # 1 hour cache
):
    with SessionLocal() as db:
        articles = db.query(NewsArticle).order_by(NewsArticle.date.asc()).all()
    
    if not articles:
        return templates.TemplateResponse("graph.html", {
            "request": request, 
            "sentiment_graph": "<p>No sentiment data available</p>",
            "bitcoin_graph": ""
        })

    
    
    sentiment_graph = create_sentiment_graph(articles)
        
    response = templates.TemplateResponse(
        'graph.html',
        {
            'request': request,
            'sentiment_graph': sentiment_graph,
        }
    )
    response.headers["Cache-Control"] = cache_control
    return response

def create_sentiment_graph(articles):
    # Data preparation
    dates = [article.date for article in articles]
    vader_compound = [article.vader_compound for article in articles]
    textblob_polarity = [article.textblob_polarity for article in articles]
    combined_sentiment = [article.combined_sentiment for article in articles]
    
    df = pd.DataFrame({
        'date': pd.to_datetime(dates),
        'vader_compound': vader_compound,
        'textblob_polarity': textblob_polarity,
        'combined_sentiment': combined_sentiment
    }).sort_values(by='date')
    
    # Calculate rolling average
    window_size = 3
    df['smoothed_avg'] = df['combined_sentiment'].rolling(
        window=window_size,
        min_periods=1,
        center=True
    ).mean()
    
    # Create figure
    fig = px.line(
        df,
        x='date',
        y=['vader_compound', 'textblob_polarity', 'combined_sentiment'],
        labels={'date': 'Date', 'value': 'Sentiment Score'},
        title='<b>News Sentiment Analysis</b>',
        width=CHART_CONFIG["width"],
        height=CHART_CONFIG["height"]
    )
    
    # Line styling
    line_styles = {
        'vader_compound': {'color': '#4285F4', 'width': 2.2, 'dash': 'solid'},
        'textblob_polarity': {'color': '#EA4335', 'width': 2.2, 'dash': 'solid'},
        'combined_sentiment': {'color': '#34A853', 'width': 3, 'dash': 'solid'}
    }
    
    for trace in fig.data:
        if trace.name in line_styles:
            trace.update(line=line_styles[trace.name])
    
    # Add smoothed average
    fig.add_scatter(
        x=df['date'],
        y=df['smoothed_avg'],
        mode='lines',
        name=f'Trend Line ({window_size}-day)',
        line=dict(color='#9D40C5', width=3.5, dash='dot')
    )
    
    # Sentiment regions
    sentiment_regions = [
        {'range': (-1, -0.5), 'color': 'rgba(234, 67, 53, 0.15)', 'label': 'Negative'},
        {'range': (-0.5, -0.1), 'color': 'rgba(234, 67, 53, 0.08)', 'label': 'Slightly Negative'},
        {'range': (-0.1, 0.1), 'color': 'rgba(189, 189, 189, 0.1)', 'label': 'Neutral'},
        {'range': (0.1, 0.5), 'color': 'rgba(52, 168, 83, 0.08)', 'label': 'Slightly Positive'},
        {'range': (0.5, 1), 'color': 'rgba(52, 168, 83, 0.15)', 'label': 'Positive'}
    ]
    
    for region in sentiment_regions:
        fig.add_shape(
            type="rect",
            x0=df['date'].min(),
            x1=df['date'].max(),
            y0=region['range'][0],
            y1=region['range'][1],
            fillcolor=region['color'],
            layer="below",
            line_width=0
        )
    
    # Zero line
    fig.add_hline(
        y=0,
        line_dash='dash',
        line_color='#5F6368',
        line_width=2,
        annotation_text="Neutral Baseline",
        annotation_position="bottom right",
        annotation_font=dict(size=12)
    )
    
    # Detect significant changes
    df['change'] = df['combined_sentiment'].diff()
    change_threshold = 0.25
    significant_changes = df[abs(df['change']) > change_threshold].nlargest(3, 'change')
    
    # Annotate changes
    for _, row in significant_changes.iterrows():
        direction = "↑" if row['change'] > 0 else "↓"
        fig.add_annotation(
            x=row['date'],
            y=row['combined_sentiment'],
            text=f"{direction} {abs(row['change']):.2f}",
            showarrow=True,
            arrowhead=2,
            ax=0,
            ay=-40 if row['change'] > 0 else 40,
            bgcolor="white",
            bordercolor="#5F6368",
            borderwidth=1,
            font=dict(size=12, color='#5F6368')
        )
        
        
    
    # Layout adjustments
    fig.update_layout(
        autosize=True,
        plot_bgcolor='white',
        paper_bgcolor='white',
        font=dict(family="Arial", size=13, color="#202124"),
        title={
            'text': "<b>News Sentiment Analysis</b>",
            'y':0.96,
            'x':0.5,
            'xanchor': 'center',
            'yanchor': 'top',
            'font': dict(size=20, color='#202124')
        },
        xaxis=dict(
            title="<b>Date</b>",
            title_font=dict(size=14),
            gridcolor='rgba(0,0,0,0.05)',
            showgrid=True,
            rangeslider=dict(visible=True, thickness=0.08),
            rangeselector=dict(
                buttons=list([
                    dict(count=7, label="1w", step="day", stepmode="backward"),
                    dict(count=1, label="1m", step="month", stepmode="backward"),
                    dict(count=3, label="3m", step="month", stepmode="backward"),
                    dict(step="all", label="Full Range")
                ]),
                font=dict(size=11),
                bgcolor='rgba(255,255,255,0.8)',
                bordercolor='rgba(0,0,0,0.1)',
                borderwidth=1
            )
        ),
        yaxis=dict(
            title="<b>Sentiment Score</b>",
            title_font=dict(size=14),
            gridcolor='rgba(0,0,0,0.05)',
            zerolinecolor='#5F6368',
            zerolinewidth=1.5,
            range=[-1.05, 1.05],
            tickvals=[-1, -0.75, -0.5, -0.25, 0, 0.25, 0.5, 0.75, 1],
            tickfont=dict(size=12)
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5,
            bgcolor='rgba(255,255,255,0.8)',
            bordercolor='rgba(0,0,0,0.1)',
            borderwidth=1,
            font=dict(size=12),
            itemwidth=30
        ),
        margin=dict(l=50, r=50, t=90, b=60),
        hovermode="x unified",
        hoverlabel=dict(
            bgcolor="white",
            font_size=12,
            font_family="Arial",
            bordercolor="#5F6368"
        )
    )
    
    fig.update_traces(
        hovertemplate="<b>%{x|%b %d, %Y}</b><br>Sentiment: %{y:.3f}<extra>%{fullData.name}</extra>"
    )
    
    return fig.to_html(full_html=False)


