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
async def graph(request: Request):
    
    with SessionLocal() as db:
        articles = db.query(NewsArticle).all()

    if not articles:
        return templates.TemplateResponse("graph.html", {"request": request, "graph": "<p>No data available</p>"})
    
    dates = [article.date for article in articles]
    vader_compound = [article.vader_compound for article in articles]
    textblob_polarity = [article.textblob_polarity for article in articles]
    combined_sentiment = [article.combined_sentiment for article in articles]
    
    # Create DataFrame
    df = pd.DataFrame({
        'date': pd.to_datetime(dates),
        'vader_compound': vader_compound,
        'textblob_polarity': textblob_polarity,
        'combined_sentiment': combined_sentiment
    }).sort_values(by='date')
    
    # Calculate rolling average before any scaling
    window_size = 3  # Slightly larger window for smoother results
    df['smoothed_avg'] = df['combined_sentiment'].rolling(
        window=window_size,
        min_periods=1,
        center=True
    ).mean()
    
    # Create figure with proper dimensions
    fig = px.line(
        df,
        x='date',
        y=['vader_compound', 'textblob_polarity', 'combined_sentiment'],
        labels={'date': 'Date', 'value': 'Sentiment Score'},
        title='<b>Sentiment Analysis Timeline</b>',
        width=1100,
        height=650
    )
    
    # Enhanced line styling
    line_styles = {
        'vader_compound': {'color': '#4285F4', 'width': 2.2, 'dash': 'solid'},
        'textblob_polarity': {'color': '#EA4335', 'width': 2.2, 'dash': 'solid'},
        'combined_sentiment': {'color': '#34A853', 'width': 3, 'dash': 'solid'}
    }
    
    for trace in fig.data:
        if trace.name in line_styles:
            trace.update(line=line_styles[trace.name])
    
    # Add smoothed average with distinct style
    fig.add_scatter(
        x=df['date'],
        y=df['smoothed_avg'],
        mode='lines',
        name=f'Trend Line ({window_size}-day)',
        line=dict(color='#9D40C5', width=3.5, dash='dot')
    )
    
    # Sentiment regions with natural scale (-1 to 1)
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
    
    # Zero line with better visibility
    fig.add_hline(
        y=0,
        line_dash='dash',
        line_color='#5F6368',
        line_width=2,
        annotation_text="Neutral Baseline",
        annotation_position="bottom right",
        annotation_font=dict(size=12)
    )
    
    # Detect significant changes (using natural scale)
    df['change'] = df['combined_sentiment'].diff()
    change_threshold = 0.25
    significant_changes = df[abs(df['change']) > change_threshold].nlargest(3, 'change')
    
    # Annotate significant changes
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
    
    # Final layout adjustments
    fig.update_layout(
        plot_bgcolor='white',
        paper_bgcolor='white',
        font=dict(family="Arial", size=13, color="#202124"),
        title={
            'text': "<b>Sentiment Analysis Timeline</b>",
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
    
    # Enhanced hover information
    fig.update_traces(
        hovertemplate="<b>%{x|%b %d, %Y}</b><br>Sentiment: %{y:.3f}<extra>%{fullData.name}</extra>"
    )
    
    graph_html = fig.to_html(full_html=False)
    
    btc = yf.Ticker("BTC-USD")
    df = btc.history(period="6mo")  # Get 6 months of data
    
    # Reset index to make Date a column
    df = df.reset_index()
    
    # Create figure with identical styling
    fig = px.line(
        df,
        x='Date',
        y='Close',
        labels={'Date': 'Date', 'Close': 'Price (USD)'},
        title='<b>Bitcoin Price Timeline</b>',
        width=1100,
        height=650
    )
    
    # Apply identical line styling
    fig.update_traces(
        line=dict(color='#4285F4', width=3, dash='solid'),
        name='BTC Price'
    )
    
    # Add moving average (30-day) to match the smoothed average in sentiment chart
    df['30_day_ma'] = df['Close'].rolling(window=30, min_periods=1).mean()
    fig.add_scatter(
        x=df['Date'],
        y=df['30_day_ma'],
        mode='lines',
        name='Trend Line (30-day)',
        line=dict(color='#9D40C5', width=3.5, dash='dot')
    )
    
    # Price regions (matching sentiment style but for price ranges)
    price_min = df['Close'].min()
    price_max = df['Close'].max()
    price_range = price_max - price_min
    price_regions = [
        {'range': (price_min, price_min + 0.2*price_range), 'color': 'rgba(234, 67, 53, 0.1)', 'label': 'Low'},
        {'range': (price_min + 0.2*price_range, price_min + 0.8*price_range), 'color': 'rgba(189, 189, 189, 0.1)', 'label': 'Mid'},
        {'range': (price_min + 0.8*price_range, price_max), 'color': 'rgba(52, 168, 83, 0.1)', 'label': 'High'}
    ]
    
    for region in price_regions:
        fig.add_shape(
            type="rect",
            x0=df['Date'].min(),
            x1=df['Date'].max(),
            y0=region['range'][0],
            y1=region['range'][1],
            fillcolor=region['color'],
            layer="below",
            line_width=0
        )
    
    # Add median line instead of zero line
    median_price = df['Close'].median()
    fig.add_hline(
        y=median_price,
        line_dash='dash',
        line_color='#5F6368',
        line_width=2,
        annotation_text=f"Median: ${median_price:,.2f}",
        annotation_position="bottom right",
        annotation_font=dict(size=12)
    )
    
    # Detect significant price changes (5% daily change)
    df['change'] = df['Close'].pct_change() * 100
    change_threshold = 5  # 5%
    significant_changes = df[abs(df['change']) > change_threshold].nlargest(3, 'change')
    
    # Annotate significant changes
    for _, row in significant_changes.iterrows():
        direction = "↑" if row['change'] > 0 else "↓"
        fig.add_annotation(
            x=row['Date'],
            y=row['Close'],
            text=f"{direction} {abs(row['change']):.1f}%",
            showarrow=True,
            arrowhead=2,
            ax=0,
            ay=-40 if row['change'] > 0 else 40,
            bgcolor="white",
            bordercolor="#5F6368",
            borderwidth=1,
            font=dict(size=12, color='#5F6368')
        )
    
    # Apply identical layout
    fig.update_layout(
        plot_bgcolor='white',
        paper_bgcolor='white',
        font=dict(family="Arial", size=13, color="#202124"),
        title={
            'text': "<b>Bitcoin Price Timeline</b>",
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
            title="<b>Price (USD)</b>",
            title_font=dict(size=14),
            gridcolor='rgba(0,0,0,0.05)',
            zeroline=False,
            tickprefix="$",
            tickformat=",.0f",
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
    
    # Enhanced hover information
    fig.update_traces(
        hovertemplate="<b>%{x|%b %d, %Y}</b><br>Price: $%{y:,.2f}<extra>%{fullData.name}</extra>"
    )
    
    bitcoin_graph = fig.to_html(full_html=False)
    
    return templates.TemplateResponse('graph.html', {'request': request, 'graph': graph_html, 'btc_graph': bitcoin_graph})


