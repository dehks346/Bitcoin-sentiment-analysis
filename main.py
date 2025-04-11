from fastapi import FastAPI, Request, Header
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from sqlalchemy import func
from datetime import datetime, timedelta
from database import SessionLocal
from models import NewsArticle, TrainingData
from sentiment_analysis import get_data, data_to_dict, sentiment_analysis
import plotly.express as px
import pandas as pd
import yfinance as yf


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

def get_unproccessed_data():
    params = {"api_key": "e7cce04dc81f518b1b49a4b778a0c71ca7956e011710ed7ce06155f8765185c0", "engine": "google_news", "hl": "en", "q": "bitcoin"}
    data = get_data(params)
    data = data_to_dict(data)
    return data


@app.get('/fetch-and-store-training-data')
async def fetch_and_store_training_data():
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
                next_day_prediction=None
            )
            db.add(record)
        else:
            record.vader_score = average_vader
            record.textblob_score = average_textblob
            record.combined_sentiment = average_combined_sentiment
            record.btc_price = today_close
            record.btc_volume = btc_data['Volume'].iloc[-1]
        db.commit()
        db.refresh(record)
        return {"message": record}



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

@app.get('/dashboard', response_class=HTMLResponse)
async def combined_graph(request: Request, cache_control: str = Header(default="max-age=3600")):
    with SessionLocal() as db:
        articles = db.query(NewsArticle).order_by(NewsArticle.date.asc()).all()
        training = db.query(TrainingData).order_by(TrainingData.date.asc()).all()
        min_date = db.query(func.min(NewsArticle.date)).scalar()
        max_date = db.query(func.max(NewsArticle.date)).scalar()
    
    if not articles: return templates.TemplateResponse("graph.html", {"request": request, "sentiment_graph": "<p>No sentiment data available</p>",})
    
    start_date = min_date.strftime('%Y-%m-%d')
    end_date = (max_date + timedelta(days=1)).strftime('%Y-%m-%d')  
    btc_data = yf.Ticker('BTC-GBP').history(start=start_date, end=end_date, interval='1h')        
    df_btc = pd.DataFrame({'date': btc_data.index, 'btc_price': btc_data['Close']})
    
    btc_graph = create_btc_graph(df_btc)
    sentiment_graph = create_sentiment_graph(articles)
    
    response = templates.TemplateResponse('index.html',{'request': request,'sentiment_graph': sentiment_graph,'btc_graph': btc_graph,'training_data': training,'articles': articles})
    response.headers["Cache-Control"] = cache_control
    return response

def create_sentiment_graph(articles):
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
    window_size = 3
    df['smoothed_avg'] = df['combined_sentiment'].rolling(
        window=window_size,
        min_periods=1,
        center=True
    ).mean()
    fig = px.line(
        df,
        x='date',
        y=['vader_compound', 'textblob_polarity', 'combined_sentiment'],
        labels={'date': 'Date', 'value': 'Sentiment Score'},
        title='<b>Bitcoin News Sentiment</b>',
    )
    line_styles = {
        'vader_compound': {'color': '#4285F4', 'width': 2.2, 'dash': 'solid'},
        'textblob_polarity': {'color': '#EA4335', 'width': 2.2, 'dash': 'solid'},
        'combined_sentiment': {'color': '#34A853', 'width': 3, 'dash': 'solid'}
    }
    for trace in fig.data:
        if trace.name in line_styles:
            trace.update(line=line_styles[trace.name])
    fig.add_scatter(
        x=df['date'],
        y=df['smoothed_avg'],
        mode='lines',
        name=f'Trend Line ({window_size}-day)',
        line=dict(color='#9D40C5', width=3.5, dash='dot')
    )
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
    fig.add_hline(
        y=0,
        line_dash='dash',
        line_color='#5F6368',
        line_width=2,
        annotation_text="Neutral Baseline",
        annotation_position="bottom right",
        annotation_font=dict(size=12)
    )
    df['change'] = df['combined_sentiment'].diff()
    change_threshold = 0.25
    significant_changes = df[abs(df['change']) > change_threshold].nlargest(3, 'change')
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

    fig.update_traces(hovertemplate="<b>%{x|%b %d, %Y}</b><br>Sentiment: %{y:.3f}<extra>%{fullData.name}</extra>")
    return fig.to_html(full_html=False)
def create_btc_graph(df_btc):
    fig = px.line(
        df_btc,
        x='date',
        y='btc_price',
        labels={'date': 'Date', 'btc_price': 'Bitcoin Price (GBP)'},
        title='<b>Bitcoin Price History</b>'
    )

    fig.update_layout(
        plot_bgcolor='white',
        paper_bgcolor='white',
        font=dict(family="Arial", size=13),
        title={'text': "<b>Bitcoin Price History</b>", 'y': 0.96, 'x': 0.5, 'xanchor': 'center', 'yanchor': 'top'},
        xaxis=dict(title="<b>Date</b>", gridcolor='rgba(0,0,0,0.05)', showgrid=True),
        yaxis=dict(title="<b>Price (GBP)</b>", gridcolor='rgba(0,0,0,0.05)', tickformat=",.0f"),
        hovermode="x unified",
    )
    
    fig.update_layout(
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
    )

    fig.update_traces(
        line=dict(color='#4285F4', width=2.5),
        hovertemplate="<b>%{x|%b %d, %Y}</b><br>Price: £%{y:,.2f}<extra></extra>"
    )

    return fig.to_html(full_html=False)