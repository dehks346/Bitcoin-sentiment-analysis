<h1><strong>Bitcoin Sentiment Analysis Tool</strong></h1>

<h3>How to Run</h3>
<ol>
  <li><strong>Install dependencies</strong> (if not already installed):
    <pre><code>pip install -r requirements.txt</code></pre>
  </li>
  <li><strong>Start the server</strong>:
    <pre><code>python3 -m uvicorn main:app --reload --host 0.0.0.0 --port 8000</code></pre>
  </li>
  <li><strong>Open your browser</strong> and navigate to:
    <ul>
      <li>Dashboard: <a href="http://localhost:8000/dashboard">http://localhost:8000/dashboard</a></li>
      <li>API Documentation: <a href="http://localhost:8000/docs">http://localhost:8000/docs</a></li>
    </ul>
  </li>
</ol>
<p><strong>Note:</strong> The <code>--reload</code> flag enables auto-reload on code changes. Remove it for production use.</p>

<p>Developed a Bitcoin sentiment analysis tool that scrapes daily news articles containing Bitcoin-related keywords from Google News. Headlines are stored in a database and processed using natural language processing libraries (VADER and TextBlob) to generate sentiment scores. An average sentiment score is computed for each article and aggregated daily. These sentiment trends are then plotted in real-time alongside live Bitcoin price data, with significant sentiment shifts highlighted on the graph. The system continuously updates and stores both sentiment and market data for historical analysis and correlation insights.</p>


<h3>Automated News Scraping & Sentiment Analysis</h3>
<p>Scrapes Bitcoin-related headlines daily and analyzes sentiment using VADER and TextBlob, storing results for each article in a database.</p>

<h3>Live Sentiment & Price Visualization</h3>
<p>Displays real-time graphs comparing Bitcoin sentiment trends with market price data, with key sentiment shifts clearly marked.</p>

<h3>Persistent Data Storage & Aggregation</h3>
<p>Stores all article data, sentiment scores, and price stats in a structured database, enabling long-term trend analysis and historical insights.</p>
