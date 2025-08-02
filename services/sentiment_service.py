import requests
import numpy as np
import streamlit as st
from typing import Optional
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from config.settings import API_KEY, NEWS_CACHE_TTL
from utils.cache_manager import CacheManager

class SentimentService:
    """Enhanced sentiment analysis service"""
    
    def __init__(self):
        self.analyzer = SentimentIntensityAnalyzer()
    
    def fetch_news_sentiment(self, pair: str, limit: int = 50) -> float:
        """Fetch and analyze news sentiment with caching"""
        cache_key = CacheManager.get_cache_key('news_sentiment', pair, limit)
        cached_data = CacheManager.get_cached_data(cache_key, NEWS_CACHE_TTL)
        
        if cached_data is not None:
            return cached_data
        
        try:
            _, to_symbol = pair.split('/')
            
            # Fetch news data
            params = {
                'function': 'NEWS_SENTIMENT',
                'tickers': f'FOREX:{to_symbol}',
                'topics': 'financial_markets,earnings,ipo,mergers_and_acquisitions,financial_markets',
                'apikey': API_KEY,
                'limit': limit,
                'sort': 'LATEST'
            }
            
            url = 'https://www.alphavantage.co/query'
            response = requests.get(url, params=params, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            
            if 'Error Message' in data:
                raise ValueError(f"API Error: {data['Error Message']}")
            
            feed = data.get('feed', [])
            
            if not feed:
                sentiment_score = 0.0
            else:
                sentiments = []
                for item in feed:
                    # Combine title and summary for analysis
                    text = f"{item.get('title', '')} {item.get('summary', '')}"
                    if text.strip():
                        sentiment = self.analyzer.polarity_scores(text)
                        # Weight by relevance score if available
                        relevance = float(item.get('relevance_score', 1.0))
                        weighted_sentiment = sentiment['compound'] * relevance
                        sentiments.append(weighted_sentiment)
                
                sentiment_score = float(np.mean(sentiments)) if sentiments else 0.0
            
            # Cache the result
            CacheManager.set_cached_data(cache_key, sentiment_score)
            return sentiment_score
            
        except Exception as e:
            st.warning(f"Could not fetch sentiment data: {str(e)}")
            return 0.0  # Return neutral sentiment on error
    
    def get_sentiment_signal(self, sentiment_score: float) -> str:
        """Convert sentiment score to signal"""
        if sentiment_score > 0.1:
            return "Bullish"
        elif sentiment_score < -0.1:
            return "Bearish"
        else:
            return "Neutral"
    
    def get_sentiment_strength(self, sentiment_score: float) -> str:
        """Get sentiment strength description"""
        abs_score = abs(sentiment_score)
        if abs_score > 0.5:
            return "Very Strong"
        elif abs_score > 0.3:
            return "Strong"
        elif abs_score > 0.1:
            return "Moderate"
        else:
            return "Weak"
