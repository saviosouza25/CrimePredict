import sys
import os

# Ensure we can find all installed packages
sys.path.insert(0, '/home/runner/workspace/.pythonlibs/lib/python3.11/site-packages')

import requests
import numpy as np
import streamlit as st
from typing import Optional, Dict, List, Tuple
from datetime import datetime, timedelta
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
    
    def analyze_sentiment_trend(self, pair: str) -> Dict[str, any]:
        """Analyze sentiment trend and predict future market sentiment"""
        try:
            # Fetch historical sentiment data over different time periods
            current_sentiment = self.fetch_news_sentiment(pair, limit=20)
            recent_sentiment = self.fetch_news_sentiment(pair, limit=10)
            
            # Calculate sentiment momentum and trend
            sentiment_momentum = recent_sentiment - current_sentiment
            
            # Predict future sentiment based on current trends
            predicted_sentiment = self._predict_future_sentiment(
                current_sentiment, sentiment_momentum, pair
            )
            
            # Analyze market psychology factors
            market_psychology = self._analyze_market_psychology(
                current_sentiment, sentiment_momentum
            )
            
            # Generate predictive signals
            future_signal = self._generate_future_signal(
                predicted_sentiment, sentiment_momentum, market_psychology
            )
            
            return {
                'current_sentiment': current_sentiment,
                'predicted_sentiment': predicted_sentiment,
                'sentiment_momentum': sentiment_momentum,
                'market_psychology': market_psychology,
                'future_signal': future_signal,
                'trend_direction': self._get_trend_direction(sentiment_momentum),
                'confidence': self._calculate_prediction_confidence(
                    current_sentiment, sentiment_momentum
                ),
                'time_horizon': self._estimate_time_horizon(sentiment_momentum),
                'risk_factors': self._identify_risk_factors(
                    current_sentiment, predicted_sentiment
                )
            }
            
        except Exception as e:
            st.warning(f"Erro na análise de tendência de sentimento: {str(e)}")
            return self._get_neutral_sentiment_analysis()
    
    def _predict_future_sentiment(self, current: float, momentum: float, pair: str) -> float:
        """Predict future sentiment based on current state and momentum"""
        # Base prediction on momentum
        base_prediction = current + (momentum * 2.0)
        
        # Apply market-specific adjustments
        market_factor = self._get_market_factor(pair)
        volatility_adjustment = self._get_volatility_adjustment(momentum)
        
        # Calculate predicted sentiment with bounds
        predicted = base_prediction * market_factor + volatility_adjustment
        
        # Keep within realistic bounds (-1 to 1)
        return max(-1.0, min(1.0, predicted))
    
    def _analyze_market_psychology(self, sentiment: float, momentum: float) -> Dict[str, str]:
        """Analyze market psychology factors"""
        psychology = {}
        
        # Fear vs Greed analysis
        if sentiment < -0.3:
            psychology['fear_greed'] = "Medo Extremo" if sentiment < -0.6 else "Medo Moderado"
        elif sentiment > 0.3:
            psychology['fear_greed'] = "Ganância Extrema" if sentiment > 0.6 else "Ganância Moderada"
        else:
            psychology['fear_greed'] = "Equilíbrio"
        
        # Momentum psychology
        if abs(momentum) > 0.2:
            psychology['momentum_state'] = "Impulso Forte"
        elif abs(momentum) > 0.1:
            psychology['momentum_state'] = "Impulso Moderado"
        else:
            psychology['momentum_state'] = "Lateral"
        
        # Market phase
        if sentiment > 0.2 and momentum > 0.1:
            psychology['market_phase'] = "Otimismo Crescente"
        elif sentiment < -0.2 and momentum < -0.1:
            psychology['market_phase'] = "Pessimismo Crescente"
        elif sentiment > 0.2 and momentum < -0.1:
            psychology['market_phase'] = "Correção do Otimismo"
        elif sentiment < -0.2 and momentum > 0.1:
            psychology['market_phase'] = "Recuperação do Pessimismo"
        else:
            psychology['market_phase'] = "Consolidação"
        
        return psychology
    
    def _generate_future_signal(self, predicted: float, momentum: float, psychology: Dict) -> Dict[str, str]:
        """Generate future trading signal based on predicted sentiment"""
        signal = {}
        
        # Primary signal based on predicted sentiment
        if predicted > 0.2:
            signal['direction'] = "COMPRA"
            signal['strength'] = "Forte" if predicted > 0.5 else "Moderada"
        elif predicted < -0.2:
            signal['direction'] = "VENDA" 
            signal['strength'] = "Forte" if predicted < -0.5 else "Moderada"
        else:
            signal['direction'] = "NEUTRO"
            signal['strength'] = "Fraca"
        
        # Timing signal based on momentum
        if abs(momentum) > 0.15:
            signal['timing'] = "Imediato"
        elif abs(momentum) > 0.05:
            signal['timing'] = "Curto Prazo"
        else:
            signal['timing'] = "Médio Prazo"
        
        # Risk level based on psychology
        if psychology.get('fear_greed') in ['Medo Extremo', 'Ganância Extrema']:
            signal['risk_level'] = "Alto"
        elif psychology.get('momentum_state') == "Impulso Forte":
            signal['risk_level'] = "Moderado-Alto"
        else:
            signal['risk_level'] = "Moderado"
        
        return signal
    
    def _get_market_factor(self, pair: str) -> float:
        """Get market-specific sentiment factor"""
        # Major pairs tend to be more stable
        major_pairs = ['EUR/USD', 'GBP/USD', 'USD/JPY', 'USD/CHF']
        if pair in major_pairs:
            return 0.8  # More conservative predictions
        else:
            return 1.2  # More volatile predictions for exotic pairs
    
    def _get_volatility_adjustment(self, momentum: float) -> float:
        """Apply volatility-based adjustment"""
        return momentum * 0.3  # Dampen extreme momentum
    
    def _get_trend_direction(self, momentum: float) -> str:
        """Get trend direction from momentum"""
        if momentum > 0.1:
            return "Crescente"
        elif momentum < -0.1:
            return "Decrescente"
        else:
            return "Lateral"
    
    def _calculate_prediction_confidence(self, current: float, momentum: float) -> float:
        """Calculate confidence in the prediction"""
        # Higher confidence for stronger sentiment and momentum
        sentiment_confidence = min(abs(current) * 2, 1.0)
        momentum_confidence = min(abs(momentum) * 3, 1.0)
        
        # Combined confidence (0 to 1)
        return (sentiment_confidence + momentum_confidence) / 2
    
    def _estimate_time_horizon(self, momentum: float) -> str:
        """Estimate time horizon for the prediction"""
        if abs(momentum) > 0.2:
            return "1-3 dias"
        elif abs(momentum) > 0.1:
            return "3-7 dias"
        else:
            return "1-2 semanas"
    
    def _identify_risk_factors(self, current: float, predicted: float) -> List[str]:
        """Identify potential risk factors"""
        risks = []
        
        if abs(predicted - current) > 0.4:
            risks.append("Mudança drástica prevista")
        
        if abs(predicted) > 0.7:
            risks.append("Sentimento extremo previsto")
        
        if current * predicted < 0:
            risks.append("Reversão de sentimento prevista")
        
        if not risks:
            risks.append("Risco baixo - previsão estável")
        
        return risks
    
    def _get_neutral_sentiment_analysis(self) -> Dict[str, any]:
        """Return neutral sentiment analysis on error"""
        return {
            'current_sentiment': 0.0,
            'predicted_sentiment': 0.0,
            'sentiment_momentum': 0.0,
            'market_psychology': {
                'fear_greed': 'Equilíbrio',
                'momentum_state': 'Lateral',
                'market_phase': 'Consolidação'
            },
            'future_signal': {
                'direction': 'NEUTRO',
                'strength': 'Fraca',
                'timing': 'Médio Prazo',
                'risk_level': 'Moderado'
            },
            'trend_direction': 'Lateral',
            'confidence': 0.0,
            'time_horizon': '1-2 semanas',
            'risk_factors': ['Dados insuficientes']
        }
