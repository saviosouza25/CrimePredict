import pandas as pd
import numpy as np
from typing import Dict, List, Optional
import requests
import streamlit as st
from datetime import datetime, timedelta
import re

# Safe imports with fallbacks
try:
    from vadersentiment import SentimentIntensityAnalyzer
    VADER_AVAILABLE = True
except ImportError:
    VADER_AVAILABLE = False
    class SentimentIntensityAnalyzer:
        def polarity_scores(self, text):
            return {'compound': 0, 'pos': 0, 'neu': 1, 'neg': 0}

try:
    from textblob import TextBlob
    TEXTBLOB_AVAILABLE = True
except ImportError:
    TEXTBLOB_AVAILABLE = False
    class TextBlob:
        def __init__(self, text):
            self.sentiment = type('obj', (object,), {'polarity': 0, 'subjectivity': 0.5})()

class AdvancedSentimentService:
    """
    Análise de sentimento avançada integrando:
    - Dados de notícias via API
    - Análise de redes sociais (simulado)
    - Processamento NLP com VADER e TextBlob
    - Pontuação de sentimento (positivo/negativo/neutro)
    """
    
    def __init__(self):
        self.vader_analyzer = SentimentIntensityAnalyzer()
        
        # Palavras-chave específicas para forex
        self.forex_keywords = {
            'positive': [
                'bull', 'bullish', 'rally', 'surge', 'rise', 'gain', 'strong', 
                'support', 'breakout', 'momentum', 'uptrend', 'growth', 
                'recovery', 'optimistic', 'confidence', 'strength'
            ],
            'negative': [
                'bear', 'bearish', 'crash', 'fall', 'decline', 'weak', 
                'resistance', 'breakdown', 'selloff', 'downtrend', 'recession',
                'concern', 'fear', 'uncertainty', 'volatility', 'risk'
            ]
        }
    
    def analyze_comprehensive_sentiment(self, pair: str, lookback_days: int = 7) -> Dict:
        """
        Análise completa de sentimento para um par de moedas
        
        Args:
            pair: Par de moedas (ex: EUR/USD)
            lookback_days: Dias para análise retrospectiva
            
        Returns:
            Dict com análise completa de sentimento
        """
        try:
            # 1. Análise de notícias
            news_sentiment = self._analyze_news_sentiment(pair, lookback_days)
            
            # 2. Análise de redes sociais (simulado)
            social_sentiment = self._analyze_social_sentiment(pair, lookback_days)
            
            # 3. Análise de eventos econômicos
            economic_sentiment = self._analyze_economic_events(pair, lookback_days)
            
            # 4. Score composto
            composite_score = self._calculate_composite_sentiment_score(
                news_sentiment, social_sentiment, economic_sentiment
            )
            
            # 5. Classificação e sinais
            sentiment_classification = self._classify_sentiment(composite_score)
            
            # 6. Impacto no trading
            trading_impact = self._assess_trading_impact(
                sentiment_classification, pair, composite_score
            )
            
            return {
                'pair': pair,
                'analysis_period': f'{lookback_days} dias',
                'news_sentiment': news_sentiment,
                'social_sentiment': social_sentiment,
                'economic_sentiment': economic_sentiment,
                'composite_score': composite_score,
                'sentiment_classification': sentiment_classification,
                'trading_impact': trading_impact,
                'sentiment_trend': self._calculate_sentiment_trend(
                    news_sentiment, social_sentiment
                ),
                'confidence_level': self._calculate_confidence_level(
                    news_sentiment, social_sentiment, economic_sentiment
                )
            }
            
        except Exception as e:
            st.warning(f"Erro na análise de sentimento para {pair}: {str(e)}")
            return self._get_default_sentiment_analysis(pair)
    
    def _analyze_news_sentiment(self, pair: str, lookback_days: int) -> Dict:
        """Análise de sentimento de notícias financeiras"""
        try:
            # Buscar notícias (simulado - em produção seria via NewsAPI ou similar)
            news_data = self._fetch_financial_news(pair, lookback_days)
            
            if not news_data:
                return self._get_pattern_news_sentiment(pair)
            
            sentiments = []
            article_count = 0
            
            for article in news_data:
                # Análise com VADER
                vader_score = self.vader_analyzer.polarity_scores(article['text'])
                
                # Análise com TextBlob
                blob = TextBlob(article['text'])
                textblob_polarity = blob.sentiment.polarity
                textblob_subjectivity = blob.sentiment.subjectivity
                
                # Ajuste baseado em palavras-chave forex
                forex_adjustment = self._calculate_forex_keyword_adjustment(article['text'])
                
                # Score final combinado
                combined_score = (
                    vader_score['compound'] * 0.4 +
                    textblob_polarity * 0.4 +
                    forex_adjustment * 0.2
                )
                
                sentiments.append({
                    'article_id': article_count,
                    'title': article.get('title', ''),
                    'vader_score': vader_score['compound'],
                    'textblob_polarity': textblob_polarity,
                    'textblob_subjectivity': textblob_subjectivity,
                    'forex_adjustment': forex_adjustment,
                    'combined_score': combined_score,
                    'publish_date': article.get('date', datetime.now())
                })
                article_count += 1
            
            # Métricas agregadas
            scores = [s['combined_score'] for s in sentiments]
            avg_sentiment = np.mean(scores) if scores else 0
            sentiment_volatility = np.std(scores) if len(scores) > 1 else 0
            
            # Distribuição de sentimento
            positive_articles = len([s for s in scores if s > 0.1])
            negative_articles = len([s for s in scores if s < -0.1])
            neutral_articles = len(scores) - positive_articles - negative_articles
            
            return {
                'avg_sentiment': round(avg_sentiment, 3),
                'sentiment_volatility': round(sentiment_volatility, 3),
                'total_articles': len(sentiments),
                'positive_articles': positive_articles,
                'negative_articles': negative_articles,
                'neutral_articles': neutral_articles,
                'sentiment_distribution': {
                    'positive_pct': round(positive_articles / max(len(scores), 1) * 100, 1),
                    'negative_pct': round(negative_articles / max(len(scores), 1) * 100, 1),
                    'neutral_pct': round(neutral_articles / max(len(scores), 1) * 100, 1)
                },
                'detailed_sentiments': sentiments[:10],  # Top 10 para exibição
                'news_momentum': self._calculate_news_momentum(sentiments)
            }
            
        except Exception:
            return self._get_pattern_news_sentiment(pair)
    
    def _analyze_social_sentiment(self, pair: str, lookback_days: int) -> Dict:
        """Análise de sentimento de redes sociais (simulado)"""
        try:
            # Em produção, seria integração com Twitter API, Reddit, etc.
            # Por enquanto, simular baseado em padrões realistas
            
            base_currency, quote_currency = pair.split('/')
            
            # Simular dados de redes sociais baseado no par
            social_patterns = self._get_social_media_patterns(pair)
            
            # Simular posts/tweets
            total_mentions = social_patterns['mentions_per_day'] * lookback_days
            sentiment_bias = social_patterns['sentiment_bias']
            
            # Distribuição de sentimento simulada
            positive_mentions = int(total_mentions * (0.4 + sentiment_bias))
            negative_mentions = int(total_mentions * (0.35 - sentiment_bias))
            neutral_mentions = total_mentions - positive_mentions - negative_mentions
            
            # Score médio de sentimento
            avg_social_sentiment = sentiment_bias + np.random.normal(0, 0.1)
            avg_social_sentiment = max(-1, min(1, avg_social_sentiment))
            
            # Engagement e influência
            engagement_rate = social_patterns['engagement_rate']
            influencer_sentiment = social_patterns['influencer_bias']
            
            # Volume de menções (tendência)
            mention_trend = 'CRESCENTE' if sentiment_bias > 0.1 else 'DECRESCENTE' if sentiment_bias < -0.1 else 'ESTÁVEL'
            
            return {
                'avg_sentiment': round(avg_social_sentiment, 3),
                'total_mentions': total_mentions,
                'positive_mentions': positive_mentions,
                'negative_mentions': negative_mentions,
                'neutral_mentions': neutral_mentions,
                'engagement_rate': round(engagement_rate, 2),
                'influencer_sentiment': round(influencer_sentiment, 3),
                'mention_trend': mention_trend,
                'sentiment_distribution': {
                    'positive_pct': round(positive_mentions / max(total_mentions, 1) * 100, 1),
                    'negative_pct': round(negative_mentions / max(total_mentions, 1) * 100, 1),
                    'neutral_pct': round(neutral_mentions / max(total_mentions, 1) * 100, 1)
                },
                'social_momentum': self._calculate_social_momentum(
                    avg_social_sentiment, engagement_rate, mention_trend
                ),
                'key_hashtags': social_patterns.get('trending_hashtags', [])
            }
            
        except Exception:
            return {
                'avg_sentiment': 0.0,
                'total_mentions': 100,
                'sentiment_distribution': {'positive_pct': 33, 'negative_pct': 33, 'neutral_pct': 34},
                'social_momentum': 'NEUTRO'
            }
    
    def _analyze_economic_events(self, pair: str, lookback_days: int) -> Dict:
        """Análise de eventos econômicos e impacto no sentimento"""
        try:
            base_currency, quote_currency = pair.split('/')
            
            # Simular eventos econômicos baseado nas moedas
            economic_events = self._get_economic_events_calendar(base_currency, quote_currency, lookback_days)
            
            # Analisar impacto dos eventos
            total_impact = 0
            high_impact_events = 0
            event_details = []
            
            for event in economic_events:
                impact_score = event['impact_score']
                total_impact += impact_score
                
                if abs(impact_score) >= 0.3:
                    high_impact_events += 1
                
                event_details.append({
                    'event_name': event['name'],
                    'currency': event['currency'],
                    'impact_score': impact_score,
                    'impact_level': event['impact_level'],
                    'actual_vs_forecast': event.get('actual_vs_forecast', 'N/A'),
                    'date': event['date']
                })
            
            # Score médio de impacto econômico
            avg_economic_impact = total_impact / max(len(economic_events), 1)
            
            # Classificação do ambiente econômico
            if avg_economic_impact >= 0.2:
                economic_environment = 'POSITIVO'
            elif avg_economic_impact <= -0.2:
                economic_environment = 'NEGATIVO'
            else:
                economic_environment = 'NEUTRO'
            
            return {
                'avg_economic_impact': round(avg_economic_impact, 3),
                'total_events': len(economic_events),
                'high_impact_events': high_impact_events,
                'economic_environment': economic_environment,
                'event_details': event_details[:5],  # Top 5 eventos
                'economic_momentum': self._calculate_economic_momentum(economic_events),
                'currency_strength': {
                    base_currency: self._calculate_currency_strength(base_currency, economic_events),
                    quote_currency: self._calculate_currency_strength(quote_currency, economic_events)
                }
            }
            
        except Exception:
            return {
                'avg_economic_impact': 0.0,
                'total_events': 5,
                'economic_environment': 'NEUTRO',
                'economic_momentum': 'ESTÁVEL'
            }
    
    def _calculate_composite_sentiment_score(self, news_sentiment: Dict, 
                                           social_sentiment: Dict, 
                                           economic_sentiment: Dict) -> Dict:
        """Calcula score composto de sentimento"""
        
        # Pesos para cada componente
        NEWS_WEIGHT = 0.4      # 40% - Notícias são muito importantes
        SOCIAL_WEIGHT = 0.3    # 30% - Redes sociais influenciam mercado
        ECONOMIC_WEIGHT = 0.3  # 30% - Eventos econômicos são fundamentais
        
        # Extrair scores individuais
        news_score = news_sentiment.get('avg_sentiment', 0)
        social_score = social_sentiment.get('avg_sentiment', 0)
        economic_score = economic_sentiment.get('avg_economic_impact', 0)
        
        # Score composto ponderado
        composite_score = (
            news_score * NEWS_WEIGHT +
            social_score * SOCIAL_WEIGHT +
            economic_score * ECONOMIC_WEIGHT
        )
        
        # Volatilidade composta (incerteza)
        news_volatility = news_sentiment.get('sentiment_volatility', 0)
        social_volatility = social_sentiment.get('engagement_rate', 0.5) - 0.5  # Normalizar
        economic_volatility = abs(economic_score) * 0.5  # Eventos de alto impacto = mais volatilidade
        
        composite_volatility = (
            news_volatility * NEWS_WEIGHT +
            abs(social_volatility) * SOCIAL_WEIGHT +
            economic_volatility * ECONOMIC_WEIGHT
        )
        
        # Força do sinal (consenso entre fontes)
        scores = [news_score, social_score, economic_score]
        signal_alignment = self._calculate_signal_alignment(scores)
        
        return {
            'composite_score': round(composite_score, 3),
            'composite_volatility': round(composite_volatility, 3),
            'signal_alignment': signal_alignment,
            'component_scores': {
                'news': round(news_score, 3),
                'social': round(social_score, 3),
                'economic': round(economic_score, 3)
            },
            'component_weights': {
                'news': NEWS_WEIGHT,
                'social': SOCIAL_WEIGHT,
                'economic': ECONOMIC_WEIGHT
            }
        }
    
    def _classify_sentiment(self, composite_score: Dict) -> Dict:
        """Classifica sentimento em categorias"""
        
        score = composite_score['composite_score']
        volatility = composite_score['composite_volatility']
        alignment = composite_score['signal_alignment']
        
        # Classificação principal
        if score >= 0.3:
            sentiment_class = 'MUITO_POSITIVO'
            color = '#00C851'
        elif score >= 0.1:
            sentiment_class = 'POSITIVO'
            color = '#4CAF50'
        elif score <= -0.3:
            sentiment_class = 'MUITO_NEGATIVO'
            color = '#F44336'
        elif score <= -0.1:
            sentiment_class = 'NEGATIVO'
            color = '#FF5722'
        else:
            sentiment_class = 'NEUTRO'
            color = '#FF9800'
        
        # Nível de certeza
        if volatility <= 0.1 and alignment >= 0.7:
            certainty_level = 'ALTA'
        elif volatility <= 0.2 and alignment >= 0.5:
            certainty_level = 'MEDIA'
        else:
            certainty_level = 'BAIXA'
        
        # Força do sinal
        if abs(score) >= 0.4 and certainty_level == 'ALTA':
            signal_strength = 'MUITO_FORTE'
        elif abs(score) >= 0.2 and certainty_level in ['ALTA', 'MEDIA']:
            signal_strength = 'FORTE'
        elif abs(score) >= 0.1:
            signal_strength = 'MODERADA'
        else:
            signal_strength = 'FRACA'
        
        return {
            'sentiment_class': sentiment_class,
            'certainty_level': certainty_level,
            'signal_strength': signal_strength,
            'color': color,
            'sentiment_score': round(score, 3),
            'volatility_level': 'ALTA' if volatility > 0.2 else 'MEDIA' if volatility > 0.1 else 'BAIXA'
        }
    
    def _assess_trading_impact(self, sentiment_classification: Dict, pair: str, composite_score: Dict) -> Dict:
        """Avalia impacto do sentimento no trading"""
        
        sentiment_class = sentiment_classification['sentiment_class']
        signal_strength = sentiment_classification['signal_strength']
        score = composite_score['composite_score']
        
        # Recomendação baseada no sentimento
        if sentiment_class in ['MUITO_POSITIVO', 'POSITIVO']:
            if signal_strength in ['MUITO_FORTE', 'FORTE']:
                trading_bias = 'COMPRA_FORTE'
                recommended_action = 'COMPRAR'
            else:
                trading_bias = 'COMPRA_LEVE'
                recommended_action = 'CONSIDERAR_COMPRA'
        elif sentiment_class in ['MUITO_NEGATIVO', 'NEGATIVO']:
            if signal_strength in ['MUITO_FORTE', 'FORTE']:
                trading_bias = 'VENDA_FORTE'
                recommended_action = 'VENDER'
            else:
                trading_bias = 'VENDA_LEVE'
                recommended_action = 'CONSIDERAR_VENDA'
        else:
            trading_bias = 'NEUTRO'
            recommended_action = 'AGUARDAR'
        
        # Timing recomendado
        volatility_level = sentiment_classification['volatility_level']
        if volatility_level == 'ALTA':
            timing = 'AGUARDAR_ESTABILIZACAO'
        elif signal_strength == 'MUITO_FORTE':
            timing = 'IMEDIATO'
        elif signal_strength == 'FORTE':
            timing = 'CURTO_PRAZO'
        else:
            timing = 'MEDIO_PRAZO'
        
        # Fatores de risco
        risk_factors = []
        if volatility_level == 'ALTA':
            risk_factors.append('Alta volatilidade de sentimento')
        if sentiment_classification['certainty_level'] == 'BAIXA':
            risk_factors.append('Baixa certeza nas fontes')
        if composite_score['signal_alignment'] < 0.5:
            risk_factors.append('Divergência entre fontes')
        
        return {
            'trading_bias': trading_bias,
            'recommended_action': recommended_action,
            'timing': timing,
            'risk_factors': risk_factors,
            'sentiment_weight_in_decision': self._calculate_sentiment_weight(sentiment_classification),
            'market_mood': self._determine_market_mood(sentiment_class, pair),
            'contrarian_opportunity': self._assess_contrarian_opportunity(score, sentiment_class)
        }
    
    # Métodos auxiliares
    def _fetch_financial_news(self, pair: str, lookback_days: int) -> List[Dict]:
        """Busca notícias financeiras (simulado)"""
        # Em produção, seria integração com NewsAPI, Bloomberg, etc.
        # Retornar None para usar padrões simulados
        return None
    
    def _get_pattern_news_sentiment(self, pair: str) -> Dict:
        """Sentimento de notícias baseado em padrões"""
        
        # Padrões baseados em pares específicos
        sentiment_patterns = {
            'EUR/USD': {'bias': 0.05, 'volatility': 0.15, 'articles': 25},
            'GBP/USD': {'bias': -0.02, 'volatility': 0.25, 'articles': 20},
            'USD/JPY': {'bias': 0.03, 'volatility': 0.12, 'articles': 18},
            'BTC/USD': {'bias': 0.15, 'volatility': 0.35, 'articles': 40},
            'ETH/USD': {'bias': 0.10, 'volatility': 0.30, 'articles': 30}
        }
        
        pattern = sentiment_patterns.get(pair, {'bias': 0.0, 'volatility': 0.2, 'articles': 15})
        
        return {
            'avg_sentiment': pattern['bias'],
            'sentiment_volatility': pattern['volatility'],
            'total_articles': pattern['articles'],
            'positive_articles': int(pattern['articles'] * 0.4),
            'negative_articles': int(pattern['articles'] * 0.35),
            'neutral_articles': int(pattern['articles'] * 0.25),
            'sentiment_distribution': {
                'positive_pct': 40,
                'negative_pct': 35,
                'neutral_pct': 25
            },
            'news_momentum': 'CRESCENTE' if pattern['bias'] > 0 else 'DECRESCENTE' if pattern['bias'] < 0 else 'ESTÁVEL'
        }
    
    def _get_social_media_patterns(self, pair: str) -> Dict:
        """Padrões de redes sociais por par"""
        
        social_patterns = {
            'EUR/USD': {
                'mentions_per_day': 150,
                'sentiment_bias': 0.08,
                'engagement_rate': 0.65,
                'influencer_bias': 0.12,
                'trending_hashtags': ['#EURUSD', '#forex', '#trading']
            },
            'BTC/USD': {
                'mentions_per_day': 500,
                'sentiment_bias': 0.20,
                'engagement_rate': 0.85,
                'influencer_bias': 0.25,
                'trending_hashtags': ['#Bitcoin', '#BTC', '#crypto', '#hodl']
            },
            'GBP/USD': {
                'mentions_per_day': 120,
                'sentiment_bias': -0.05,
                'engagement_rate': 0.55,
                'influencer_bias': -0.08,
                'trending_hashtags': ['#GBPUSD', '#Brexit', '#BOE']
            }
        }
        
        return social_patterns.get(pair, {
            'mentions_per_day': 80,
            'sentiment_bias': 0.0,
            'engagement_rate': 0.6,
            'influencer_bias': 0.0,
            'trending_hashtags': ['#forex']
        })
    
    def _get_economic_events_calendar(self, base_currency: str, quote_currency: str, days: int) -> List[Dict]:
        """Calendário de eventos econômicos simulado"""
        
        events = []
        
        # Eventos base por moeda
        currency_events = {
            'USD': [
                {'name': 'NFP', 'impact_level': 'HIGH', 'impact_score': 0.3},
                {'name': 'CPI', 'impact_level': 'HIGH', 'impact_score': 0.25},
                {'name': 'FOMC', 'impact_level': 'VERY_HIGH', 'impact_score': 0.4}
            ],
            'EUR': [
                {'name': 'ECB Rate Decision', 'impact_level': 'VERY_HIGH', 'impact_score': 0.35},
                {'name': 'PMI', 'impact_level': 'MEDIUM', 'impact_score': 0.15}
            ],
            'GBP': [
                {'name': 'BOE Rate Decision', 'impact_level': 'VERY_HIGH', 'impact_score': 0.3},
                {'name': 'GDP', 'impact_level': 'HIGH', 'impact_score': 0.2}
            ],
            'JPY': [
                {'name': 'BOJ Meeting', 'impact_level': 'HIGH', 'impact_score': 0.25},
                {'name': 'Tankan Survey', 'impact_level': 'MEDIUM', 'impact_score': 0.1}
            ]
        }
        
        # Adicionar eventos para ambas as moedas
        for currency in [base_currency, quote_currency]:
            if currency in currency_events:
                for event_template in currency_events[currency]:
                    event = event_template.copy()
                    event['currency'] = currency
                    event['date'] = datetime.now() - timedelta(days=np.random.randint(0, days))
                    
                    # Ajustar score baseado em resultado simulado
                    if np.random.random() > 0.5:
                        event['impact_score'] *= -1  # Resultado negativo
                    
                    events.append(event)
        
        return events
    
    def _calculate_forex_keyword_adjustment(self, text: str) -> float:
        """Ajuste baseado em palavras-chave específicas do forex"""
        
        text_lower = text.lower()
        positive_count = sum(1 for word in self.forex_keywords['positive'] if word in text_lower)
        negative_count = sum(1 for word in self.forex_keywords['negative'] if word in text_lower)
        
        # Normalizar baseado no comprimento do texto
        text_length = len(text.split())
        if text_length == 0:
            return 0
        
        positive_density = positive_count / text_length
        negative_density = negative_count / text_length
        
        # Score ajustado
        adjustment = (positive_density - negative_density) * 2
        return max(-0.5, min(0.5, adjustment))
    
    def _calculate_sentiment_trend(self, news_sentiment: Dict, social_sentiment: Dict) -> str:
        """Calcula tendência de sentimento"""
        
        news_momentum = news_sentiment.get('news_momentum', 'ESTÁVEL')
        social_momentum = social_sentiment.get('social_momentum', 'NEUTRO')
        
        if news_momentum == 'CRESCENTE' and social_momentum in ['CRESCENTE', 'POSITIVO']:
            return 'MELHORANDO'
        elif news_momentum == 'DECRESCENTE' and social_momentum in ['DECRESCENTE', 'NEGATIVO']:
            return 'PIORANDO'
        elif news_momentum == 'CRESCENTE' or social_momentum in ['CRESCENTE', 'POSITIVO']:
            return 'LEVE_MELHORA'
        elif news_momentum == 'DECRESCENTE' or social_momentum in ['DECRESCENTE', 'NEGATIVO']:
            return 'LEVE_PIORA'
        else:
            return 'ESTÁVEL'
    
    def _calculate_signal_alignment(self, scores: List[float]) -> float:
        """Calcula alinhamento entre sinais"""
        
        if not scores:
            return 0
        
        # Verificar se todos os scores têm o mesmo sinal
        positive_count = sum(1 for s in scores if s > 0.05)
        negative_count = sum(1 for s in scores if s < -0.05)
        neutral_count = len(scores) - positive_count - negative_count
        
        # Alinhamento baseado na concentração
        max_consensus = max(positive_count, negative_count, neutral_count)
        alignment = max_consensus / len(scores)
        
        return round(alignment, 2)
    
    def _calculate_sentiment_weight(self, sentiment_classification: Dict) -> float:
        """Calcula peso do sentimento na decisão de trading"""
        
        signal_strength = sentiment_classification['signal_strength']
        certainty_level = sentiment_classification['certainty_level']
        
        # Matriz de pesos
        weight_matrix = {
            ('MUITO_FORTE', 'ALTA'): 0.8,
            ('MUITO_FORTE', 'MEDIA'): 0.7,
            ('FORTE', 'ALTA'): 0.6,
            ('FORTE', 'MEDIA'): 0.5,
            ('MODERADA', 'ALTA'): 0.4,
            ('MODERADA', 'MEDIA'): 0.3,
            ('FRACA', 'ALTA'): 0.2,
            ('FRACA', 'MEDIA'): 0.1
        }
        
        return weight_matrix.get((signal_strength, certainty_level), 0.1)
    
    def _determine_market_mood(self, sentiment_class: str, pair: str) -> str:
        """Determina humor geral do mercado"""
        
        mood_map = {
            'MUITO_POSITIVO': 'EUFÓRICO',
            'POSITIVO': 'OTIMISTA',
            'NEUTRO': 'CAUTELOSO',
            'NEGATIVO': 'PESSIMISTA',
            'MUITO_NEGATIVO': 'PÂNICO'
        }
        
        return mood_map.get(sentiment_class, 'INDEFINIDO')
    
    def _assess_contrarian_opportunity(self, score: float, sentiment_class: str) -> Dict:
        """Avalia oportunidade contrária"""
        
        # Sinais extremos podem indicar reversão
        if sentiment_class == 'MUITO_POSITIVO' and score > 0.6:
            return {
                'opportunity': True,
                'signal': 'VENDA_CONTRARIA',
                'reason': 'Sentimento extremamente positivo pode indicar topo'
            }
        elif sentiment_class == 'MUITO_NEGATIVO' and score < -0.6:
            return {
                'opportunity': True,
                'signal': 'COMPRA_CONTRARIA',
                'reason': 'Sentimento extremamente negativo pode indicar fundo'
            }
        else:
            return {
                'opportunity': False,
                'signal': 'SEGUIR_TENDENCIA',
                'reason': 'Sentimento não está em níveis extremos'
            }
    
    # Métodos de cálculo auxiliares
    def _calculate_news_momentum(self, sentiments: List[Dict]) -> str:
        """Calcula momentum das notícias"""
        if len(sentiments) < 3:
            return 'INDEFINIDO'
        
        recent_scores = [s['combined_score'] for s in sentiments[-3:]]
        older_scores = [s['combined_score'] for s in sentiments[:-3]] if len(sentiments) > 3 else [0]
        
        recent_avg = np.mean(recent_scores)
        older_avg = np.mean(older_scores)
        
        if recent_avg > older_avg + 0.1:
            return 'CRESCENTE'
        elif recent_avg < older_avg - 0.1:
            return 'DECRESCENTE'
        else:
            return 'ESTÁVEL'
    
    def _calculate_social_momentum(self, sentiment: float, engagement: float, trend: str) -> str:
        """Calcula momentum social"""
        if trend == 'CRESCENTE' and sentiment > 0.1 and engagement > 0.7:
            return 'POSITIVO'
        elif trend == 'DECRESCENTE' and sentiment < -0.1 and engagement > 0.7:
            return 'NEGATIVO'
        else:
            return 'NEUTRO'
    
    def _calculate_economic_momentum(self, events: List[Dict]) -> str:
        """Calcula momentum econômico"""
        if not events:
            return 'ESTÁVEL'
        
        recent_impact = np.mean([e['impact_score'] for e in events])
        
        if recent_impact > 0.15:
            return 'POSITIVO'
        elif recent_impact < -0.15:
            return 'NEGATIVO'
        else:
            return 'ESTÁVEL'
    
    def _calculate_currency_strength(self, currency: str, events: List[Dict]) -> float:
        """Calcula força da moeda baseada em eventos"""
        currency_events = [e for e in events if e['currency'] == currency]
        
        if not currency_events:
            return 0.0
        
        strength = np.mean([e['impact_score'] for e in currency_events])
        return round(strength, 2)
    
    def _calculate_confidence_level(self, news_sentiment: Dict, social_sentiment: Dict, economic_sentiment: Dict) -> Dict:
        """Calcula nível de confiança da análise"""
        
        # Fatores de confiança
        news_articles = news_sentiment.get('total_articles', 0)
        social_mentions = social_sentiment.get('total_mentions', 0)
        economic_events = economic_sentiment.get('total_events', 0)
        
        # Score de volume de dados
        data_volume_score = min(1.0, (news_articles + social_mentions/10 + economic_events*5) / 100)
        
        # Score de volatilidade (menos volatilidade = mais confiança)
        news_volatility = news_sentiment.get('sentiment_volatility', 0.5)
        volatility_score = max(0, 1 - news_volatility * 2)
        
        # Score final de confiança
        confidence_score = (data_volume_score * 0.6 + volatility_score * 0.4)
        
        if confidence_score >= 0.8:
            confidence_level = 'MUITO_ALTA'
        elif confidence_score >= 0.6:
            confidence_level = 'ALTA'
        elif confidence_score >= 0.4:
            confidence_level = 'MEDIA'
        else:
            confidence_level = 'BAIXA'
        
        return {
            'confidence_level': confidence_level,
            'confidence_score': round(confidence_score, 2),
            'data_volume_score': round(data_volume_score, 2),
            'volatility_score': round(volatility_score, 2)
        }
    
    def _get_default_sentiment_analysis(self, pair: str) -> Dict:
        """Análise padrão em caso de erro"""
        return {
            'pair': pair,
            'analysis_period': '7 dias',
            'composite_score': {
                'composite_score': 0.0,
                'composite_volatility': 0.2,
                'signal_alignment': 0.5
            },
            'sentiment_classification': {
                'sentiment_class': 'NEUTRO',
                'certainty_level': 'BAIXA',
                'signal_strength': 'FRACA',
                'sentiment_score': 0.0
            },
            'trading_impact': {
                'trading_bias': 'NEUTRO',
                'recommended_action': 'AGUARDAR',
                'timing': 'MEDIO_PRAZO'
            }
        }