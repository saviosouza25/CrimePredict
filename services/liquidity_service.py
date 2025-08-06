import requests
import pandas as pd
import numpy as np
from typing import Optional, Dict
from config.settings import API_KEY
import streamlit as st

class LiquidityService:
    """Serviço de análise de liquidez usando dados reais da Alpha Vantage"""
    
    @staticmethod
    def get_market_liquidity(pair: str, market_type: str = 'forex') -> Dict:
        """Analisa liquidez real do mercado usando dados da Alpha Vantage"""
        try:
            # Buscar dados de spread e volume para análise de liquidez
            liquidity_data = LiquidityService._fetch_liquidity_data(pair, market_type)
            
            if not liquidity_data:
                return LiquidityService._get_default_liquidity()
            
            # Calcular métricas de liquidez
            spread_analysis = LiquidityService._analyze_spread(liquidity_data)
            depth_analysis = LiquidityService._analyze_market_depth(pair, market_type)
            volatility_analysis = LiquidityService._analyze_volatility_liquidity(liquidity_data)
            
            # Combinar análises para score de liquidez
            liquidity_score = LiquidityService._calculate_liquidity_score(
                spread_analysis, depth_analysis, volatility_analysis
            )
            
            return {
                'liquidity_level': liquidity_score['level'],
                'liquidity_score': liquidity_score['score'],
                'spread_tightness': spread_analysis['tightness'],
                'market_depth': depth_analysis['depth'],
                'volatility_impact': volatility_analysis['impact'],
                'trading_recommendation': liquidity_score['recommendation'],
                'confidence': liquidity_score['confidence']
            }
            
        except Exception as e:
            st.warning(f"Erro ao analisar liquidez para {pair}: {str(e)}")
            return LiquidityService._get_default_liquidity()
    
    @staticmethod
    def _fetch_liquidity_data(pair: str, market_type: str) -> Optional[pd.DataFrame]:
        """Busca dados para análise de liquidez"""
        try:
            # Para forex, usar dados de alta frequência para análise de spread
            if market_type == 'forex':
                from_symbol, to_symbol = pair.split('/')
                params = {
                    'function': 'FX_INTRADAY',
                    'from_symbol': from_symbol,
                    'to_symbol': to_symbol,
                    'interval': '1min',
                    'apikey': API_KEY,
                    'outputsize': 'compact'
                }
            else:
                # Para crypto, usar dados digitais
                symbol = pair.split('/')[0]
                params = {
                    'function': 'DIGITAL_CURRENCY_INTRADAY',
                    'symbol': symbol,
                    'market': 'USD',
                    'interval': '5min',
                    'apikey': API_KEY
                }
            
            url = 'https://www.alphavantage.co/query'
            response = requests.get(url, params=params, timeout=30)
            
            if response.status_code != 200:
                return None
                
            data = response.json()
            
            # Verificar erros da API
            if 'Error Message' in data or 'Note' in data or 'Information' in data:
                return None
            
            # Encontrar chave de série temporal
            time_series_key = None
            for key in data.keys():
                if 'Time Series' in key:
                    time_series_key = key
                    break
            
            if not time_series_key:
                return None
            
            df = pd.DataFrame.from_dict(data[time_series_key], orient='index')
            
            # Normalizar colunas
            column_mapping = {
                '1. open': 'open',
                '2. high': 'high',
                '3. low': 'low',
                '4. close': 'close'
            }
            
            df = df.rename(columns=column_mapping)
            df = df.astype(float)
            df.index = pd.to_datetime(df.index)
            df = df.sort_index()
            
            return df
            
        except Exception:
            return None
    
    @staticmethod
    def _analyze_spread(df: pd.DataFrame) -> Dict:
        """Analisa tightness do spread usando dados OHLC"""
        if df is None or df.empty:
            return {'tightness': 'Média', 'spread_pct': 0.1}
        
        try:
            # Calcular spread médio usando high-low como proxy
            recent_data = df.tail(20)
            spreads = (recent_data['high'] - recent_data['low']) / recent_data['close'] * 100
            avg_spread = spreads.mean()
            
            # Classificar tightness do spread
            if avg_spread < 0.05:  # Spread muito baixo
                tightness = 'Muito Alta'
            elif avg_spread < 0.1:
                tightness = 'Alta'
            elif avg_spread < 0.2:
                tightness = 'Média'
            elif avg_spread < 0.5:
                tightness = 'Baixa'
            else:
                tightness = 'Muito Baixa'
            
            return {
                'tightness': tightness,
                'spread_pct': round(avg_spread, 4)
            }
            
        except Exception:
            return {'tightness': 'Média', 'spread_pct': 0.1}
    
    @staticmethod
    def _analyze_market_depth(pair: str, market_type: str) -> Dict:
        """Analisa profundidade do mercado baseado no tipo de par"""
        try:
            # Classificação de liquidez baseada em padrões reais do mercado
            major_forex_pairs = [
                'EUR/USD', 'GBP/USD', 'USD/JPY', 'USD/CHF', 
                'AUD/USD', 'USD/CAD', 'NZD/USD'
            ]
            
            minor_forex_pairs = [
                'EUR/GBP', 'EUR/JPY', 'GBP/JPY', 'CHF/JPY',
                'EUR/CHF', 'AUD/JPY', 'GBP/CHF'
            ]
            
            major_crypto_pairs = ['BTC/USD', 'ETH/USD']
            
            if market_type == 'forex':
                if pair in major_forex_pairs:
                    depth = 'Muito Alta'
                    depth_score = 0.9
                elif pair in minor_forex_pairs:
                    depth = 'Alta'
                    depth_score = 0.7
                else:
                    depth = 'Média'
                    depth_score = 0.5
            else:  # crypto
                if pair in major_crypto_pairs:
                    depth = 'Alta'
                    depth_score = 0.8
                else:
                    depth = 'Média'
                    depth_score = 0.6
            
            return {
                'depth': depth,
                'depth_score': depth_score
            }
            
        except Exception:
            return {'depth': 'Média', 'depth_score': 0.5}
    
    @staticmethod
    def _analyze_volatility_liquidity(df: pd.DataFrame) -> Dict:
        """Analisa impacto da volatilidade na liquidez"""
        if df is None or df.empty:
            return {'impact': 'Médio', 'volatility_score': 0.5}
        
        try:
            # Calcular volatilidade recente
            recent_prices = df['close'].tail(20)
            returns = recent_prices.pct_change().dropna()
            volatility = returns.std() * np.sqrt(24)  # Volatilidade diária
            
            # Classificar impacto da volatilidade na liquidez
            if volatility < 0.01:  # Baixa volatilidade = Melhor liquidez
                impact = 'Baixo'
                vol_score = 0.8
            elif volatility < 0.02:
                impact = 'Médio'
                vol_score = 0.6
            elif volatility < 0.05:
                impact = 'Alto'
                vol_score = 0.4
            else:
                impact = 'Muito Alto'
                vol_score = 0.2
            
            return {
                'impact': impact,
                'volatility_score': vol_score,
                'volatility_pct': round(volatility * 100, 2)
            }
            
        except Exception:
            return {'impact': 'Médio', 'volatility_score': 0.5}
    
    @staticmethod
    def _calculate_liquidity_score(spread_analysis: Dict, depth_analysis: Dict, volatility_analysis: Dict) -> Dict:
        """Calcula score final de liquidez"""
        try:
            # Pesos para cada componente
            spread_weight = 0.4
            depth_weight = 0.4
            volatility_weight = 0.2
            
            # Converter tightness para score
            tightness_scores = {
                'Muito Alta': 0.9,
                'Alta': 0.7,
                'Média': 0.5,
                'Baixa': 0.3,
                'Muito Baixa': 0.1
            }
            
            spread_score = tightness_scores.get(spread_analysis['tightness'], 0.5)
            depth_score = depth_analysis['depth_score']
            vol_score = volatility_analysis['volatility_score']
            
            # Score final ponderado
            final_score = (
                spread_score * spread_weight +
                depth_score * depth_weight +
                vol_score * volatility_weight
            )
            
            # Classificar nível de liquidez
            if final_score >= 0.8:
                level = 'Muito Alta'
                recommendation = 'ÓTIMA'
                confidence = 'Alta'
            elif final_score >= 0.6:
                level = 'Alta'
                recommendation = 'BOA'
                confidence = 'Alta'
            elif final_score >= 0.4:
                level = 'Média'
                recommendation = 'MODERADA'
                confidence = 'Média'
            elif final_score >= 0.2:
                level = 'Baixa'
                recommendation = 'CUIDADO'
                confidence = 'Baixa'
            else:
                level = 'Muito Baixa'
                recommendation = 'EVITAR'
                confidence = 'Muito Baixa'
            
            return {
                'level': level,
                'score': round(final_score, 3),
                'recommendation': recommendation,
                'confidence': confidence
            }
            
        except Exception:
            return {
                'level': 'Média',
                'score': 0.5,
                'recommendation': 'MODERADA',
                'confidence': 'Baixa'
            }
    
    @staticmethod
    def _get_default_liquidity() -> Dict:
        """Retorna valores padrão para liquidez"""
        return {
            'liquidity_level': 'Média',
            'liquidity_score': 0.5,
            'spread_tightness': 'Média',
            'market_depth': 'Média',
            'volatility_impact': 'Médio',
            'trading_recommendation': 'MODERADA',
            'confidence': 'Baixa'
        }