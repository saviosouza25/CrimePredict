"""
Advanced Alpha Vantage Technical Indicators Service
Optimized for precision trend analysis by operational profile
"""
import requests
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from config.settings import API_KEY
from utils.cache_manager import CacheManager

class AlphaVantageIndicators:
    """Enhanced technical indicators using Alpha Vantage API with profile-specific optimization"""
    
    # Profile-specific indicator combinations for maximum trend precision
    PROFILE_INDICATORS = {
        'scalping': {
            'primary': ['EMA', 'MACD', 'RSI', 'STOCH'],
            'timeframes': ['1min', '5min'],
            'ema_periods': [8, 21],
            'rsi_period': 14,
            'macd_fast': 12, 'macd_slow': 26, 'macd_signal': 9,
            'stoch_k': 14, 'stoch_d': 3,
            'precision_weight': 0.85  # High precision needed for short-term
        },
        'intraday': {
            'primary': ['EMA', 'SMA', 'MACD', 'RSI', 'BBANDS', 'ADX'],
            'timeframes': ['15min', '30min', '60min'],
            'ema_periods': [12, 26, 50],
            'sma_periods': [20, 50],
            'rsi_period': 14,
            'macd_fast': 12, 'macd_slow': 26, 'macd_signal': 9,
            'bbands_period': 20, 'bbands_stddev': 2,
            'adx_period': 14,
            'precision_weight': 0.90
        },
        'swing': {
            'primary': ['SMA', 'EMA', 'MACD', 'RSI', 'BBANDS', 'ADX', 'SAR'],
            'timeframes': ['60min', '4hour', 'daily'],
            'sma_periods': [20, 50, 100, 200],
            'ema_periods': [21, 50, 100],
            'rsi_period': 14,
            'macd_fast': 12, 'macd_slow': 26, 'macd_signal': 9,
            'bbands_period': 20, 'bbands_stddev': 2,
            'adx_period': 14,
            'precision_weight': 0.95  # Maximum precision for medium-term
        },
        'position': {
            'primary': ['SMA', 'EMA', 'MACD', 'ADX', 'SAR', 'AROON'],
            'timeframes': ['daily', 'weekly'],
            'sma_periods': [50, 100, 200],
            'ema_periods': [50, 100, 200],
            'macd_fast': 12, 'macd_slow': 26, 'macd_signal': 9,
            'adx_period': 14,
            'aroon_period': 25,
            'precision_weight': 0.92
        }
    }
    
    @staticmethod
    def get_trend_analysis(pair: str, profile: str, interval: str) -> Dict:
        """
        Get comprehensive trend analysis optimized for specific trading profile
        Returns Alpha Vantage indicators with precision scoring
        """
        if profile not in AlphaVantageIndicators.PROFILE_INDICATORS:
            raise ValueError(f"Profile {profile} not supported")
            
        config = AlphaVantageIndicators.PROFILE_INDICATORS[profile]
        results = {
            'profile': profile,
            'pair': pair,
            'interval': interval,
            'indicators': {},
            'trend_signals': {},
            'precision_score': 0.0,
            'confidence': 0.0
        }
        
        # Get primary indicators for this profile
        for indicator in config['primary']:
            try:
                if indicator == 'SMA':
                    results['indicators']['sma'] = AlphaVantageIndicators._get_sma_signals(
                        pair, interval, config.get('sma_periods', [20, 50])
                    )
                elif indicator == 'EMA':
                    results['indicators']['ema'] = AlphaVantageIndicators._get_ema_signals(
                        pair, interval, config.get('ema_periods', [12, 26])
                    )
                elif indicator == 'MACD':
                    results['indicators']['macd'] = AlphaVantageIndicators._get_macd_signals(
                        pair, interval, config['macd_fast'], config['macd_slow'], config['macd_signal']
                    )
                elif indicator == 'RSI':
                    results['indicators']['rsi'] = AlphaVantageIndicators._get_rsi_signals(
                        pair, interval, config['rsi_period']
                    )
                elif indicator == 'BBANDS':
                    results['indicators']['bbands'] = AlphaVantageIndicators._get_bbands_signals(
                        pair, interval, config['bbands_period'], config['bbands_stddev']
                    )
                elif indicator == 'ADX':
                    results['indicators']['adx'] = AlphaVantageIndicators._get_adx_signals(
                        pair, interval, config['adx_period']
                    )
                elif indicator == 'STOCH':
                    results['indicators']['stoch'] = AlphaVantageIndicators._get_stoch_signals(
                        pair, interval, config['stoch_k'], config['stoch_d']
                    )
                elif indicator == 'SAR':
                    results['indicators']['sar'] = AlphaVantageIndicators._get_sar_signals(
                        pair, interval
                    )
                elif indicator == 'AROON':
                    results['indicators']['aroon'] = AlphaVantageIndicators._get_aroon_signals(
                        pair, interval, config['aroon_period']
                    )
                    
            except Exception as e:
                print(f"Error getting {indicator} for {pair}: {e}")
                continue
        
        # Calculate unified trend signals and precision
        results['trend_signals'] = AlphaVantageIndicators._calculate_unified_trend(
            results['indicators'], config
        )
        results['precision_score'] = config['precision_weight']
        
        return results
    
    @staticmethod
    def _get_sma_signals(pair: str, interval: str, periods: List[int]) -> Dict:
        """Get SMA signals from Alpha Vantage"""
        signals = {'values': {}, 'trend': 'NEUTRAL', 'strength': 0.0}
        
        for period in periods:
            try:
                url = f"https://www.alphavantage.co/query"
                params = {
                    'function': 'SMA',
                    'symbol': pair.replace('/', ''),
                    'interval': interval,
                    'time_period': period,
                    'series_type': 'close',
                    'apikey': API_KEY
                }
                
                response = requests.get(url, params=params, timeout=30)
                data = response.json()
                
                if 'Technical Analysis: SMA' in data:
                    sma_data = data['Technical Analysis: SMA']
                    latest_date = sorted(sma_data.keys())[-1]
                    signals['values'][f'sma_{period}'] = float(sma_data[latest_date]['SMA'])
                    
            except Exception as e:
                print(f"Error fetching SMA {period}: {e}")
                continue
        
        # Calculate SMA trend signals
        if len(signals['values']) >= 2:
            sma_values = list(signals['values'].values())
            if sma_values[0] > sma_values[1]:  # Short SMA > Long SMA
                signals['trend'] = 'BULLISH'
                signals['strength'] = 0.8
            elif sma_values[0] < sma_values[1]:  # Short SMA < Long SMA
                signals['trend'] = 'BEARISH'
                signals['strength'] = 0.8
        
        return signals
    
    @staticmethod
    def _get_ema_signals(pair: str, interval: str, periods: List[int]) -> Dict:
        """Get EMA signals from Alpha Vantage"""
        signals = {'values': {}, 'trend': 'NEUTRAL', 'strength': 0.0}
        
        for period in periods:
            try:
                url = f"https://www.alphavantage.co/query"
                params = {
                    'function': 'EMA',
                    'symbol': pair.replace('/', ''),
                    'interval': interval,
                    'time_period': period,
                    'series_type': 'close',
                    'apikey': API_KEY
                }
                
                response = requests.get(url, params=params, timeout=30)
                data = response.json()
                
                if 'Technical Analysis: EMA' in data:
                    ema_data = data['Technical Analysis: EMA']
                    latest_date = sorted(ema_data.keys())[-1]
                    signals['values'][f'ema_{period}'] = float(ema_data[latest_date]['EMA'])
                    
            except Exception as e:
                print(f"Error fetching EMA {period}: {e}")
                continue
        
        # Calculate EMA trend signals (similar to SMA but more responsive)
        if len(signals['values']) >= 2:
            ema_values = list(signals['values'].values())
            if ema_values[0] > ema_values[1]:
                signals['trend'] = 'BULLISH'
                signals['strength'] = 0.85  # EMA more responsive than SMA
            elif ema_values[0] < ema_values[1]:
                signals['trend'] = 'BEARISH'
                signals['strength'] = 0.85
        
        return signals
    
    @staticmethod
    def _get_macd_signals(pair: str, interval: str, fast: int, slow: int, signal: int) -> Dict:
        """Get MACD signals from Alpha Vantage"""
        signals = {'macd': 0, 'signal': 0, 'histogram': 0, 'trend': 'NEUTRAL', 'strength': 0.0}
        
        try:
            url = f"https://www.alphavantage.co/query"
            params = {
                'function': 'MACD',
                'symbol': pair.replace('/', ''),
                'interval': interval,
                'series_type': 'close',
                'fastperiod': fast,
                'slowperiod': slow,
                'signalperiod': signal,
                'apikey': API_KEY
            }
            
            response = requests.get(url, params=params, timeout=30)
            data = response.json()
            
            if 'Technical Analysis: MACD' in data:
                macd_data = data['Technical Analysis: MACD']
                latest_date = sorted(macd_data.keys())[-1]
                
                signals['macd'] = float(macd_data[latest_date]['MACD'])
                signals['signal'] = float(macd_data[latest_date]['MACD_Signal'])
                signals['histogram'] = float(macd_data[latest_date]['MACD_Hist'])
                
                # MACD trend analysis
                if signals['macd'] > signals['signal'] and signals['histogram'] > 0:
                    signals['trend'] = 'BULLISH'
                    signals['strength'] = min(abs(signals['histogram']) * 10, 1.0)
                elif signals['macd'] < signals['signal'] and signals['histogram'] < 0:
                    signals['trend'] = 'BEARISH'
                    signals['strength'] = min(abs(signals['histogram']) * 10, 1.0)
                    
        except Exception as e:
            print(f"Error fetching MACD: {e}")
        
        return signals
    
    @staticmethod
    def _get_rsi_signals(pair: str, interval: str, period: int) -> Dict:
        """Get RSI signals from Alpha Vantage"""
        signals = {'value': 50, 'trend': 'NEUTRAL', 'strength': 0.0, 'overbought': False, 'oversold': False}
        
        try:
            url = f"https://www.alphavantage.co/query"
            params = {
                'function': 'RSI',
                'symbol': pair.replace('/', ''),
                'interval': interval,
                'time_period': period,
                'series_type': 'close',
                'apikey': API_KEY
            }
            
            response = requests.get(url, params=params, timeout=30)
            data = response.json()
            
            if 'Technical Analysis: RSI' in data:
                rsi_data = data['Technical Analysis: RSI']
                latest_date = sorted(rsi_data.keys())[-1]
                
                signals['value'] = float(rsi_data[latest_date]['RSI'])
                
                # RSI analysis
                if signals['value'] > 70:
                    signals['overbought'] = True
                    signals['trend'] = 'BEARISH'  # Reversal expected
                    signals['strength'] = (signals['value'] - 70) / 30
                elif signals['value'] < 30:
                    signals['oversold'] = True
                    signals['trend'] = 'BULLISH'  # Reversal expected
                    signals['strength'] = (30 - signals['value']) / 30
                elif signals['value'] > 50:
                    signals['trend'] = 'BULLISH'
                    signals['strength'] = 0.5
                elif signals['value'] < 50:
                    signals['trend'] = 'BEARISH'
                    signals['strength'] = 0.5
                    
        except Exception as e:
            print(f"Error fetching RSI: {e}")
        
        return signals
    
    @staticmethod
    def _get_bbands_signals(pair: str, interval: str, period: int, stddev: float) -> Dict:
        """Get Bollinger Bands signals from Alpha Vantage"""
        signals = {'upper': 0, 'middle': 0, 'lower': 0, 'trend': 'NEUTRAL', 'strength': 0.0}
        
        try:
            url = f"https://www.alphavantage.co/query"
            params = {
                'function': 'BBANDS',
                'symbol': pair.replace('/', ''),
                'interval': interval,
                'time_period': period,
                'series_type': 'close',
                'nbdevup': stddev,
                'nbdevdn': stddev,
                'matype': 0,  # SMA
                'apikey': API_KEY
            }
            
            response = requests.get(url, params=params, timeout=30)
            data = response.json()
            
            if 'Technical Analysis: BBANDS' in data:
                bb_data = data['Technical Analysis: BBANDS']
                latest_date = sorted(bb_data.keys())[-1]
                
                signals['upper'] = float(bb_data[latest_date]['Real Upper Band'])
                signals['middle'] = float(bb_data[latest_date]['Real Middle Band'])
                signals['lower'] = float(bb_data[latest_date]['Real Lower Band'])
                
                # Get current price to compare with bands
                # This would need current price - simplified for now
                band_width = signals['upper'] - signals['lower']
                middle_position = (signals['upper'] + signals['lower']) / 2
                
                # Trend based on band expansion/contraction
                signals['trend'] = 'NEUTRAL'
                signals['strength'] = 0.6
                    
        except Exception as e:
            print(f"Error fetching Bollinger Bands: {e}")
        
        return signals
    
    @staticmethod
    def _get_adx_signals(pair: str, interval: str, period: int) -> Dict:
        """Get ADX (Average Directional Index) signals from Alpha Vantage"""
        signals = {'adx': 0, 'trend': 'NEUTRAL', 'strength': 0.0, 'trend_strong': False}
        
        try:
            url = f"https://www.alphavantage.co/query"
            params = {
                'function': 'ADX',
                'symbol': pair.replace('/', ''),
                'interval': interval,
                'time_period': period,
                'apikey': API_KEY
            }
            
            response = requests.get(url, params=params, timeout=30)
            data = response.json()
            
            if 'Technical Analysis: ADX' in data:
                adx_data = data['Technical Analysis: ADX']
                latest_date = sorted(adx_data.keys())[-1]
                
                signals['adx'] = float(adx_data[latest_date]['ADX'])
                
                # ADX trend strength analysis
                if signals['adx'] > 25:
                    signals['trend_strong'] = True
                    signals['strength'] = min(signals['adx'] / 100, 1.0)
                    signals['trend'] = 'STRONG_TREND'  # Direction determined by other indicators
                else:
                    signals['trend_strong'] = False
                    signals['trend'] = 'WEAK_TREND'
                    signals['strength'] = 0.3
                    
        except Exception as e:
            print(f"Error fetching ADX: {e}")
        
        return signals
    
    @staticmethod
    def _get_stoch_signals(pair: str, interval: str, k_period: int, d_period: int) -> Dict:
        """Get Stochastic signals from Alpha Vantage"""
        signals = {'k': 50, 'd': 50, 'trend': 'NEUTRAL', 'strength': 0.0}
        
        try:
            url = f"https://www.alphavantage.co/query"
            params = {
                'function': 'STOCH',
                'symbol': pair.replace('/', ''),
                'interval': interval,
                'fastkperiod': k_period,
                'slowdperiod': d_period,
                'apikey': API_KEY
            }
            
            response = requests.get(url, params=params, timeout=30)
            data = response.json()
            
            if 'Technical Analysis: STOCH' in data:
                stoch_data = data['Technical Analysis: STOCH']
                latest_date = sorted(stoch_data.keys())[-1]
                
                signals['k'] = float(stoch_data[latest_date]['SlowK'])
                signals['d'] = float(stoch_data[latest_date]['SlowD'])
                
                # Stochastic analysis
                if signals['k'] > 80 and signals['d'] > 80:
                    signals['trend'] = 'BEARISH'  # Overbought
                    signals['strength'] = 0.8
                elif signals['k'] < 20 and signals['d'] < 20:
                    signals['trend'] = 'BULLISH'  # Oversold
                    signals['strength'] = 0.8
                elif signals['k'] > signals['d']:
                    signals['trend'] = 'BULLISH'
                    signals['strength'] = 0.6
                else:
                    signals['trend'] = 'BEARISH'
                    signals['strength'] = 0.6
                    
        except Exception as e:
            print(f"Error fetching Stochastic: {e}")
        
        return signals
    
    @staticmethod
    def _get_sar_signals(pair: str, interval: str) -> Dict:
        """Get Parabolic SAR signals from Alpha Vantage"""
        signals = {'sar': 0, 'trend': 'NEUTRAL', 'strength': 0.0}
        
        try:
            url = f"https://www.alphavantage.co/query"
            params = {
                'function': 'SAR',
                'symbol': pair.replace('/', ''),
                'interval': interval,
                'acceleration': 0.02,
                'maximum': 0.20,
                'apikey': API_KEY
            }
            
            response = requests.get(url, params=params, timeout=30)
            data = response.json()
            
            if 'Technical Analysis: SAR' in data:
                sar_data = data['Technical Analysis: SAR']
                latest_date = sorted(sar_data.keys())[-1]
                
                signals['sar'] = float(sar_data[latest_date]['SAR'])
                
                # SAR analysis requires current price comparison
                # Simplified for now
                signals['trend'] = 'NEUTRAL'
                signals['strength'] = 0.7
                    
        except Exception as e:
            print(f"Error fetching SAR: {e}")
        
        return signals
    
    @staticmethod
    def _get_aroon_signals(pair: str, interval: str, period: int) -> Dict:
        """Get Aroon signals from Alpha Vantage"""
        signals = {'aroon_up': 0, 'aroon_down': 0, 'trend': 'NEUTRAL', 'strength': 0.0}
        
        try:
            url = f"https://www.alphavantage.co/query"
            params = {
                'function': 'AROON',
                'symbol': pair.replace('/', ''),
                'interval': interval,
                'time_period': period,
                'apikey': API_KEY
            }
            
            response = requests.get(url, params=params, timeout=30)
            data = response.json()
            
            if 'Technical Analysis: AROON' in data:
                aroon_data = data['Technical Analysis: AROON']
                latest_date = sorted(aroon_data.keys())[-1]
                
                signals['aroon_up'] = float(aroon_data[latest_date]['Aroon Up'])
                signals['aroon_down'] = float(aroon_data[latest_date]['Aroon Down'])
                
                # Aroon analysis
                if signals['aroon_up'] > 70 and signals['aroon_down'] < 30:
                    signals['trend'] = 'BULLISH'
                    signals['strength'] = 0.9
                elif signals['aroon_down'] > 70 and signals['aroon_up'] < 30:
                    signals['trend'] = 'BEARISH'
                    signals['strength'] = 0.9
                elif signals['aroon_up'] > signals['aroon_down']:
                    signals['trend'] = 'BULLISH'
                    signals['strength'] = 0.6
                else:
                    signals['trend'] = 'BEARISH'
                    signals['strength'] = 0.6
                    
        except Exception as e:
            print(f"Error fetching Aroon: {e}")
        
        return signals
    
    @staticmethod
    def _calculate_unified_trend(indicators: Dict, config: Dict) -> Dict:
        """Calculate unified trend signals from all indicators"""
        trends = []
        strengths = []
        
        for indicator, data in indicators.items():
            if 'trend' in data and 'strength' in data:
                if data['trend'] == 'BULLISH':
                    trends.append(1)
                elif data['trend'] == 'BEARISH':
                    trends.append(-1)
                else:
                    trends.append(0)
                strengths.append(data['strength'])
        
        if not trends:
            return {'unified_trend': 'NEUTRAL', 'confidence': 0.0, 'strength': 0.0}
        
        # Calculate weighted average
        avg_trend = sum(trends) / len(trends)
        avg_strength = sum(strengths) / len(strengths)
        
        # Determine unified trend
        if avg_trend > 0.3:
            unified_trend = 'BULLISH'
        elif avg_trend < -0.3:
            unified_trend = 'BEARISH'
        else:
            unified_trend = 'NEUTRAL'
        
        # Confidence based on agreement between indicators
        agreement = len([t for t in trends if t == (1 if avg_trend > 0 else -1 if avg_trend < 0 else 0)]) / len(trends)
        confidence = agreement * config['precision_weight']
        
        return {
            'unified_trend': unified_trend,
            'confidence': confidence,
            'strength': avg_strength,
            'indicators_count': len(trends),
            'agreement_rate': agreement
        }