import requests
import pandas as pd
import numpy as np
import streamlit as st
from typing import Tuple, Optional
from config.settings import API_KEY, DATA_CACHE_TTL
from utils.cache_manager import CacheManager

class DataService:
    """Enhanced data service with caching and error handling"""
    
    @staticmethod
    def fetch_forex_data(pair: str, interval: str = '60min', outputsize: str = 'compact', market_type: str = 'forex') -> pd.DataFrame:
        """Fetch forex or crypto data with caching and improved error handling"""
        cache_key = CacheManager.get_cache_key(f'{market_type}_data', pair, interval, outputsize)
        cached_data = CacheManager.get_cached_data(cache_key, DATA_CACHE_TTL)
        
        if cached_data is not None:
            return cached_data
        
        try:
            # Special handling for gold (XAU/USD) which has limited support
            if pair in ['XAU/USD', 'XAUUSD']:
                df = DataService._fetch_gold_data_internal(pair, interval, outputsize)
            elif market_type == 'crypto':
                df = DataService._fetch_crypto_data_internal(pair, interval, outputsize)
            else:
                df = DataService._fetch_forex_data_internal(pair, interval, outputsize)
            
            # Cache the successful data
            CacheManager.set_cached_data(cache_key, df)
            return df
                
        except Exception as e:
            st.error(f"Error fetching {market_type} data for {pair}: {str(e)}")
            raise

    @staticmethod
    def _fetch_forex_data_internal(pair: str, interval: str, outputsize: str) -> pd.DataFrame:
        """Internal forex data fetching"""
        from_symbol, to_symbol = pair.split('/')
        function = 'FX_INTRADAY' if interval != 'daily' else 'FX_DAILY'
        
        params = {
            'function': function,
            'from_symbol': from_symbol,
            'to_symbol': to_symbol,
            'apikey': API_KEY,
            'outputsize': outputsize
        }
        
        if interval != 'daily':
            params['interval'] = interval
        
        url = 'https://www.alphavantage.co/query'
        response = requests.get(url, params=params, timeout=30)
        response.raise_for_status()
        
        data = response.json()
        
        # Handle API errors
        if 'Error Message' in data:
            raise ValueError(f"API Error: {data['Error Message']}")
        if 'Note' in data:
            raise ValueError(f"API Limit: {data['Note']}")
        if 'Information' in data:
            raise ValueError(f"API Info: {data['Information']}")
        
        # Find the correct time series key
        time_series_key = None
        for key in data.keys():
            if 'Time Series' in key:
                time_series_key = key
                break
        
        if not time_series_key:
            raise ValueError("No time series data found in API response")
        
        df = pd.DataFrame.from_dict(data[time_series_key], orient='index')
        df = df.astype(np.float32)
        df.index = pd.to_datetime(df.index)
        df = df.sort_index()
        
        # Standardize column names
        column_mapping = {
            '1. open': 'open',
            '2. high': 'high', 
            '3. low': 'low',
            '4. close': 'close'
        }
        df = df.rename(columns=column_mapping)
        
        if df.empty:
            raise ValueError("No data received from API")
        
        return df


    
    @staticmethod
    def validate_data(df: pd.DataFrame, min_rows: int = 100) -> bool:
        """Validate data quality"""
        if df.empty:
            return False
        if len(df) < min_rows:
            return False
        if df.isnull().sum().sum() > len(df) * 0.1:  # More than 10% null values
            return False
        return True
    
    @staticmethod
    def get_latest_price(pair: str, market_type: str = 'forex') -> Optional[float]:
        """Get the latest price for a currency pair or crypto"""
        try:
            # Special handling for gold - use real-time exchange rate
            if pair in ['XAU/USD', 'XAUUSD']:
                params = {
                    'function': 'CURRENCY_EXCHANGE_RATE',
                    'from_currency': 'XAU',
                    'to_currency': 'USD',
                    'apikey': API_KEY
                }
                url = 'https://www.alphavantage.co/query'
                response = requests.get(url, params=params, timeout=30)
                response.raise_for_status()
                data = response.json()
                
                if 'Realtime Currency Exchange Rate' in data:
                    return float(data['Realtime Currency Exchange Rate']['5. Exchange Rate'])
                    
            df = DataService.fetch_forex_data(pair, '5min', 'compact', market_type)
            if not df.empty:
                return float(df['close'].iloc[-1])
        except Exception:
            pass
        return None

    @staticmethod
    def _fetch_crypto_data_internal(pair: str, interval: str, outputsize: str) -> pd.DataFrame:
        """Internal crypto data fetching using Alpha Vantage crypto endpoints"""
        symbol, market = pair.split('/')
        
        # Use appropriate crypto function based on interval
        if interval == 'daily':
            function = 'DIGITAL_CURRENCY_DAILY'
        else:
            function = 'CRYPTO_INTRADAY'
        
        params = {
            'function': function,
            'symbol': symbol,
            'market': market,
            'apikey': API_KEY
        }
        
        if function == 'CRYPTO_INTRADAY':
            params['interval'] = interval
            if outputsize != 'compact':
                params['outputsize'] = outputsize
        
        url = 'https://www.alphavantage.co/query'
        response = requests.get(url, params=params, timeout=30)
        response.raise_for_status()
        
        data = response.json()
        
        # Handle API errors
        if 'Error Message' in data:
            raise ValueError(f"API Error: {data['Error Message']}")
        if 'Note' in data:
            raise ValueError(f"API Limit: {data['Note']}")
        if 'Information' in data:
            raise ValueError(f"API Info: {data['Information']}")
        
        # Find the correct time series key for crypto
        time_series_key = None
        for key in data.keys():
            if 'Time Series' in key or 'Crypto' in key:
                time_series_key = key
                break
        
        if not time_series_key:
            raise ValueError("No time series data found in crypto API response")
        
        df = pd.DataFrame.from_dict(data[time_series_key], orient='index')
        
        # Handle different column formats for crypto
        if function == 'DIGITAL_CURRENCY_DAILY':
            # Daily format has different column names
            column_mapping = {
                '1a. open (USD)': 'open',
                '2a. high (USD)': 'high', 
                '3a. low (USD)': 'low',
                '4a. close (USD)': 'close',
                '5. volume': 'volume'
            }
        else:
            # Intraday format
            column_mapping = {
                '1. open': 'open',
                '2. high': 'high',
                '3. low': 'low', 
                '4. close': 'close',
                '5. volume': 'volume'
            }
        
        # Rename columns to match expected format
        df = df.rename(columns=column_mapping)
        
        # Ensure all required columns exist
        required_columns = ['open', 'high', 'low', 'close']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
        
        df = df[required_columns + (['volume'] if 'volume' in df.columns else [])]
        df = df.astype(np.float32)
        df.index = pd.to_datetime(df.index)
        df = df.sort_index()
        
        return df

    @staticmethod
    def _fetch_gold_data_internal(pair: str, interval: str, outputsize: str) -> pd.DataFrame:
        """Internal gold data fetching using Alpha Vantage with limited support"""
        # Gold has limited support - use currency exchange rate function
        params = {
            'function': 'CURRENCY_EXCHANGE_RATE',
            'from_currency': 'XAU',
            'to_currency': 'USD',
            'apikey': API_KEY
        }
        
        url = 'https://www.alphavantage.co/query'
        response = requests.get(url, params=params, timeout=30)
        response.raise_for_status()
        
        data = response.json()
        
        # Handle API errors
        if 'Error Message' in data:
            raise ValueError(f"API Error: {data['Error Message']}")
        if 'Note' in data:
            raise ValueError(f"API Limit: {data['Note']}")
        if 'Information' in data:
            raise ValueError(f"API Info: {data['Information']}")
        
        # Gold returns real-time exchange rate only
        if 'Realtime Currency Exchange Rate' in data:
            rate_data = data['Realtime Currency Exchange Rate']
            current_price = float(rate_data['5. Exchange Rate'])
            
            # Gold only provides real-time exchange rate, not historical OHLC data
            # Alpha Vantage limitation - no historical data available for gold
            raise ValueError(f"❌ ANÁLISE BLOQUEADA: Gold (XAU/USD) não possui dados históricos OHLC na API Alpha Vantage. Apenas taxa de câmbio em tempo real disponível: ${current_price:.2f}. Use apenas para preço atual, não para análise técnica.")
        else:
            raise ValueError("No gold exchange rate data found in API response")
