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
            if market_type == 'crypto':
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
        
        # Standardize column names for forex
        column_mapping = {
            '1. open': 'open',
            '2. high': 'high', 
            '3. low': 'low',
            '4. close': 'close',
            '5. volume': 'volume'
        }
        df = df.rename(columns=column_mapping)
        
        # Ensure required columns exist and normalize case
        required_columns = ['open', 'high', 'low', 'close']
        df.columns = [col.lower() for col in df.columns]
        
        # Capitalize column names to match expected format
        df = df.rename(columns={
            'open': 'Open',
            'high': 'High', 
            'low': 'Low',
            'close': 'Close',
            'volume': 'Volume'
        })
        
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
            df = DataService.fetch_forex_data(pair, '5min', 'compact', market_type)
            if not df.empty:
                return float(df['Close'].iloc[-1])
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
        
        # Handle different column formats for crypto - using actual API response format
        if function == 'DIGITAL_CURRENCY_DAILY':
            # Daily crypto format (confirmed by API test)
            column_mapping = {
                '1. open': 'open',
                '2. high': 'high', 
                '3. low': 'low',
                '4. close': 'close',
                '5. volume': 'volume'
            }
        else:
            # Intraday crypto format
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
        
        # Select only the columns we need
        df = df[required_columns + (['volume'] if 'volume' in df.columns else [])]
        
        # Normalize column names to match expected format (capitalized)
        df = df.rename(columns={
            'open': 'Open',
            'high': 'High', 
            'low': 'Low',
            'close': 'Close',
            'volume': 'Volume'
        })
        
        df = df.astype(np.float32)
        df.index = pd.to_datetime(df.index)
        df = df.sort_index()
        
        return df


