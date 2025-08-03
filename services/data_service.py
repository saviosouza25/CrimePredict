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
    def fetch_forex_data(pair: str, interval: str = '60min', outputsize: str = 'compact') -> pd.DataFrame:
        """Fetch forex data with caching and improved error handling"""
        cache_key = CacheManager.get_cache_key('forex_data', pair, interval, outputsize)
        cached_data = CacheManager.get_cached_data(cache_key, DATA_CACHE_TTL)
        
        if cached_data is not None:
            return cached_data
        
        try:
            # Special handling for different asset types
            if pair.startswith('XAU') or pair.startswith('XAG') or pair.startswith('XPT') or pair.startswith('XPD'):
                # Precious metals - use different function
                return DataService._fetch_commodity_data(pair, interval, outputsize)
            elif pair.startswith('BTC') or pair.startswith('ETH') or pair.startswith('LTC') or pair.startswith('XRP'):
                # Cryptocurrencies - use different function
                return DataService._fetch_crypto_data(pair, interval, outputsize)
            elif pair.startswith('WTI') or pair.startswith('BRT'):
                # Oil/Energy - use commodity function
                return DataService._fetch_commodity_data(pair, interval, outputsize)
            else:
                # Regular forex pairs
                return DataService._fetch_forex_data_internal(pair, interval, outputsize)
                
        except Exception as e:
            # If API fails, use demo data with warning
            st.warning(f"⚠️ API indisponível para {pair}. Usando dados de demonstração.")
            return DataService._generate_demo_data(pair)

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
    def _fetch_commodity_data(pair: str, interval: str, outputsize: str) -> pd.DataFrame:
        """Fetch commodity data (Gold, Silver, Oil, etc.)"""
        try:
            # For precious metals, use different Alpha Vantage functions
            if pair.startswith('XAU'):
                function = 'TIME_SERIES_DAILY'
                symbol = 'GLD'  # Gold ETF as proxy
            elif pair.startswith('XAG'):
                function = 'TIME_SERIES_DAILY' 
                symbol = 'SLV'  # Silver ETF as proxy
            else:
                # For other commodities, generate demo data
                raise ValueError("Commodity not supported by API")
                
            params = {
                'function': function,
                'symbol': symbol,
                'apikey': API_KEY,
                'outputsize': outputsize
            }
            
            url = 'https://www.alphavantage.co/query'
            response = requests.get(url, params=params, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            
            if 'Error Message' in data or 'Note' in data or 'Information' in data:
                raise ValueError("API error for commodity data")
            
            # Find time series data
            time_series_key = 'Time Series (Daily)'
            if time_series_key in data:
                df = pd.DataFrame.from_dict(data[time_series_key], orient='index')
                df = df.astype(np.float32)
                df.index = pd.to_datetime(df.index)
                df = df.sort_index()
                
                column_mapping = {
                    '1. open': 'open',
                    '2. high': 'high',
                    '3. low': 'low', 
                    '4. close': 'close'
                }
                df = df.rename(columns=column_mapping)
                return df
            else:
                raise ValueError("No commodity data found")
                
        except Exception:
            # Generate demo data for commodities
            return DataService._generate_demo_data(pair)

    @staticmethod
    def _fetch_crypto_data(pair: str, interval: str, outputsize: str) -> pd.DataFrame:
        """Fetch cryptocurrency data"""
        try:
            from_symbol = pair.split('/')[0]
            params = {
                'function': 'DIGITAL_CURRENCY_DAILY',
                'symbol': from_symbol,
                'market': 'USD',
                'apikey': API_KEY
            }
            
            url = 'https://www.alphavantage.co/query'
            response = requests.get(url, params=params, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            
            if 'Error Message' in data or 'Note' in data or 'Information' in data:
                raise ValueError("API error for crypto data")
                
            time_series_key = 'Time Series (Digital Currency Daily)'
            if time_series_key in data:
                df = pd.DataFrame.from_dict(data[time_series_key], orient='index')
                
                # Use USD prices
                df = df[[col for col in df.columns if '(USD)' in col]]
                df.columns = ['open', 'high', 'low', 'close', 'volume', 'market_cap']
                df = df[['open', 'high', 'low', 'close']].astype(np.float32)
                df.index = pd.to_datetime(df.index)
                df = df.sort_index()
                return df
            else:
                raise ValueError("No crypto data found")
                
        except Exception:
            # Generate demo data for crypto
            return DataService._generate_demo_data(pair)

    @staticmethod
    def _generate_demo_data(pair: str) -> pd.DataFrame:
        """Generate realistic demo data for any trading pair"""
        import datetime
        
        # Base prices for different asset types
        base_prices = {
            'EUR/USD': 1.0500, 'USD/JPY': 149.50, 'GBP/USD': 1.2600, 'AUD/USD': 0.6500,
            'USD/CAD': 1.3600, 'USD/CHF': 0.8950, 'NZD/USD': 0.5900, 'EUR/GBP': 0.8650,
            'EUR/JPY': 157.00, 'GBP/JPY': 188.50, 'XAU/USD': 2050.00, 'XAG/USD': 24.50,
            'BTC/USD': 42500.00, 'ETH/USD': 2650.00, 'WTI/USD': 75.50, 'BRT/USD': 80.20
        }
        
        base_price = base_prices.get(pair, 1.0000)
        
        # Generate 200 data points
        dates = pd.date_range(end=datetime.datetime.now(), periods=200, freq='1H')
        np.random.seed(42)  # For reproducible demo data
        
        # Generate realistic price movements
        returns = np.random.normal(0, 0.005, 200)  # 0.5% volatility
        prices = []
        current_price = base_price
        
        for ret in returns:
            current_price = current_price * (1 + ret)
            prices.append(current_price)
        
        # Create OHLC data
        data = []
        for i, price in enumerate(prices):
            high = price * (1 + abs(np.random.normal(0, 0.002)))
            low = price * (1 - abs(np.random.normal(0, 0.002)))
            open_price = prices[i-1] if i > 0 else price
            
            data.append({
                'open': open_price,
                'high': max(price, high, open_price),
                'low': min(price, low, open_price),
                'close': price
            })
        
        df = pd.DataFrame(data, index=dates)
        return df.astype(np.float32)
    
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
    def get_latest_price(pair: str) -> Optional[float]:
        """Get the latest price for a currency pair"""
        try:
            df = DataService.fetch_forex_data(pair, '1min', 'compact')
            if not df.empty:
                return float(df['close'].iloc[-1])
        except Exception:
            pass
        return None
