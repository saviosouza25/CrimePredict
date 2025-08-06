import pandas as pd
import numpy as np
from typing import Tuple

class TechnicalIndicators:
    """Enhanced technical indicators for forex analysis"""
    
    @staticmethod
    def sma(series: pd.Series, window: int) -> pd.Series:
        """Simple Moving Average"""
        return series.rolling(window=window).mean()
    
    @staticmethod
    def ema(series: pd.Series, window: int) -> pd.Series:
        """Exponential Moving Average"""
        return series.ewm(span=window).mean()
    
    @staticmethod
    def rsi(series: pd.Series, window: int = 14) -> pd.Series:
        """Relative Strength Index"""
        delta = series.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    @staticmethod
    def macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """MACD Indicator"""
        ema_fast = TechnicalIndicators.ema(series, fast)
        ema_slow = TechnicalIndicators.ema(series, slow)
        macd_line = ema_fast - ema_slow
        signal_line = TechnicalIndicators.ema(macd_line, signal)
        histogram = macd_line - signal_line
        return macd_line, signal_line, histogram
    
    @staticmethod
    def bollinger_bands(series: pd.Series, window: int = 20, std_dev: int = 2) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Bollinger Bands"""
        sma = TechnicalIndicators.sma(series, window)
        std = series.rolling(window=window).std()
        upper_band = sma + (std * std_dev)
        lower_band = sma - (std * std_dev)
        return upper_band, sma, lower_band
    
    @staticmethod
    def stochastic(high: pd.Series, low: pd.Series, close: pd.Series, k_window: int = 14, d_window: int = 3) -> Tuple[pd.Series, pd.Series]:
        """Stochastic Oscillator"""
        lowest_low = low.rolling(window=k_window).min()
        highest_high = high.rolling(window=k_window).max()
        k_percent = 100 * ((close - lowest_low) / (highest_high - lowest_low))
        d_percent = k_percent.rolling(window=d_window).mean()
        return k_percent, d_percent
    
    @staticmethod
    def atr(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14) -> pd.Series:
        """Average True Range"""
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        return true_range.rolling(window=window).mean()
    
    @staticmethod
    def add_all_indicators(df: pd.DataFrame) -> pd.DataFrame:
        """Add all technical indicators to dataframe"""
        df = df.copy()
        
        # Ensure we have the required columns
        required_cols = ['open', 'high', 'low', 'close']
        for col in required_cols:
            if col not in df.columns:
                raise ValueError(f"Required column '{col}' not found in dataframe")
        
        # Moving Averages
        df.loc[:, 'SMA_10'] = TechnicalIndicators.sma(df['close'], 10)
        df.loc[:, 'SMA_20'] = TechnicalIndicators.sma(df['close'], 20)
        df.loc[:, 'SMA_50'] = TechnicalIndicators.sma(df['close'], 50)
        df.loc[:, 'EMA_12'] = TechnicalIndicators.ema(df['close'], 12)
        df.loc[:, 'EMA_26'] = TechnicalIndicators.ema(df['close'], 26)
        
        # RSI
        df.loc[:, 'RSI'] = TechnicalIndicators.rsi(df['close'])
        
        # MACD
        macd, signal, histogram = TechnicalIndicators.macd(df['close'])
        df.loc[:, 'MACD'] = macd
        df.loc[:, 'MACD_Signal'] = signal
        df.loc[:, 'MACD_Histogram'] = histogram
        
        # Bollinger Bands
        bb_upper, bb_middle, bb_lower = TechnicalIndicators.bollinger_bands(df['close'])
        df.loc[:, 'BB_Upper'] = bb_upper
        df.loc[:, 'BB_Middle'] = bb_middle
        df.loc[:, 'BB_Lower'] = bb_lower
        df.loc[:, 'BB_Width'] = (bb_upper - bb_lower) / bb_middle
        df.loc[:, 'BB_Position'] = (df['close'] - bb_lower) / (bb_upper - bb_lower)
        
        # Stochastic
        stoch_k, stoch_d = TechnicalIndicators.stochastic(df['high'], df['low'], df['close'])
        df.loc[:, 'Stoch_K'] = stoch_k
        df.loc[:, 'Stoch_D'] = stoch_d
        
        # ATR
        df.loc[:, 'ATR'] = TechnicalIndicators.atr(df['high'], df['low'], df['close'])
        
        # Liquidity-based indicators removed - using dedicated liquidity service instead
        
        return df.dropna()
    
    @staticmethod
    def get_trading_signals(df: pd.DataFrame) -> pd.DataFrame:
        """Generate trading signals based on indicators"""
        signals = pd.DataFrame(index=df.index)
        
        # RSI signals
        signals['RSI_Oversold'] = df['RSI'] < 30
        signals['RSI_Overbought'] = df['RSI'] > 70
        
        # MACD signals
        signals['MACD_Bullish'] = (df['MACD'] > df['MACD_Signal']) & (df['MACD'].shift() <= df['MACD_Signal'].shift())
        signals['MACD_Bearish'] = (df['MACD'] < df['MACD_Signal']) & (df['MACD'].shift() >= df['MACD_Signal'].shift())
        
        # Moving Average signals
        signals['SMA_Bullish'] = df['close'] > df['SMA_20']
        signals['SMA_Bearish'] = df['close'] < df['SMA_20']
        
        # Bollinger Bands signals
        signals['BB_Squeeze'] = df['BB_Width'] < df['BB_Width'].rolling(20).mean()
        signals['BB_Breakout_Upper'] = df['close'] > df['BB_Upper']
        signals['BB_Breakout_Lower'] = df['close'] < df['BB_Lower']
        
        return signals
