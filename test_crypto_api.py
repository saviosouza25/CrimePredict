#!/usr/bin/env python3
"""
Test crypto API response to understand data format
"""

import requests
import json
import os
import pandas as pd

def test_crypto_api():
    """Test crypto API and understand response format"""
    
    API_KEY = os.getenv('ALPHA_VANTAGE_API_KEY')
    
    if not API_KEY:
        print("❌ No API key found")
        return
    
    print(f"Testing with API key: {API_KEY[:8]}...")
    
    # Test crypto daily data
    params = {
        'function': 'DIGITAL_CURRENCY_DAILY',
        'symbol': 'BTC',
        'market': 'USD',
        'apikey': API_KEY
    }
    
    url = 'https://www.alphavantage.co/query'
    
    try:
        print("Fetching BTC/USD daily data...")
        response = requests.get(url, params=params, timeout=30)
        response.raise_for_status()
        
        data = response.json()
        
        print("API Response keys:", list(data.keys()))
        
        # Check if there's an error
        if 'Error Message' in data:
            print(f"❌ API Error: {data['Error Message']}")
            return
        
        if 'Note' in data:
            print(f"⚠️ API Note: {data['Note']}")
            return
            
        if 'Information' in data:
            print(f"ℹ️ API Info: {data['Information']}")
            return
        
        # Find time series data
        time_series_key = None
        for key in data.keys():
            if 'Time Series' in key or 'Digital Currency' in key:
                time_series_key = key
                break
        
        if not time_series_key:
            print("❌ No time series key found")
            print("Available keys:", list(data.keys()))
            return
        
        print(f"✓ Found time series key: {time_series_key}")
        
        # Get first few entries to understand format
        time_series = data[time_series_key]
        first_date = list(time_series.keys())[0]
        first_entry = time_series[first_date]
        
        print(f"Sample date: {first_date}")
        print(f"Sample data keys: {list(first_entry.keys())}")
        print(f"Sample data: {first_entry}")
        
        # Try to create DataFrame
        df = pd.DataFrame.from_dict(time_series, orient='index')
        print(f"DataFrame shape: {df.shape}")
        print(f"DataFrame columns: {list(df.columns)}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_crypto_api()