#!/usr/bin/env python3
"""
Debug script to test multi-timeframe analysis
"""

import sys
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Add current directory to path
sys.path.append('.')

def test_multi_timeframe_analysis():
    """Test the multi-timeframe analysis system"""
    
    print("=== Debug Multi-Timeframe Analysis ===")
    
    try:
        # Import main functions
        from app import (
            add_ema_indicators, 
            analyze_timeframe_trend,
            calculate_multi_timeframe_consensus,
            calculate_multi_timeframe_opportunity_score
        )
        print("✓ Successfully imported analysis functions")
        
        # Test with sample data
        dates = pd.date_range(start='2024-01-01', periods=300, freq='D')
        np.random.seed(42)  # For reproducible results
        
        # Create realistic forex price data
        base_price = 1.1000
        price_changes = np.random.normal(0, 0.001, 300)
        prices = [base_price]
        
        for change in price_changes[1:]:
            new_price = prices[-1] * (1 + change)
            prices.append(new_price)
        
        # Create OHLC data
        sample_data = pd.DataFrame({
            'Date': dates,
            'Open': [p * np.random.uniform(0.999, 1.001) for p in prices],
            'High': [p * np.random.uniform(1.001, 1.005) for p in prices],
            'Low': [p * np.random.uniform(0.995, 0.999) for p in prices],
            'Close': prices,
            'Volume': np.random.randint(1000000, 10000000, 300)
        })
        
        print(f"✓ Created sample data: {len(sample_data)} rows")
        print(f"  Price range: {sample_data['Close'].min():.5f} - {sample_data['Close'].max():.5f}")
        
        # Test EMA indicators
        df_with_ema = add_ema_indicators(sample_data)
        print(f"✓ Added EMA indicators")
        print(f"  EMA_20: {df_with_ema['EMA_20'].iloc[-1]:.5f}")
        print(f"  EMA_200: {df_with_ema['EMA_200'].iloc[-1]:.5f}")
        print(f"  Current Price: {df_with_ema['Close'].iloc[-1]:.5f}")
        
        # Test individual timeframe analysis
        timeframes = ['M5', 'M15', 'H1', 'D1']
        timeframe_results = {}
        
        for tf in timeframes:
            try:
                result = analyze_timeframe_trend(df_with_ema, 'EURUSD', tf, 'forex')
                timeframe_results[tf] = result
                print(f"✓ {tf}: {result['ema_signal']} (Prob: {result['probability']:.1f}%)")
            except Exception as e:
                print(f"✗ {tf}: Error - {e}")
                timeframe_results[tf] = None
        
        # Test consensus calculation
        if timeframe_results:
            consensus = calculate_multi_timeframe_consensus(timeframe_results)
            print(f"\n=== Consensus Results ===")
            print(f"Overall Direction: {consensus['overall_direction']}")
            print(f"Consensus Probability: {consensus['consensus_probability']:.1f}%")
            print(f"Consensus Confidence: {consensus['consensus_confidence']}")
            print(f"Timeframe Alignment: {consensus['timeframe_alignment']}")
            
            # Test opportunity score
            opportunity_score = calculate_multi_timeframe_opportunity_score(timeframe_results, 0.1)
            print(f"Opportunity Score: {opportunity_score:.1f}/100")
            
            # Check why no trends might be detected
            valid_count = sum(1 for tf, data in timeframe_results.items() if data and data.get('probability', 0) != 50)
            print(f"\nValid timeframes with trends: {valid_count}/{len(timeframes)}")
            
            for tf, data in timeframe_results.items():
                if data:
                    ema_20 = data.get('ema_20', 0)
                    ema_200 = data.get('ema_200', 0)
                    current_price = data.get('current_price', 0)
                    distance = data.get('ema_distance', 0)
                    
                    print(f"{tf}: Price={current_price:.5f}, EMA20={ema_20:.5f}, EMA200={ema_200:.5f}, Distance={distance:.2f}%")
        
        return True
        
    except Exception as e:
        print(f"✗ Error in analysis: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_data_service():
    """Test data service functionality"""
    
    print("\n=== Testing Data Service ===")
    
    try:
        from services.data_service import DataService
        from config.settings import ALPHA_VANTAGE_API_KEY
        
        data_service = DataService()
        
        # Test fetch for a simple pair
        test_pair = 'EURUSD'
        print(f"Testing data fetch for {test_pair}...")
        
        df = data_service.fetch_forex_data(test_pair, 'daily', 'compact', 'forex')
        
        if df is not None and not df.empty:
            print(f"✓ Data fetched successfully: {len(df)} rows")
            print(f"  Latest price: {df['Close'].iloc[-1]:.5f}")
            print(f"  Date range: {df.index[0]} to {df.index[-1]}")
            return True
        else:
            print("✗ No data returned from API")
            return False
            
    except Exception as e:
        print(f"✗ Data service error: {e}")
        return False

if __name__ == "__main__":
    print("Starting multi-timeframe analysis debug...")
    
    # Test analysis functions
    analysis_ok = test_multi_timeframe_analysis()
    
    # Test data service
    data_ok = test_data_service()
    
    if analysis_ok and data_ok:
        print("\n✓ All tests passed - system should work correctly")
    else:
        print("\n✗ Some tests failed - investigating issues")