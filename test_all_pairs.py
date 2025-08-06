#!/usr/bin/env python3
"""
Test all pairs to identify specific issues
"""

import sys
sys.path.append('.')

from services.data_service import DataService
from config.settings import PAIRS, CRYPTO_PAIRS

def test_all_pairs():
    """Test all forex and crypto pairs to identify issues"""
    
    data_service = DataService()
    
    print("=== Testing Forex Pairs ===")
    forex_success = 0
    forex_failed = []
    
    for pair in PAIRS:
        try:
            df = data_service.fetch_forex_data(pair, 'daily', 'compact', 'forex')
            if df is not None and not df.empty:
                print(f"✓ {pair}: {len(df)} rows")
                forex_success += 1
            else:
                print(f"✗ {pair}: No data")
                forex_failed.append(pair)
        except Exception as e:
            print(f"✗ {pair}: {str(e)}")
            forex_failed.append(pair)
    
    print(f"\nForex Summary: {forex_success}/{len(PAIRS)} successful")
    if forex_failed:
        print(f"Failed pairs: {forex_failed}")
    
    print("\n=== Testing Crypto Pairs ===")
    crypto_success = 0
    crypto_failed = []
    
    # Test only first 5 crypto pairs to avoid API limits
    test_cryptos = CRYPTO_PAIRS[:5]
    
    for pair in test_cryptos:
        try:
            df = data_service.fetch_forex_data(pair, 'daily', 'compact', 'crypto')
            if df is not None and not df.empty:
                print(f"✓ {pair}: {len(df)} rows")
                crypto_success += 1
            else:
                print(f"✗ {pair}: No data")
                crypto_failed.append(pair)
        except Exception as e:
            print(f"✗ {pair}: {str(e)}")
            crypto_failed.append(pair)
    
    print(f"\nCrypto Summary: {crypto_success}/{len(test_cryptos)} successful")
    if crypto_failed:
        print(f"Failed pairs: {crypto_failed}")

if __name__ == "__main__":
    test_all_pairs()