#!/usr/bin/env python3

import sys
import os

# Add the pythonlibs path to ensure we can find packages
sys.path.insert(0, '/home/runner/workspace/.pythonlibs/lib/python3.11/site-packages')

print("Python path:")
for path in sys.path:
    print(f"  {path}")

print("\nTesting imports:")

# Test basic imports
try:
    import streamlit as st
    print("✓ Streamlit imported successfully")
except Exception as e:
    print(f"✗ Streamlit import failed: {e}")

try:
    import pandas as pd
    print("✓ Pandas imported successfully")
except Exception as e:
    print(f"✗ Pandas import failed: {e}")

try:
    import numpy as np
    print("✓ NumPy imported successfully")
except Exception as e:
    print(f"✗ NumPy import failed: {e}")

try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    print("✓ VaderSentiment imported successfully")
except Exception as e:
    print(f"✗ VaderSentiment import failed: {e}")

try:
    import torch
    print("✓ PyTorch imported successfully")
except Exception as e:
    print(f"✗ PyTorch import failed: {e}")

try:
    import plotly.graph_objects as go
    print("✓ Plotly imported successfully")
except Exception as e:
    print(f"✗ Plotly import failed: {e}")

try:
    from sklearn.preprocessing import MinMaxScaler
    print("✓ Scikit-learn imported successfully")
except Exception as e:
    print(f"✗ Scikit-learn import failed: {e}")

print("\nAll critical imports tested.")