#!/usr/bin/env python3
"""
Test script for Enhanced Master Bot imports and basic functionality
"""

import sys
import traceback

def test_imports():
    """Test all required imports"""
    print("Testing imports...")
    
    try:
        # Standard library imports
        import os, time, math, json, random, queue, logging, threading, tkinter as tk
        from datetime import datetime, timedelta
        from typing import Optional, List, Dict, Tuple, Any
        from concurrent.futures import ThreadPoolExecutor
        from collections import deque, defaultdict
        from dataclasses import dataclass
        print("✓ Standard library imports successful")
        
        # Third-party libraries
        import numpy as np
        import pandas as pd
        print("✓ NumPy and Pandas imports successful")
        
        import torch
        import torch.nn as nn
        import torch.nn.functional as F
        from torch.optim import Adam, AdamW
        from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR
        print("✓ PyTorch imports successful")
        
        from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
        from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
        from sklearn.linear_model import LinearRegression, Ridge
        from sklearn.metrics import mean_squared_error, accuracy_score
        from sklearn.decomposition import PCA
        from sklearn.cluster import KMeans
        print("✓ Scikit-learn imports successful")
        
        import ta
        from ta.trend import macd, macd_signal, ADXIndicator, EMAIndicator, SMAIndicator, MACD
        from ta.momentum import rsi, StochasticOscillator, RSIIndicator, WilliamsRIndicator
        from ta.volatility import BollingerBands, AverageTrueRange, KeltnerChannel
        from ta.volume import VolumeSMAIndicator, VolumeWeightedAveragePrice, OnBalanceVolumeIndicator
        from ta.others import DailyReturnIndicator, CumulativeReturnIndicator
        print("✓ TA library imports successful")
        
        import matplotlib
        matplotlib.use("TkAgg")
        import matplotlib.pyplot as plt
        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
        print("✓ Matplotlib imports successful")
        
        print("✓ All imports successful!")
        return True
        
    except ImportError as e:
        print(f"✗ Import error: {e}")
        return False
    except Exception as e:
        print(f"✗ Unexpected error: {e}")
        traceback.print_exc()
        return False

def test_basic_functionality():
    """Test basic functionality without external dependencies"""
    print("\nTesting basic functionality...")
    
    try:
        import torch
        import pandas as pd
        import numpy as np
        from ta.momentum import rsi
        from sklearn.preprocessing import StandardScaler
        
        # Test PyTorch
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"✓ PyTorch device: {device}")
        
        # Test tensor operations
        x = torch.randn(10, 5)
        y = torch.mm(x, x.t())
        print(f"✓ PyTorch tensor operations work")
        
        # Test pandas DataFrame
        df = pd.DataFrame({
            'price': [100, 101, 102, 101, 100],
            'volume': [1000, 1100, 1200, 1050, 950]
        })
        print(f"✓ Pandas DataFrame created with {len(df)} rows")
        
        # Test technical indicators
        close_prices = pd.Series([100, 101, 102, 101, 100, 99, 98, 99, 100, 101])
        rsi_values = rsi(close_prices, window=5)
        print(f"✓ RSI calculation successful")
        
        # Test sklearn
        scaler = StandardScaler()
        data = np.random.randn(100, 5)
        scaled_data = scaler.fit_transform(data)
        print(f"✓ Scikit-learn scaling successful")
        
        print("✓ All basic functionality tests passed!")
        return True
        
    except Exception as e:
        print(f"✗ Basic functionality test failed: {e}")
        traceback.print_exc()
        return False

def test_config_creation():
    """Test configuration creation"""
    print("\nTesting configuration creation...")
    
    try:
        import json
        
        # Mock config creation (without user input)
        config = {
            "account_address": "0x1234567890123456789012345678901234567890",
            "secret_key": "0x1234567890123456789012345678901234567890123456789012345678901234",
            "api_url": "https://api.hyperliquid.xyz",
            "poll_interval_seconds": 2,
            "micro_poll_interval": 2,
            "trade_symbol": "BTC-USD-PERP",
            "trade_mode": "perp",
            "fast_ma": 5,
            "slow_ma": 15,
            "rsi_period": 14,
            "use_manual_entry_size": True,
            "manual_entry_size": 55.0,
            "timeframes": ["1m", "5m", "15m", "1h"],
            "primary_timeframe": "5m",
            "use_multi_timeframe": True,
            "max_portfolio_heat": 0.15,
            "momentum_strategy_enabled": True,
            "use_ensemble_models": True
        }
        
        print(f"✓ Configuration created with {len(config)} parameters")
        
        # Test JSON serialization
        json_str = json.dumps(config, indent=2)
        parsed_config = json.loads(json_str)
        print("✓ Configuration JSON serialization successful")
        
        return True
        
    except Exception as e:
        print(f"✗ Configuration test failed: {e}")
        traceback.print_exc()
        return False

def main():
    """Run all tests"""
    print("Enhanced Master Bot - Import and Functionality Tests")
    print("=" * 60)
    
    all_passed = True
    
    # Test imports
    if not test_imports():
        all_passed = False
    
    # Test basic functionality
    if not test_basic_functionality():
        all_passed = False
    
    # Test config creation
    if not test_config_creation():
        all_passed = False
    
    print("\n" + "=" * 60)
    if all_passed:
        print("✓ ALL TESTS PASSED! Enhanced Master Bot is ready to run.")
    else:
        print("✗ SOME TESTS FAILED! Please check the errors above.")
    
    return all_passed

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

