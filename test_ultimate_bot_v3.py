#!/usr/bin/env python3
"""
Comprehensive test suite for Ultimate Master Bot v3.0
Tests all functionality including startup, API connections, and GUI
"""

import sys
import os
import time
import threading
import queue
import json
from unittest.mock import Mock, patch

def test_imports():
    """Test all imports work correctly"""
    print("Testing imports...")
    
    try:
        # Test basic imports
        import numpy as np
        import pandas as pd
        import torch
        import sklearn
        import ta
        import matplotlib.pyplot as plt
        import tkinter as tk
        from tkinter import ttk
        
        print("✅ All basic imports successful")
        
        # Test the ultimate bot import
        sys.path.append('/home/ubuntu')
        from ultimate_master_bot_v3 import (
            UltimateEnhancedMasterBot,
            UltimateComprehensiveGUI,
            UltimateFeatureEngineer,
            UltimateStrategyEngine,
            CONFIG
        )
        
        print("✅ Ultimate bot imports successful")
        return True
        
    except Exception as e:
        print(f"❌ Import error: {e}")
        return False

def test_config_loading():
    """Test configuration loading"""
    print("\nTesting configuration...")
    
    try:
        from ultimate_master_bot_v3 import CONFIG
        
        # Check required config keys
        required_keys = [
            "account_address", "secret_key", "trade_symbol", 
            "api_url", "manual_entry_size"
        ]
        
        for key in required_keys:
            if key not in CONFIG:
                print(f"❌ Missing config key: {key}")
                return False
        
        print("✅ Configuration loaded successfully")
        print(f"   - Symbol: {CONFIG['trade_symbol']}")
        print(f"   - API URL: {CONFIG['api_url']}")
        print(f"   - Entry Size: {CONFIG['manual_entry_size']}")
        
        return True
        
    except Exception as e:
        print(f"❌ Config error: {e}")
        return False

def test_bot_initialization():
    """Test bot initialization"""
    print("\nTesting bot initialization...")
    
    try:
        from ultimate_master_bot_v3 import UltimateEnhancedMasterBot, CONFIG
        
        # Create log queue
        log_queue = queue.Queue()
        
        # Initialize bot
        bot = UltimateEnhancedMasterBot(CONFIG, log_queue)
        
        print("✅ Bot initialized successfully")
        print(f"   - Symbol: {bot.symbol}")
        print(f"   - Mode: {bot.trade_mode}")
        print(f"   - Device: {bot.device}")
        print(f"   - Model loaded: {hasattr(bot, 'model')}")
        print(f"   - Strategy engine: {hasattr(bot, 'strategy_engine')}")
        print(f"   - Feature engineer: {hasattr(bot, 'feature_engineer')}")
        
        return True
        
    except Exception as e:
        print(f"❌ Bot initialization error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_api_connection():
    """Test API connection"""
    print("\nTesting API connection...")
    
    try:
        from ultimate_master_bot_v3 import UltimateEnhancedMasterBot, CONFIG
        
        log_queue = queue.Queue()
        bot = UltimateEnhancedMasterBot(CONFIG, log_queue)
        
        # Test equity retrieval (should work with mock data)
        equity = bot.get_equity()
        print(f"✅ Equity retrieval successful: ${equity:.2f}")
        
        # Test price/volume fetching
        pv = bot.fetch_price_volume()
        if pv and pv.get("price", 0) > 0:
            print(f"✅ Price/volume fetch successful: ${pv['price']:.2f}, Vol: {pv['volume']:.0f}")
        else:
            print("❌ Price/volume fetch failed")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ API connection error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_feature_engineering():
    """Test feature engineering"""
    print("\nTesting feature engineering...")
    
    try:
        from ultimate_master_bot_v3 import UltimateFeatureEngineer, CONFIG
        import pandas as pd
        import numpy as np
        
        # Create feature engineer
        fe = UltimateFeatureEngineer(CONFIG)
        
        # Create sample data
        dates = pd.date_range('2024-01-01', periods=100, freq='5min')
        prices = 50000 + np.cumsum(np.random.randn(100) * 10)
        volumes = np.random.uniform(100, 1000, 100)
        
        df = pd.DataFrame({
            'time': dates,
            'price': prices,
            'volume': volumes,
            'high': prices * 1.001,
            'low': prices * 0.999,
            'close': prices
        })
        
        # Test feature engineering
        enhanced_df = fe.engineer_features(df)
        
        print(f"✅ Feature engineering successful")
        print(f"   - Original columns: {len(df.columns)}")
        print(f"   - Enhanced columns: {len(enhanced_df.columns)}")
        print(f"   - New features: {len(enhanced_df.columns) - len(df.columns)}")
        
        return True
        
    except Exception as e:
        print(f"❌ Feature engineering error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_strategy_engine():
    """Test strategy engine"""
    print("\nTesting strategy engine...")
    
    try:
        from ultimate_master_bot_v3 import UltimateStrategyEngine, MarketRegime, CONFIG
        import pandas as pd
        import numpy as np
        
        # Create strategy engine
        engine = UltimateStrategyEngine(CONFIG)
        
        # Create sample market data
        dates = pd.date_range('2024-01-01', periods=100, freq='5min')
        prices = 50000 + np.cumsum(np.random.randn(100) * 10)
        
        df = pd.DataFrame({
            'time': dates,
            'price': prices,
            'volume': np.random.uniform(100, 1000, 100),
            'rsi_14': np.random.uniform(30, 70, 100),
            'macd_histogram': np.random.uniform(-10, 10, 100),
            'sma_5': prices * 0.999,
            'sma_20': prices * 0.998,
            'bb_upper': prices * 1.02,
            'bb_lower': prices * 0.98,
            'bb_position': np.random.uniform(0, 1, 100),
            'volume_ratio_20': np.random.uniform(0.8, 1.5, 100),
            'atr': np.random.uniform(0.003, 0.008, 100),
            'price_zscore': np.random.uniform(-2, 2, 100),
            'spread_proxy': np.random.uniform(0.0005, 0.002, 100)
        })
        
        # Create market regime
        regime = MarketRegime(
            trend_strength=0.6,
            volatility_level="medium",
            volume_profile="neutral",
            momentum_state="bullish",
            mean_reversion_signal=0.1,
            regime_confidence=0.7,
            microstructure_signal=0.2
        )
        
        # Test signal generation
        signals = engine.generate_signals(df, regime)
        
        print(f"✅ Strategy engine successful")
        print(f"   - Active strategies: {len(engine.strategies)}")
        print(f"   - Generated signals: {len(signals)}")
        
        for signal in signals[:3]:  # Show first 3 signals
            print(f"   - {signal.strategy_source}: {signal.direction} (conf: {signal.confidence:.2f})")
        
        return True
        
    except Exception as e:
        print(f"❌ Strategy engine error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_ml_models():
    """Test ML models"""
    print("\nTesting ML models...")
    
    try:
        from ultimate_master_bot_v3 import (
            UltimateTransformerModel, UltimateLSTMModel, UltimateCNNModel
        )
        import torch
        
        # Test parameters
        input_size_per_bar = 12
        lookback_bars = 30
        hidden_size = 64
        batch_size = 4
        
        # Create sample input
        sample_input = torch.randn(batch_size, lookback_bars * input_size_per_bar)
        
        # Test Transformer model
        transformer = UltimateTransformerModel(input_size_per_bar, lookback_bars, hidden_size)
        reg_out, cls_out, conf_out = transformer(sample_input)
        print(f"✅ Transformer model: reg={reg_out.shape}, cls={cls_out.shape}, conf={conf_out.shape}")
        
        # Test LSTM model
        lstm = UltimateLSTMModel(input_size_per_bar, lookback_bars, hidden_size)
        reg_out, cls_out, conf_out = lstm(sample_input)
        print(f"✅ LSTM model: reg={reg_out.shape}, cls={cls_out.shape}, conf={conf_out.shape}")
        
        # Test CNN model
        cnn = UltimateCNNModel(input_size_per_bar, lookback_bars)
        reg_out, cls_out, conf_out = cnn(sample_input)
        print(f"✅ CNN model: reg={reg_out.shape}, cls={cls_out.shape}, conf={conf_out.shape}")
        
        return True
        
    except Exception as e:
        print(f"❌ ML models error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_gui_components():
    """Test GUI components (skip in headless environments)"""
    print("\nTesting GUI components...")
    
    try:
        import os
        if os.environ.get('DISPLAY') is None:
            print("⚠️  Skipping GUI test in headless environment")
            return True
        
        import tkinter as tk
        from ultimate_master_bot_v3 import UltimateComprehensiveGUI
        
        # Create root window (but don't show it)
        root = tk.Tk()
        root.withdraw()  # Hide the window
        
        # Test GUI initialization
        gui = UltimateComprehensiveGUI(root)
        
        print("✅ GUI initialization successful")
        print(f"   - Main frame created: {hasattr(gui, 'main_frame')}")
        print(f"   - Canvas created: {hasattr(gui, 'canvas')}")
        print(f"   - Notebook created: {hasattr(gui, 'notebook')}")
        print(f"   - Bot instance: {hasattr(gui, 'bot')}")
        print(f"   - Log queue: {hasattr(gui, 'log_queue')}")
        
        # Test some GUI variables
        print(f"   - Symbol var: {gui.symbol_var.get()}")
        print(f"   - Manual size var: {gui.manual_size_var.get()}")
        
        # Clean up
        root.destroy()
        
        return True
        
    except Exception as e:
        print(f"❌ GUI components error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_data_processing():
    """Test data processing and indicators"""
    print("\nTesting data processing...")
    
    try:
        from ultimate_master_bot_v3 import UltimateEnhancedMasterBot, CONFIG
        import pandas as pd
        import numpy as np
        
        log_queue = queue.Queue()
        bot = UltimateEnhancedMasterBot(CONFIG, log_queue)
        
        # Create sample historical data
        dates = pd.date_range('2024-01-01', periods=100, freq='5min')
        prices = 50000 + np.cumsum(np.random.randn(100) * 10)
        volumes = np.random.uniform(100, 1000, 100)
        
        # Build historical data
        for i, (date, price, volume) in enumerate(zip(dates, prices, volumes)):
            new_row = pd.DataFrame([[date.isoformat(), price, volume] + [np.nan]*11], 
                                 columns=bot.hist_data.columns)
            bot.hist_data = pd.concat([bot.hist_data, new_row], ignore_index=True)
        
        # Test indicator computation
        row = bot.compute_indicators(bot.hist_data)
        
        if row is not None:
            print("✅ Data processing successful")
            print(f"   - Historical data rows: {len(bot.hist_data)}")
            print(f"   - Latest price: ${row['price']:.2f}")
            print(f"   - RSI: {row.get('rsi', 'N/A')}")
            print(f"   - Fast MA: ${row.get('fast_ma', 0):.2f}")
            print(f"   - Slow MA: ${row.get('slow_ma', 0):.2f}")
            
            # Test feature building
            if len(bot.hist_data) >= bot.lookback_bars:
                features = bot.build_input_features(bot.hist_data.iloc[-bot.lookback_bars:])
                expected_features = bot.lookback_bars * bot.features_per_bar
                
                if len(features) == expected_features:
                    print(f"✅ Feature building successful: {len(features)} features")
                else:
                    print(f"❌ Feature building failed: got {len(features)}, expected {expected_features}")
                    return False
            
            return True
        else:
            print("❌ Indicator computation failed")
            return False
        
    except Exception as e:
        print(f"❌ Data processing error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_risk_management():
    """Test risk management system"""
    print("\nTesting risk management...")
    
    try:
        from ultimate_master_bot_v3 import UltimateEnhancedMasterBot, CONFIG
        import numpy as np
        
        log_queue = queue.Queue()
        bot = UltimateEnhancedMasterBot(CONFIG, log_queue)
        
        # Add some sample trade PnLs
        sample_pnls = np.random.normal(0.001, 0.01, 50)  # 50 trades with small positive bias
        bot.trade_pnls.extend(sample_pnls)
        
        # Test risk metrics calculation
        risk_metrics = bot.calculate_risk_metrics()
        
        print("✅ Risk management successful")
        print(f"   - Portfolio heat: {risk_metrics.portfolio_heat:.3f}")
        print(f"   - Max drawdown: {risk_metrics.max_drawdown:.3f}")
        print(f"   - Sharpe ratio: {risk_metrics.sharpe_ratio:.3f}")
        print(f"   - Sortino ratio: {risk_metrics.sortino_ratio:.3f}")
        print(f"   - VaR 95%: {risk_metrics.var_95:.3f}")
        print(f"   - Kelly fraction: {risk_metrics.kelly_fraction:.3f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Risk management error: {e}")
        import traceback
        traceback.print_exc()
        return False

def run_comprehensive_test():
    """Run all tests"""
    print("🚀 ULTIMATE MASTER BOT v3.0 - COMPREHENSIVE TEST SUITE")
    print("=" * 60)
    
    tests = [
        test_imports,
        test_config_loading,
        test_bot_initialization,
        test_api_connection,
        test_feature_engineering,
        test_strategy_engine,
        test_ml_models,
        test_gui_components,
        test_data_processing,
        test_risk_management
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            if test():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"❌ Test {test.__name__} crashed: {e}")
            failed += 1
        
        print("-" * 40)
    
    print(f"\n📊 TEST RESULTS:")
    print(f"✅ Passed: {passed}")
    print(f"❌ Failed: {failed}")
    print(f"📈 Success Rate: {(passed/(passed+failed)*100):.1f}%")
    
    if failed == 0:
        print("\n🎉 ALL TESTS PASSED! Ultimate Master Bot v3.0 is ready for deployment!")
        return True
    else:
        print(f"\n⚠️  {failed} tests failed. Please review the errors above.")
        return False

if __name__ == "__main__":
    success = run_comprehensive_test()
    sys.exit(0 if success else 1)

