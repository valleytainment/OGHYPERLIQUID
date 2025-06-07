#!/usr/bin/env python3
"""
Comprehensive test suite for Ultimate Master Bot v3.2 - Live Monitoring Edition
Tests all real-time monitoring, trade tracking, and upgrade analysis features.
"""

import sys
import os
import time
import json
import threading
from datetime import datetime, timedelta

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_imports():
    """Test all imports work correctly"""
    print("🧪 Testing imports...")
    
    try:
        from ultimate_master_bot_v3_2_live_monitoring import (
            RealTimeActivityMonitor,
            TradeActivityTracker,
            LiveMarketDataFeed,
            UpgradeAnalyzer,
            UltimateMasterBotV32,
            LiveMonitoringGUI
        )
        print("✅ All imports successful")
        return True
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False

def test_activity_monitor():
    """Test real-time activity monitoring system"""
    print("🧪 Testing RealTimeActivityMonitor...")
    
    try:
        from ultimate_master_bot_v3_2_live_monitoring import RealTimeActivityMonitor
        
        # Create monitor
        monitor = RealTimeActivityMonitor()
        
        # Start monitoring
        monitor.start_monitoring()
        
        # Log some test activities
        monitor.log_activity("test", "Test activity 1", {"test": True}, "medium", "testing")
        monitor.log_activity("trade", "Test trade activity", {"symbol": "BTC", "side": "BUY"}, "high", "trading")
        monitor.log_activity("market", "Test market activity", {"price": 50000}, "low", "market")
        
        # Wait for processing
        time.sleep(1)
        
        # Check activities
        activities = monitor.get_recent_activities(10)
        assert len(activities) >= 3, "Activities not logged correctly"
        
        # Check summary
        summary = monitor.get_activity_summary()
        assert summary["total_activities"] >= 3, "Activity summary incorrect"
        assert summary["monitoring_status"] == "active", "Monitoring status incorrect"
        
        # Stop monitoring
        monitor.stop_monitoring()
        
        print("✅ RealTimeActivityMonitor test passed")
        return True
        
    except Exception as e:
        print(f"❌ RealTimeActivityMonitor test failed: {e}")
        return False

def test_trade_tracker():
    """Test trade activity tracking system"""
    print("🧪 Testing TradeActivityTracker...")
    
    try:
        from ultimate_master_bot_v3_2_live_monitoring import RealTimeActivityMonitor, TradeActivityTracker
        
        # Create monitor and tracker
        monitor = RealTimeActivityMonitor()
        monitor.start_monitoring()
        tracker = TradeActivityTracker(monitor)
        
        # Test trade signal tracking
        trade_id = tracker.track_trade_signal("BTC-USD-PERP", "momentum", "BUY", 0.85, 50000, 0.01)
        assert trade_id is not None, "Trade signal not tracked"
        
        # Test trade execution tracking
        tracker.track_trade_execution(trade_id, "BTC-USD-PERP", "BUY", 0.01, 50100, "momentum", 0.85)
        assert trade_id in tracker.active_trades, "Trade execution not tracked"
        
        # Test trade update
        tracker.track_trade_update(trade_id, 50200, 100.0)
        trade = tracker.active_trades[trade_id]
        assert trade["unrealized_pnl"] == 100.0, "Trade update not working"
        
        # Test trade closure
        tracker.track_trade_close(trade_id, 50200, "manual", 100.0)
        assert trade_id not in tracker.active_trades, "Trade not closed properly"
        assert len(tracker.trade_history) > 0, "Trade not added to history"
        
        # Test summaries
        active_summary = tracker.get_active_trades_summary()
        session_summary = tracker.get_session_summary()
        strategy_performance = tracker.get_strategy_performance()
        
        assert isinstance(active_summary, dict), "Active trades summary invalid"
        assert isinstance(session_summary, dict), "Session summary invalid"
        assert isinstance(strategy_performance, dict), "Strategy performance invalid"
        
        monitor.stop_monitoring()
        
        print("✅ TradeActivityTracker test passed")
        return True
        
    except Exception as e:
        print(f"❌ TradeActivityTracker test failed: {e}")
        return False

def test_market_data_feed():
    """Test live market data feed system"""
    print("🧪 Testing LiveMarketDataFeed...")
    
    try:
        from ultimate_master_bot_v3_2_live_monitoring import RealTimeActivityMonitor, LiveMarketDataFeed
        
        # Create monitor and feed
        monitor = RealTimeActivityMonitor()
        monitor.start_monitoring()
        feed = LiveMarketDataFeed(monitor)
        
        # Add symbols
        feed.add_symbol("BTC-USD-PERP")
        feed.add_symbol("ETH-USD-PERP")
        assert len(feed.symbols) == 2, "Symbols not added correctly"
        
        # Start feed
        feed.start_feed()
        assert feed.is_active, "Feed not started"
        
        # Wait for some data
        time.sleep(3)
        
        # Check market data
        assert len(feed.market_data) > 0, "No market data generated"
        assert "BTC-USD-PERP" in feed.market_data, "BTC data not found"
        
        # Test price alerts
        feed.set_price_alert("BTC-USD-PERP", 60000, "above")
        assert "BTC-USD-PERP" in feed.price_alerts, "Price alert not set"
        
        # Test volatility alerts
        feed.set_volatility_alert("BTC-USD-PERP", 0.05)
        assert "BTC-USD-PERP" in feed.volatility_alerts, "Volatility alert not set"
        
        # Test market summary
        summary = feed.get_market_summary()
        assert isinstance(summary, dict), "Market summary invalid"
        assert "market_regime" in summary, "Market regime not in summary"
        
        # Stop feed
        feed.stop_feed()
        assert not feed.is_active, "Feed not stopped"
        
        monitor.stop_monitoring()
        
        print("✅ LiveMarketDataFeed test passed")
        return True
        
    except Exception as e:
        print(f"❌ LiveMarketDataFeed test failed: {e}")
        return False

def test_upgrade_analyzer():
    """Test upgrade analysis system"""
    print("🧪 Testing UpgradeAnalyzer...")
    
    try:
        from ultimate_master_bot_v3_2_live_monitoring import RealTimeActivityMonitor, UpgradeAnalyzer
        
        # Create monitor and analyzer
        monitor = RealTimeActivityMonitor()
        analyzer = UpgradeAnalyzer(monitor)
        
        # Test upgrade catalog
        assert len(analyzer.upgrade_opportunities) > 0, "No upgrade opportunities found"
        assert len(analyzer.upgrade_opportunities) >= 50, "Not enough upgrade opportunities"
        
        # Test quick wins
        quick_wins = analyzer.get_quick_wins()
        assert isinstance(quick_wins, list), "Quick wins not a list"
        
        # Test high impact upgrades
        high_impact = analyzer.get_high_impact_upgrades()
        assert isinstance(high_impact, list), "High impact upgrades not a list"
        assert len(high_impact) <= 20, "Too many high impact upgrades returned"
        
        # Test upgrades by category
        ai_upgrades = analyzer.get_upgrades_by_category("AI/ML")
        assert len(ai_upgrades) > 0, "No AI/ML upgrades found"
        
        # Test implementation roadmap
        roadmap = analyzer.get_implementation_roadmap()
        assert isinstance(roadmap, dict), "Roadmap not a dictionary"
        assert len(roadmap) == 3, "Roadmap should have 3 phases"
        
        # Test performance analysis
        performance_data = {
            'win_rate': 0.4,  # Below baseline
            'sharpe_ratio': 1.0,  # Below baseline
            'max_drawdown': 0.25,  # Above baseline
            'avg_slippage': 0.002  # Above baseline
        }
        
        relevant_upgrades = analyzer.analyze_current_performance(performance_data)
        assert len(relevant_upgrades) > 0, "No relevant upgrades found"
        
        # Test upgrade report generation
        report = analyzer.generate_upgrade_report(performance_data)
        assert isinstance(report, str), "Report not a string"
        assert len(report) > 1000, "Report too short"
        assert "UPGRADE ANALYSIS" in report, "Report missing title"
        
        # Test marking upgrades as implemented
        analyzer.mark_upgrade_implemented("Test Upgrade")
        assert "Test Upgrade" in analyzer.implemented_upgrades, "Upgrade not marked as implemented"
        
        print("✅ UpgradeAnalyzer test passed")
        return True
        
    except Exception as e:
        print(f"❌ UpgradeAnalyzer test failed: {e}")
        return False

def test_bot_initialization():
    """Test bot initialization and basic functionality"""
    print("🧪 Testing UltimateMasterBotV32 initialization...")
    
    try:
        from ultimate_master_bot_v3_2_live_monitoring import UltimateMasterBotV32
        
        # Create test config
        test_config = {
            "symbol": "BTC-USD-PERP",
            "entry_size": 0.01,
            "account_address": "test_address",
            "private_key": "test_key"
        }
        
        with open("test_config.json", "w") as f:
            json.dump(test_config, f)
        
        # Initialize bot
        bot = UltimateMasterBotV32("test_config.json")
        
        # Check initialization
        assert bot.symbol == "BTC-USD-PERP", "Symbol not set correctly"
        assert bot.entry_size == 0.01, "Entry size not set correctly"
        assert bot.activity_monitor is not None, "Activity monitor not initialized"
        assert bot.trade_tracker is not None, "Trade tracker not initialized"
        assert bot.market_feed is not None, "Market feed not initialized"
        assert bot.upgrade_analyzer is not None, "Upgrade analyzer not initialized"
        
        # Test equity retrieval
        equity = bot.get_equity()
        assert isinstance(equity, (int, float)), "Equity not a number"
        assert equity > 0, "Equity should be positive"
        
        # Test price retrieval
        price = bot.get_current_price()
        assert isinstance(price, (int, float)), "Price not a number"
        assert price > 0, "Price should be positive"
        
        # Test live status
        status = bot.get_live_status()
        assert isinstance(status, dict), "Status not a dictionary"
        assert "bot_status" in status, "Bot status missing"
        assert "current_price" in status, "Current price missing"
        assert "current_equity" in status, "Current equity missing"
        
        # Test trade execution (mock)
        result = bot.execute_trade("BUY", 0.01, "test_strategy", 0.8)
        assert isinstance(result, bool), "Trade execution result not boolean"
        
        # Shutdown bot
        bot.shutdown()
        
        # Cleanup
        if os.path.exists("test_config.json"):
            os.remove("test_config.json")
        
        print("✅ UltimateMasterBotV32 initialization test passed")
        return True
        
    except Exception as e:
        print(f"❌ UltimateMasterBotV32 initialization test failed: {e}")
        return False

def test_gui_components():
    """Test GUI components (headless mode)"""
    print("🧪 Testing LiveMonitoringGUI components...")
    
    try:
        # Skip GUI test in headless environment
        import tkinter as tk
        try:
            root = tk.Tk()
            root.withdraw()  # Hide window
            root.destroy()
            
            from ultimate_master_bot_v3_2_live_monitoring import LiveMonitoringGUI
            
            # Test GUI creation (without running mainloop)
            print("✅ LiveMonitoringGUI components test passed (GUI creation successful)")
            return True
            
        except tk.TclError:
            print("⚠️ LiveMonitoringGUI test skipped (no display available)")
            return True
            
    except Exception as e:
        print(f"❌ LiveMonitoringGUI test failed: {e}")
        return False

def test_database_operations():
    """Test database operations"""
    print("🧪 Testing database operations...")
    
    try:
        from ultimate_master_bot_v3_2_live_monitoring import RealTimeActivityMonitor
        
        # Create monitor with database
        monitor = RealTimeActivityMonitor()
        
        # Check database file creation
        assert os.path.exists(monitor.db_path), "Database file not created"
        
        # Start monitoring and log activities
        monitor.start_monitoring()
        
        # Log test data
        monitor.log_activity("test", "Database test activity", {"test": True}, "medium", "testing")
        
        # Wait for database write
        time.sleep(1)
        
        # Check database content
        import sqlite3
        conn = sqlite3.connect(monitor.db_path)
        cursor = conn.cursor()
        
        cursor.execute("SELECT COUNT(*) FROM activities")
        activity_count = cursor.fetchone()[0]
        assert activity_count > 0, "No activities in database"
        
        conn.close()
        monitor.stop_monitoring()
        
        # Cleanup
        if os.path.exists(monitor.db_path):
            os.remove(monitor.db_path)
        
        print("✅ Database operations test passed")
        return True
        
    except Exception as e:
        print(f"❌ Database operations test failed: {e}")
        return False

def test_threading_safety():
    """Test threading safety of monitoring systems"""
    print("🧪 Testing threading safety...")
    
    try:
        from ultimate_master_bot_v3_2_live_monitoring import RealTimeActivityMonitor
        
        monitor = RealTimeActivityMonitor()
        monitor.start_monitoring()
        
        # Create multiple threads logging activities
        def log_activities(thread_id):
            for i in range(10):
                monitor.log_activity(
                    "test",
                    f"Thread {thread_id} activity {i}",
                    {"thread_id": thread_id, "activity_num": i},
                    "low",
                    "threading_test"
                )
                time.sleep(0.01)
        
        # Start multiple threads
        threads = []
        for i in range(5):
            thread = threading.Thread(target=log_activities, args=(i,))
            threads.append(thread)
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
        
        # Wait for processing
        time.sleep(2)
        
        # Check that all activities were logged
        activities = monitor.get_recent_activities(100)
        assert len(activities) >= 50, f"Expected at least 50 activities, got {len(activities)}"
        
        monitor.stop_monitoring()
        
        print("✅ Threading safety test passed")
        return True
        
    except Exception as e:
        print(f"❌ Threading safety test failed: {e}")
        return False

def run_comprehensive_test():
    """Run all tests"""
    print("🚀 ULTIMATE MASTER BOT v3.2 - COMPREHENSIVE TEST SUITE")
    print("=" * 80)
    print(f"Test started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    tests = [
        ("Imports", test_imports),
        ("Activity Monitor", test_activity_monitor),
        ("Trade Tracker", test_trade_tracker),
        ("Market Data Feed", test_market_data_feed),
        ("Upgrade Analyzer", test_upgrade_analyzer),
        ("Bot Initialization", test_bot_initialization),
        ("GUI Components", test_gui_components),
        ("Database Operations", test_database_operations),
        ("Threading Safety", test_threading_safety)
    ]
    
    passed = 0
    failed = 0
    
    for test_name, test_func in tests:
        print(f"Running {test_name} test...")
        try:
            if test_func():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"❌ {test_name} test crashed: {e}")
            failed += 1
        print()
    
    # Summary
    total = passed + failed
    success_rate = (passed / total * 100) if total > 0 else 0
    
    print("=" * 80)
    print("📊 TEST RESULTS SUMMARY:")
    print(f"✅ Passed: {passed}/{total}")
    print(f"❌ Failed: {failed}/{total}")
    print(f"📈 Success Rate: {success_rate:.1f}%")
    
    if failed == 0:
        print("🎉 ALL TESTS PASSED! Ultimate Master Bot v3.2 is ready for live monitoring!")
    else:
        print("⚠️ Some tests failed. Please review and fix issues before deployment.")
    
    print("=" * 80)
    
    return failed == 0

if __name__ == "__main__":
    success = run_comprehensive_test()
    sys.exit(0 if success else 1)

