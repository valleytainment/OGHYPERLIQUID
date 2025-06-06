#!/usr/bin/env python3
"""
Quick startup test for the Enhanced Master Bot
"""

import sys
import os

# Add current directory to path
sys.path.insert(0, '/home/ubuntu')

def test_bot_startup():
    """Test that the bot can initialize without errors"""
    try:
        print("Testing Enhanced Master Bot startup...")
        
        # Import the bot (this will test all imports and basic initialization)
        from enhanced_master_bot_tested import EnhancedUltimateMasterBot, CONFIG
        import queue
        
        # Create a test log queue
        log_queue = queue.Queue()
        
        # Initialize the bot
        bot = EnhancedUltimateMasterBot(CONFIG, log_queue)
        
        print("✓ Bot initialization successful")
        print(f"✓ Symbol: {bot.symbol}")
        print(f"✓ Trade mode: {bot.trade_mode}")
        print(f"✓ Timeframes: {bot.timeframes}")
        print(f"✓ Primary timeframe: {bot.primary_timeframe}")
        print(f"✓ ML system enabled: {bot.model_system is not None}")
        
        # Test basic functionality
        equity = bot.get_equity()
        print(f"✓ Current equity: ${equity:.2f}")
        
        positions = bot.get_user_positions()
        print(f"✓ Current positions: {len(positions)}")
        
        print("\n✅ ALL STARTUP TESTS PASSED!")
        print("Enhanced Master Bot is ready for deployment!")
        
        return True
        
    except Exception as e:
        print(f"❌ Startup test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_bot_startup()
    sys.exit(0 if success else 1)

