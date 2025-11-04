"""
Test script to verify NSE Delivery Tracker components
"""

import sys
import os
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

def test_imports():
    """Test if all modules can be imported"""
    print("Testing imports...")
    
    try:
        # Test interfaces
        from src.interfaces import IDataFetcher, IDataRepository, IAnalysisEngine, DeliverySpike
        print("✓ Interfaces imported successfully")
        
        # Test data modules
        from src.data.fetcher import NSEDataFetcher
        from src.data.repository import CSVDataRepository, SQLiteDataRepository
        print("✓ Data modules imported successfully")
        
        # Test analysis modules
        from src.analysis.engine import DeliveryAnalysisEngine, AdvancedSpikeDetector
        from src.analysis.filters import VolumeFilter, IndexFilter
        print("✓ Analysis modules imported successfully")
        
        # Test reporting modules
        from src.reporting.excel_generator import ExcelReportGenerator
        print("✓ Reporting modules imported successfully")
        
        # Test configuration
        from src.config.manager import ConfigurationManager, AppConfig
        print("✓ Configuration modules imported successfully")
        
        # Test main application
        from main import NSEDeliveryTracker, create_app
        print("✓ Main application imported successfully")
        
        return True
        
    except ImportError as e:
        print(f"✗ Import error: {e}")
        return False


def test_configuration():
    """Test configuration loading"""
    print("\nTesting configuration...")
    
    try:
        from src.config.manager import ConfigurationManager
        
        # Test loading config
        config_manager = ConfigurationManager("config.yaml")
        config = config_manager.get_config()
        
        # Verify key settings
        assert config.lookback_days > 0, "Invalid lookback_days"
        assert config.spike_multiplier > 0, "Invalid spike_multiplier"
        print(f"✓ Configuration loaded: lookback={config.lookback_days}, multiplier={config.spike_multiplier}")
        
        return True
        
    except Exception as e:
        print(f"✗ Configuration error: {e}")
        return False


def test_directory_structure():
    """Test if required directories exist or can be created"""
    print("\nTesting directory structure...")
    
    directories = ["data", "reports", "logs"]
    
    for dir_name in directories:
        dir_path = Path(dir_name)
        if not dir_path.exists():
            try:
                dir_path.mkdir(parents=True, exist_ok=True)
                print(f"✓ Created directory: {dir_name}/")
            except Exception as e:
                print(f"✗ Could not create {dir_name}/: {e}")
                return False
        else:
            print(f"✓ Directory exists: {dir_name}/")
    
    return True


def test_sample_analysis():
    """Test creating a simple analysis pipeline"""
    print("\nTesting analysis pipeline creation...")
    
    try:
        from main import create_app
        import pandas as pd
        from datetime import date
        
        # Create application
        app = create_app("config.yaml")
        print("✓ Application created successfully")
        
        # Create sample data
        sample_data = pd.DataFrame({
            'symbol': ['TEST1', 'TEST2', 'TEST3'] * 5,
            'date': [date(2024, 1, i) for i in range(1, 6)] * 3,
            'open': [100] * 15,
            'high': [105] * 15,
            'low': [95] * 15,
            'close': [102] * 15,
            'volume': [1000000] * 15,
            'delivery_qty': [100000, 200000, 500000] * 5,  # TEST3 has 5x delivery
            'delivery_percent': [10, 20, 50] * 5
        })
        
        # Test spike detection
        spikes = app.analysis_engine.detect_spikes(
            sample_data, 
            lookback_days=3, 
            spike_multiplier=2.0
        )
        
        print(f"✓ Analysis engine works: Found {len(spikes)} spikes in sample data")
        
        return True
        
    except Exception as e:
        print(f"✗ Analysis pipeline error: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests"""
    print("="*50)
    print("NSE Delivery Tracker - System Test")
    print("="*50)
    
    tests = [
        ("Import Test", test_imports),
        ("Configuration Test", test_configuration),
        ("Directory Test", test_directory_structure),
        ("Analysis Pipeline Test", test_sample_analysis)
    ]
    
    results = []
    for test_name, test_func in tests:
        result = test_func()
        results.append((test_name, result))
    
    print("\n" + "="*50)
    print("Test Results Summary")
    print("="*50)
    
    for test_name, result in results:
        status = "PASSED" if result else "FAILED"
        symbol = "✓" if result else "✗"
        print(f"{symbol} {test_name}: {status}")
    
    all_passed = all(r[1] for r in results)
    
    if all_passed:
        print("\n🎉 All tests passed! The system is ready to use.")
        print("\nNext steps:")
        print("1. Activate virtual environment: source venv/bin/activate")
        print("2. Run the tracker: python main.py")
    else:
        print("\n⚠️ Some tests failed. Please check the errors above.")
    
    return all_passed


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
