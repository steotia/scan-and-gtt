#!/usr/bin/env python
"""
Diagnostic script to check why no spikes are found
"""

import pandas as pd
from datetime import date, timedelta
from pathlib import Path
import glob


def diagnose_data():
    """Check what's happening with the data"""
    
    print("="*60)
    print("DATA DIAGNOSTIC FOR NOVEMBER 4, 2025")
    print("="*60)
    
    # Check if data files exist
    data_dir = Path("data/2025/11")
    print(f"\n1. Checking data directory: {data_dir}")
    
    if data_dir.exists():
        csv_files = list(data_dir.glob("*.csv"))
        print(f"   Found {len(csv_files)} CSV files")
        for f in csv_files[:5]:
            print(f"   - {f.name}")
    else:
        print(f"   ❌ Directory doesn't exist!")
        return
    
    # Load Nov 4 data
    nov4_file = data_dir / "nse_data_2025-11-04.csv"
    if nov4_file.exists():
        print(f"\n2. Loading {nov4_file.name}...")
        df = pd.read_csv(nov4_file)
        print(f"   ✓ Loaded {len(df)} rows")
        print(f"   Columns: {df.columns.tolist()}")
        
        # Check data quality
        print("\n3. Data Quality Check:")
        print(f"   Unique symbols: {df['symbol'].nunique()}")
        print(f"   Delivery qty > 0: {(df['delivery_qty'] > 0).sum()} stocks")
        print(f"   Delivery % > 30: {(df['delivery_percent'] > 30).sum()} stocks")
        print(f"   Volume > 100000: {(df['volume'] > 100000).sum()} stocks")
        
        # Check for actual delivery data
        print("\n4. Sample Data (Top 5 by delivery qty):")
        top_delivery = df.nlargest(5, 'delivery_qty')[['symbol', 'volume', 'delivery_qty', 'delivery_percent', 'close']]
        print(top_delivery.to_string())
        
        # Check historical data availability
        print("\n5. Historical Data Check:")
        historical_dates = []
        for i in range(1, 21):
            check_date = date(2025, 11, 4) - timedelta(days=i)
            check_file = Path(f"data/{check_date.year}/{check_date.month:02d}/nse_data_{check_date}.csv")
            if check_file.exists():
                historical_dates.append(check_date)
        
        print(f"   Found historical data for {len(historical_dates)} days")
        if len(historical_dates) < 10:
            print(f"   ⚠️  WARNING: Only {len(historical_dates)} days of history (need 20)")
            print(f"   This is why spikes can't be calculated!")
        
        # Check spike calculation for a sample stock
        if len(historical_dates) >= 5:
            print("\n6. Sample Spike Calculation (RELIANCE):")
            
            # Get RELIANCE data for Nov 4
            rel_nov4 = df[df['symbol'] == 'RELIANCE']
            if not rel_nov4.empty:
                current_del = rel_nov4.iloc[0]['delivery_qty']
                print(f"   Current delivery: {current_del:,}")
                
                # Get historical average
                historical_deliveries = []
                for hist_date in historical_dates[:20]:
                    hist_file = Path(f"data/{hist_date.year}/{hist_date.month:02d}/nse_data_{hist_date}.csv")
                    if hist_file.exists():
                        hist_df = pd.read_csv(hist_file)
                        rel_hist = hist_df[hist_df['symbol'] == 'RELIANCE']
                        if not rel_hist.empty:
                            historical_deliveries.append(rel_hist.iloc[0]['delivery_qty'])
                
                if historical_deliveries:
                    avg_delivery = sum(historical_deliveries) / len(historical_deliveries)
                    spike_ratio = current_del / avg_delivery if avg_delivery > 0 else 0
                    print(f"   Avg delivery: {avg_delivery:,.0f}")
                    print(f"   Spike ratio: {spike_ratio:.2f}x")
                    print(f"   Would need {5.0:.1f}x for spike (current threshold)")
    else:
        print(f"   ❌ File not found: {nov4_file}")
    
    print("\n" + "="*60)
    print("DIAGNOSIS COMPLETE")
    print("="*60)
    
    print("\nRECOMMENDATIONS:")
    print("1. If insufficient historical data: Fetch more data")
    print("   python main.py --date 2025-10-15 --lookback 20")
    print("   python main.py --date 2025-10-20 --lookback 20")
    print("   python main.py --date 2025-10-25 --lookback 20")
    print("\n2. If no high spike ratios: Lower threshold")
    print("   python main.py --date 2025-11-04 --multiplier 2 --no-smart-volume")
    print("\n3. Try October 31 instead (if it has more history):")
    print("   python main.py --date 2025-10-31 --multiplier 3 --no-smart-volume")


if __name__ == "__main__":
    diagnose_data()