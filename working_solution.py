# working_solution_fixed.py
from jugaad_data.nse import full_bhavcopy_save
from datetime import date, timedelta
import pandas as pd
import os
import glob

class NSEDeliveryAnalyzer:
    """Analyzer using jugaad-data's full_bhavcopy with delivery data"""
    
    def __init__(self, data_dir="data"):
        self.data_dir = data_dir
        os.makedirs(data_dir, exist_ok=True)
    
    def fetch_date(self, fetch_date):
        """Fetch data with delivery info for a date"""
        
        # Check cache
        cache_file = f"{self.data_dir}/nse_{fetch_date.strftime('%Y%m%d')}.csv"
        if os.path.exists(cache_file):
            print(f"Using cached data for {fetch_date}")
            return pd.read_csv(cache_file)
        
        print(f"Fetching full bhavcopy for {fetch_date}...")
        
        # Download to temp location
        temp_dir = f"{self.data_dir}/temp"
        os.makedirs(temp_dir, exist_ok=True)
        
        try:
            # Download full bhavcopy with delivery data
            full_bhavcopy_save(fetch_date, temp_dir)
            
            # Find the downloaded file
            files = glob.glob(f"{temp_dir}/*.csv")
            if files:
                # Read and process
                df = pd.read_csv(files[0])
                
                # Clean column names (remove leading spaces)
                df.columns = df.columns.str.strip()
                
                # FIX: Clean numeric columns that might have spaces
                numeric_columns = ['PREV_CLOSE', 'OPEN_PRICE', 'HIGH_PRICE', 'LOW_PRICE', 
                                 'LAST_PRICE', 'CLOSE_PRICE', 'AVG_PRICE', 'TTL_TRD_QNTY',
                                 'TURNOVER_LACS', 'NO_OF_TRADES', 'DELIV_QTY', 'DELIV_PER']
                
                for col in numeric_columns:
                    if col in df.columns:
                        # Convert to string, remove spaces, then to numeric
                        df[col] = df[col].astype(str).str.replace(' ', '').str.replace(',', '')
                        df[col] = pd.to_numeric(df[col], errors='coerce')
                
                # Rename to our standard format
                df = df.rename(columns={
                    'SYMBOL': 'symbol',
                    'OPEN_PRICE': 'open',
                    'HIGH_PRICE': 'high',
                    'LOW_PRICE': 'low',
                    'CLOSE_PRICE': 'close',
                    'TTL_TRD_QNTY': 'volume',
                    'DELIV_QTY': 'delivery_qty',
                    'DELIV_PER': 'delivery_percent',
                    'AVG_PRICE': 'vwap'
                })
                
                # Filter out rows with invalid data
                df = df[df['delivery_qty'].notna() & (df['delivery_qty'] > 0)]
                
                # Add date column
                df['date'] = fetch_date
                
                # Save to cache
                df.to_csv(cache_file, index=False)
                
                # Clean up temp file
                os.remove(files[0])
                
                print(f"✅ Fetched {len(df)} stocks with valid delivery data")
                return df
                
        except Exception as e:
            print(f"Error fetching {fetch_date}: {e}")
            return pd.DataFrame()
    
    def analyze_spikes(self, target_date, lookback_days=20, spike_multiplier=5.0):
        """Analyze delivery spikes"""
        
        # Fetch historical data
        all_data = []
        current_date = target_date
        
        print(f"\nFetching historical data...")
        for i in range(lookback_days + 1):
            if current_date.weekday() < 5:  # Skip weekends
                df = self.fetch_date(current_date)
                if not df.empty:
                    all_data.append(df)
            current_date -= timedelta(days=1)
        
        if not all_data:
            print("No data available")
            return pd.DataFrame()
        
        # Combine all data
        combined = pd.concat(all_data, ignore_index=True)
        
        # Get latest day data
        latest_data = combined[combined['date'] == target_date].copy()
        
        if latest_data.empty:
            print(f"No data for target date {target_date}")
            return pd.DataFrame()
        
        print(f"Analyzing {len(latest_data)} stocks for spikes...")
        
        # Calculate averages for each stock
        spikes = []
        
        for symbol in latest_data['symbol'].unique():
            try:
                # Get historical data for this stock
                stock_history = combined[
                    (combined['symbol'] == symbol) & 
                    (combined['date'] < target_date)
                ]
                
                if len(stock_history) > 0:
                    # Calculate average delivery (ensure numeric)
                    avg_delivery = stock_history['delivery_qty'].mean()
                    
                    # Skip if average is invalid or zero
                    if pd.isna(avg_delivery) or avg_delivery <= 0:
                        continue
                    
                    # Get current delivery
                    current = latest_data[latest_data['symbol'] == symbol].iloc[0]
                    current_delivery = current['delivery_qty']
                    
                    # Skip if current delivery is invalid
                    if pd.isna(current_delivery) or current_delivery <= 0:
                        continue
                    
                    # Calculate spike ratio
                    spike_ratio = current_delivery / avg_delivery
                    
                    if spike_ratio >= spike_multiplier:
                        spikes.append({
                            'symbol': symbol,
                            'spike_ratio': spike_ratio,
                            'current_delivery': int(current_delivery),
                            'avg_delivery': int(avg_delivery),
                            'delivery_percent': current['delivery_percent'],
                            'close': current['close'],
                            'volume': int(current['volume'])
                        })
                        
            except Exception as e:
                # Skip stocks with data issues
                continue
        
        # Convert to DataFrame and sort
        spikes_df = pd.DataFrame(spikes)
        if not spikes_df.empty:
            spikes_df = spikes_df.sort_values('spike_ratio', ascending=False)
        
        return spikes_df


# Test it
if __name__ == "__main__":
    analyzer = NSEDeliveryAnalyzer()
    
    # Analyze October 31, 2025
    target_date = date(2025, 10, 31)
    
    print(f"\n{'='*60}")
    print(f"Analyzing delivery spikes for {target_date}")
    print(f"{'='*60}")
    
    spikes = analyzer.analyze_spikes(
        target_date=target_date,
        lookback_days=20,
        spike_multiplier=3.0  # 3x spikes
    )
    
    if not spikes.empty:
        print(f"\n✅ Found {len(spikes)} delivery spikes!\n")
        print("Top 10 Delivery Spikes:")
        print("-" * 60)
        
        for idx, row in spikes.head(10).iterrows():
            print(f"{row['symbol']:12} | "
                  f"Spike: {row['spike_ratio']:5.1f}x | "
                  f"Delivery: {row['current_delivery']:10,} | "
                  f"Avg: {row['avg_delivery']:10,} | "
                  f"Del%: {row['delivery_percent']:5.2f}%")
    else:
        print("No spikes found. Try lowering the multiplier.")
        print("\nTroubleshooting:")
        print("1. Check if data directory has cached files")
        print("2. Try a lower multiplier (e.g., 2.0)")
        print("3. Try a different date")