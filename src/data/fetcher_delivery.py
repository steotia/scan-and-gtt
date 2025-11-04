"""
NSE Data Fetcher with Delivery Data
Uses jugaad-data's full_bhavcopy for complete delivery information
"""

from jugaad_data.nse import full_bhavcopy_save
from datetime import date, timedelta
from pathlib import Path
from typing import Optional
import pandas as pd
import glob
import os
from loguru import logger


class DeliveryDataFetcher:
    """
    NSE Data Fetcher that includes delivery data
    Uses jugaad-data's full_bhavcopy_save
    """
    
    def __init__(self, cache_dir: str = "data"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.temp_dir = self.cache_dir / "temp"
        self.temp_dir.mkdir(exist_ok=True)
        
    async def fetch_daily_data(self, fetch_date: date) -> pd.DataFrame:
        """Fetch daily data with delivery information"""
        
        # Check cache first
        cache_file = self.cache_dir / f"{fetch_date.year}" / f"{fetch_date.month:02d}" / f"nse_data_{fetch_date.isoformat()}.csv"
        
        if cache_file.exists():
            logger.info(f"Loading cached data for {fetch_date}")
            return pd.read_csv(cache_file)
        
        # Skip weekends
        if fetch_date.weekday() >= 5:
            logger.warning(f"{fetch_date} is a weekend")
            return pd.DataFrame()
        
        try:
            logger.info(f"Fetching full bhavcopy for {fetch_date}")
            
            # Download full bhavcopy with delivery data
            full_bhavcopy_save(fetch_date, str(self.temp_dir))
            
            # Find downloaded file
            pattern = f"*{fetch_date.strftime('%d%b%Y')}*.csv"
            files = list(self.temp_dir.glob(pattern))
            
            if not files:
                logger.error(f"No file downloaded for {fetch_date}")
                return pd.DataFrame()
            
            # Read and process
            df = pd.read_csv(files[0])
            
            # Clean column names
            df.columns = df.columns.str.strip()
            
            # Fix numeric columns with spaces
            numeric_cols = ['PREV_CLOSE', 'OPEN_PRICE', 'HIGH_PRICE', 'LOW_PRICE',
                          'CLOSE_PRICE', 'AVG_PRICE', 'TTL_TRD_QNTY', 'NO_OF_TRADES',
                          'DELIV_QTY', 'DELIV_PER', 'TURNOVER_LACS']
            
            for col in numeric_cols:
                if col in df.columns:
                    df[col] = df[col].astype(str).str.replace(' ', '').str.replace(',', '')
                    df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # Rename columns to standard format
            df = df.rename(columns={
                'SYMBOL': 'symbol',
                'SERIES': 'series',
                'OPEN_PRICE': 'open',
                'HIGH_PRICE': 'high',
                'LOW_PRICE': 'low',
                'CLOSE_PRICE': 'close',
                'PREV_CLOSE': 'prev_close',
                'TTL_TRD_QNTY': 'volume',
                'DELIV_QTY': 'delivery_qty',
                'DELIV_PER': 'delivery_percent',
                'AVG_PRICE': 'vwap',
                'NO_OF_TRADES': 'trades',
                'TURNOVER_LACS': 'turnover'
            })
            
            # Filter only EQ series and valid data
            if 'series' in df.columns:
                df = df[df['series'].str.strip() == 'EQ']
            df = df[df['delivery_qty'].notna() & (df['delivery_qty'] > 0)]
            
            # Add date
            df['date'] = fetch_date
            
            # Save to cache
            cache_file.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(cache_file, index=False)
            
            # Cleanup temp
            files[0].unlink()
            
            logger.success(f"Fetched {len(df)} stocks with delivery data for {fetch_date}")
            return df
            
        except Exception as e:
            logger.error(f"Error fetching {fetch_date}: {e}")
            return pd.DataFrame()
    
    async def fetch_historical_data(self, start_date: date, end_date: date) -> pd.DataFrame:
        """Fetch historical data for date range"""
        
        all_data = []
        current = start_date
        
        while current <= end_date:
            if current.weekday() < 5:
                df = await self.fetch_daily_data(current)
                if not df.empty:
                    all_data.append(df)
            current += timedelta(days=1)
        
        if all_data:
            return pd.concat(all_data, ignore_index=True)
        return pd.DataFrame()
    
    def validate_data(self, data: pd.DataFrame) -> bool:
        """Validate fetched data"""
        
        if data.empty:
            return False
        
        required_cols = ['symbol', 'open', 'high', 'low', 'close', 
                        'volume', 'delivery_qty', 'delivery_percent']
        
        if not all(col in data.columns for col in required_cols):
            logger.error(f"Missing required columns")
            return False
        
        # Check data quality
        if data['delivery_qty'].sum() == 0:
            logger.error("No delivery data found")
            return False
        
        return True
