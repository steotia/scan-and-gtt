"""
Data Fetcher Implementation
Responsible for fetching NSE data
"""

import pandas as pd
import requests
from datetime import date, timedelta
from pathlib import Path
from typing import Optional, Dict, Any
from enum import Enum
import asyncio
import aiohttp
from loguru import logger
import zipfile
import io

from src.interfaces import IDataFetcher, DataSource


class NSEDataFetcher(IDataFetcher):
    """
    NSE Data Fetcher implementation
    Note: This is the base implementation. Use DeliveryDataFetcher for production
    """
    
    def __init__(self, cache_dir: Optional[Path] = None):
        """
        Initialize NSE data fetcher
        
        Args:
            cache_dir: Optional cache directory
        """
        self.base_url = "https://archives.nseindia.com"
        self.cache_dir = cache_dir or Path("data/cache")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Session for connection pooling
        self.session = None
        
        logger.info("NSE Data Fetcher initialized")
    
    async def fetch_daily_data(self, fetch_date: date) -> pd.DataFrame:
        """
        Fetch daily bhavcopy data
        
        Args:
            fetch_date: Date to fetch data for
            
        Returns:
            DataFrame with daily data
        """
        # Skip weekends
        if fetch_date.weekday() >= 5:
            logger.info(f"Skipping weekend date: {fetch_date}")
            return pd.DataFrame()
        
        # Check cache first
        cache_file = self.cache_dir / f"nse_{fetch_date.isoformat()}.csv"
        if cache_file.exists():
            logger.info(f"Loading from cache: {cache_file}")
            return pd.read_csv(cache_file)
        
        try:
            # Fetch from NSE
            data = await self._fetch_bhavcopy(fetch_date)
            
            if not data.empty:
                # Cache the data
                data.to_csv(cache_file, index=False)
                logger.info(f"Data cached: {cache_file}")
            
            return data
            
        except Exception as e:
            logger.error(f"Error fetching data for {fetch_date}: {e}")
            return pd.DataFrame()
    
    async def fetch_historical_data(self, start_date: date, end_date: date) -> pd.DataFrame:
        """
        Fetch historical data for a date range
        
        Args:
            start_date: Start date
            end_date: End date
            
        Returns:
            Combined DataFrame with all data
        """
        all_data = []
        current_date = start_date
        
        async with aiohttp.ClientSession() as session:
            self.session = session
            
            tasks = []
            while current_date <= end_date:
                if current_date.weekday() < 5:  # Skip weekends
                    tasks.append(self.fetch_daily_data(current_date))
                current_date += timedelta(days=1)
            
            # Fetch all dates concurrently
            results = await asyncio.gather(*tasks)
            
            for result in results:
                if not result.empty:
                    all_data.append(result)
        
        if all_data:
            combined = pd.concat(all_data, ignore_index=True)
            logger.info(f"Fetched {len(combined)} records from {start_date} to {end_date}")
            return combined
        
        return pd.DataFrame()
    
    def validate_data(self, data: pd.DataFrame) -> bool:
        """
        Validate fetched data
        
        Args:
            data: DataFrame to validate
            
        Returns:
            True if valid
        """
        if data.empty:
            return False
        
        required_columns = ['symbol', 'open', 'high', 'low', 'close', 'volume']
        
        for col in required_columns:
            if col not in data.columns:
                logger.error(f"Missing required column: {col}")
                return False
        
        # Check for valid values
        if data['volume'].sum() == 0:
            logger.error("No volume data found")
            return False
        
        return True
    
    async def _fetch_bhavcopy(self, fetch_date: date) -> pd.DataFrame:
        """
        Fetch bhavcopy from NSE website
        
        Args:
            fetch_date: Date to fetch
            
        Returns:
            DataFrame with bhavcopy data
        """
        # Format date for NSE URL
        date_str = fetch_date.strftime("%d%b%Y").upper()
        month_str = fetch_date.strftime("%b").upper()
        year = fetch_date.year
        
        # Construct URL
        url = f"{self.base_url}/content/historical/EQUITIES/{year}/{month_str}/cm{date_str}bhav.csv.zip"
        
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        
        try:
            if self.session:
                async with self.session.get(url, headers=headers) as response:
                    if response.status == 200:
                        content = await response.read()
                        return self._process_zip_content(content)
            else:
                # Fallback to requests
                response = requests.get(url, headers=headers)
                if response.status_code == 200:
                    return self._process_zip_content(response.content)
                    
        except Exception as e:
            logger.error(f"Error downloading bhavcopy: {e}")
        
        return pd.DataFrame()
    
    def _process_zip_content(self, content: bytes) -> pd.DataFrame:
        """Process zip file content and extract CSV"""
        try:
            with zipfile.ZipFile(io.BytesIO(content)) as z:
                csv_name = z.namelist()[0]
                with z.open(csv_name) as f:
                    df = pd.read_csv(f)
                    
                    # Process columns
                    df.columns = df.columns.str.strip()
                    
                    # Rename columns to standard format
                    column_mapping = {
                        'SYMBOL': 'symbol',
                        'OPEN': 'open',
                        'HIGH': 'high',
                        'LOW': 'low',
                        'CLOSE': 'close',
                        'LAST': 'last',
                        'PREVCLOSE': 'prev_close',
                        'TOTTRDQTY': 'volume',
                        'TOTTRDVAL': 'value',
                        'TOTALTRADES': 'trades',
                        'ISIN': 'isin'
                    }
                    
                    df = df.rename(columns=column_mapping)
                    
                    # Filter only EQ series
                    if 'SERIES' in df.columns:
                        df = df[df['SERIES'] == 'EQ']
                    
                    return df
                    
        except Exception as e:
            logger.error(f"Error processing zip content: {e}")
            return pd.DataFrame()


class NSEDataFetcherFactory:
    """Factory for creating data fetchers"""
    
    @staticmethod
    def create(source: DataSource, **kwargs) -> IDataFetcher:
        """
        Create appropriate data fetcher
        
        Args:
            source: Data source type
            **kwargs: Additional arguments
            
        Returns:
            Data fetcher instance
        """
        if source == DataSource.NSE_BHAVCOPY:
            return NSEDataFetcher(kwargs.get('cache'))
        elif source == DataSource.NSE_DELIVERY:
            # Use the DeliveryDataFetcher for delivery data
            from src.data.fetcher_delivery import DeliveryDataFetcher
            return DeliveryDataFetcher(kwargs.get('cache_dir', 'data'))
        else:
            raise ValueError(f"Unsupported data source: {source}")
