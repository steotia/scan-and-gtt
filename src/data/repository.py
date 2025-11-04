"""
Data Repository Implementation
Responsible for data storage and retrieval
"""

import pandas as pd
import sqlite3
from pathlib import Path
from datetime import date, datetime
from typing import Optional, List
from loguru import logger
import json

from src.interfaces import IDataRepository


class CSVDataRepository(IDataRepository):
    """
    CSV-based data repository
    Single Responsibility: Store and retrieve data in CSV format
    """
    
    def __init__(self, base_path: str = "data"):
        """
        Initialize CSV repository
        
        Args:
            base_path: Base directory for CSV files
        """
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)
        logger.info(f"CSV Repository initialized at {self.base_path}")
    
    def save(self, data: pd.DataFrame, data_date: date) -> bool:
        """
        Save data to CSV
        
        Args:
            data: DataFrame to save
            data_date: Date of the data
            
        Returns:
            True if successful
        """
        try:
            # Create year/month subdirectory
            year_month_dir = self.base_path / str(data_date.year) / f"{data_date.month:02d}"
            year_month_dir.mkdir(parents=True, exist_ok=True)
            
            # Save to CSV
            file_path = year_month_dir / f"nse_data_{data_date.isoformat()}.csv"
            data.to_csv(file_path, index=False)
            
            logger.info(f"Data saved: {file_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving data: {e}")
            return False
    
    def load(self, data_date: date) -> Optional[pd.DataFrame]:
        """
        Load data from CSV
        
        Args:
            data_date: Date to load
            
        Returns:
            DataFrame or None if not found
        """
        try:
            file_path = self.base_path / str(data_date.year) / f"{data_date.month:02d}" / f"nse_data_{data_date.isoformat()}.csv"
            
            if file_path.exists():
                data = pd.read_csv(file_path)
                logger.debug(f"Data loaded: {file_path}")
                return data
            else:
                logger.debug(f"File not found: {file_path}")
                return None
                
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            return None
    
    def load_range(self, start_date: date, end_date: date) -> pd.DataFrame:
        """
        Load data for a date range
        
        Args:
            start_date: Start date
            end_date: End date
            
        Returns:
            Combined DataFrame
        """
        all_data = []
        current_date = start_date
        
        while current_date <= end_date:
            if current_date.weekday() < 5:  # Skip weekends
                data = self.load(current_date)
                if data is not None:
                    data['date'] = current_date
                    all_data.append(data)
            
            current_date = pd.Timestamp(current_date) + pd.Timedelta(days=1)
            current_date = current_date.date()
        
        if all_data:
            combined = pd.concat(all_data, ignore_index=True)
            logger.info(f"Loaded {len(combined)} records from {start_date} to {end_date}")
            return combined
        
        return pd.DataFrame()
    
    def exists(self, data_date: date) -> bool:
        """
        Check if data exists for a date
        
        Args:
            data_date: Date to check
            
        Returns:
            True if exists
        """
        file_path = self.base_path / str(data_date.year) / f"{data_date.month:02d}" / f"nse_data_{data_date.isoformat()}.csv"
        return file_path.exists()
    
    def get_available_dates(self) -> List[date]:
        """
        Get list of dates with available data
        
        Returns:
            List of dates
        """
        dates = []
        
        # Walk through all CSV files
        for csv_file in self.base_path.rglob("nse_data_*.csv"):
            try:
                # Extract date from filename
                date_str = csv_file.stem.replace("nse_data_", "")
                data_date = date.fromisoformat(date_str)
                dates.append(data_date)
            except:
                continue
        
        return sorted(dates)


class SQLiteDataRepository(IDataRepository):
    """
    SQLite-based data repository
    Single Responsibility: Store and retrieve data in SQLite database
    """
    
    def __init__(self, db_path: str = "data/nse_data.db"):
        """
        Initialize SQLite repository
        
        Args:
            db_path: Path to SQLite database
        """
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Create table if not exists
        self._create_table()
        logger.info(f"SQLite Repository initialized at {self.db_path}")
    
    def _create_table(self):
        """Create table structure"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS stock_data (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    date DATE NOT NULL,
                    symbol TEXT NOT NULL,
                    open REAL,
                    high REAL,
                    low REAL,
                    close REAL,
                    volume INTEGER,
                    delivery_qty INTEGER,
                    delivery_percent REAL,
                    vwap REAL,
                    trades INTEGER,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(date, symbol)
                )
            """)
            
            # Create index for faster queries
            conn.execute("CREATE INDEX IF NOT EXISTS idx_date ON stock_data(date)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_symbol ON stock_data(symbol)")
    
    def save(self, data: pd.DataFrame, data_date: date) -> bool:
        """
        Save data to SQLite
        
        Args:
            data: DataFrame to save
            data_date: Date of the data
            
        Returns:
            True if successful
        """
        try:
            # Add date column if not present
            if 'date' not in data.columns:
                data['date'] = data_date
            
            with sqlite3.connect(self.db_path) as conn:
                # Use replace to handle duplicates
                data.to_sql('stock_data', conn, if_exists='append', index=False)
            
            logger.info(f"Data saved to SQLite: {len(data)} records for {data_date}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving to SQLite: {e}")
            return False
    
    def load(self, data_date: date) -> Optional[pd.DataFrame]:
        """
        Load data from SQLite
        
        Args:
            data_date: Date to load
            
        Returns:
            DataFrame or None if not found
        """
        try:
            query = "SELECT * FROM stock_data WHERE date = ?"
            
            with sqlite3.connect(self.db_path) as conn:
                data = pd.read_sql_query(query, conn, params=(data_date.isoformat(),))
            
            if not data.empty:
                logger.debug(f"Loaded {len(data)} records for {data_date}")
                return data
            else:
                return None
                
        except Exception as e:
            logger.error(f"Error loading from SQLite: {e}")
            return None
    
    def load_range(self, start_date: date, end_date: date) -> pd.DataFrame:
        """
        Load data for a date range
        
        Args:
            start_date: Start date
            end_date: End date
            
        Returns:
            Combined DataFrame
        """
        try:
            query = "SELECT * FROM stock_data WHERE date BETWEEN ? AND ?"
            
            with sqlite3.connect(self.db_path) as conn:
                data = pd.read_sql_query(
                    query, 
                    conn, 
                    params=(start_date.isoformat(), end_date.isoformat())
                )
            
            if not data.empty:
                # Convert date strings to date objects
                data['date'] = pd.to_datetime(data['date']).dt.date
                logger.info(f"Loaded {len(data)} records from {start_date} to {end_date}")
            
            return data
            
        except Exception as e:
            logger.error(f"Error loading range from SQLite: {e}")
            return pd.DataFrame()
    
    def exists(self, data_date: date) -> bool:
        """
        Check if data exists for a date
        
        Args:
            data_date: Date to check
            
        Returns:
            True if exists
        """
        try:
            query = "SELECT COUNT(*) FROM stock_data WHERE date = ?"
            
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute(query, (data_date.isoformat(),))
                count = cursor.fetchone()[0]
            
            return count > 0
            
        except Exception as e:
            logger.error(f"Error checking existence: {e}")
            return False
    
    def get_available_dates(self) -> List[date]:
        """
        Get list of dates with available data
        
        Returns:
            List of dates
        """
        try:
            query = "SELECT DISTINCT date FROM stock_data ORDER BY date"
            
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute(query)
                dates = [date.fromisoformat(row[0]) for row in cursor.fetchall()]
            
            return dates
            
        except Exception as e:
            logger.error(f"Error getting available dates: {e}")
            return []


class DataRepositoryFactory:
    """Factory for creating data repositories"""
    
    @staticmethod
    def create(storage_type: str, **kwargs) -> IDataRepository:
        """
        Create appropriate data repository
        
        Args:
            storage_type: Type of storage (csv, sqlite)
            **kwargs: Additional arguments
            
        Returns:
            Data repository instance
        """
        if storage_type.lower() == "csv":
            return CSVDataRepository(kwargs.get('base_path', 'data'))
        elif storage_type.lower() == "sqlite":
            return SQLiteDataRepository(kwargs.get('db_path', 'data/nse_data.db'))
        else:
            raise ValueError(f"Unsupported storage type: {storage_type}")
