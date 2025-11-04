"""
Interfaces (Abstract Base Classes) for the NSE Delivery Tracker
Following the Interface Segregation and Dependency Inversion principles
"""

from abc import ABC, abstractmethod
from datetime import date, datetime
from typing import Dict, List, Optional, Any, Protocol
import pandas as pd
from dataclasses import dataclass
from enum import Enum


class DataSource(Enum):
    """Enumeration of data sources"""
    NSE_BHAVCOPY = "nse_bhavcopy"
    NSE_DELIVERY = "nse_delivery"
    BSE = "bse"


@dataclass
class StockData:
    """Data class for stock information"""
    symbol: str
    date: date
    open: float
    high: float
    low: float
    close: float
    volume: int
    delivery_qty: int
    delivery_percent: float
    trades: int
    vwap: float
    
    @property
    def delivery_value(self) -> float:
        """Calculate delivery value"""
        return self.delivery_qty * self.vwap


@dataclass
class DeliverySpike:
    """Data class for delivery spike information"""
    symbol: str
    spike_date: date
    current_delivery: int
    avg_delivery: float
    spike_ratio: float
    price_change: float
    volume_change: float
    
    def __str__(self) -> str:
        return f"{self.symbol}: {self.spike_ratio:.1f}x spike on {self.spike_date}"


class IDataFetcher(ABC):
    """Interface for data fetching operations"""
    
    @abstractmethod
    async def fetch_daily_data(self, fetch_date: date) -> pd.DataFrame:
        """Fetch daily bhavcopy/delivery data"""
        pass
    
    @abstractmethod
    async def fetch_historical_data(self, start_date: date, end_date: date) -> pd.DataFrame:
        """Fetch historical data for a date range"""
        pass
    
    @abstractmethod
    def validate_data(self, data: pd.DataFrame) -> bool:
        """Validate fetched data for completeness"""
        pass


class IDataRepository(ABC):
    """Interface for data storage operations"""
    
    @abstractmethod
    def save(self, data: pd.DataFrame, data_date: date) -> bool:
        """Save data to storage"""
        pass
    
    @abstractmethod
    def load(self, data_date: date) -> Optional[pd.DataFrame]:
        """Load data from storage"""
        pass
    
    @abstractmethod
    def load_range(self, start_date: date, end_date: date) -> pd.DataFrame:
        """Load data for a date range"""
        pass
    
    @abstractmethod
    def exists(self, data_date: date) -> bool:
        """Check if data exists for a date"""
        pass
    
    @abstractmethod
    def get_available_dates(self) -> List[date]:
        """Get list of dates with available data"""
        pass


class IAnalysisEngine(ABC):
    """Interface for analysis operations"""
    
    @abstractmethod
    def calculate_average_delivery(self, 
                                  symbol: str, 
                                  data: pd.DataFrame, 
                                  lookback_days: int) -> float:
        """Calculate average delivery for a symbol"""
        pass
    
    @abstractmethod
    def detect_spikes(self, 
                     data: pd.DataFrame, 
                     lookback_days: int, 
                     spike_multiplier: float) -> List[DeliverySpike]:
        """Detect delivery spikes in the data"""
        pass
    
    @abstractmethod
    def analyze_trends(self, 
                      symbol: str, 
                      data: pd.DataFrame, 
                      window: int) -> Dict[str, Any]:
        """Analyze delivery trends for a symbol"""
        pass


class IFilterStrategy(ABC):
    """Interface for filtering strategies"""
    
    @abstractmethod
    def filter(self, data: pd.DataFrame) -> pd.DataFrame:
        """Apply filter to the data"""
        pass
    
    @abstractmethod
    def get_description(self) -> str:
        """Get description of the filter"""
        pass


class IReportGenerator(ABC):
    """Interface for report generation"""
    
    @abstractmethod
    def generate(self, 
                analysis_results: Dict[str, Any], 
                output_path: str) -> bool:
        """Generate report from analysis results"""
        pass
    
    @abstractmethod
    def add_chart(self, chart_data: pd.DataFrame, chart_type: str) -> None:
        """Add chart to the report"""
        pass


class INotificationService(ABC):
    """Interface for notification services"""
    
    @abstractmethod
    async def send_alert(self, 
                        subject: str, 
                        message: str, 
                        attachments: Optional[List[str]] = None) -> bool:
        """Send alert notification"""
        pass
    
    @abstractmethod
    def format_spike_alert(self, spikes: List[DeliverySpike]) -> str:
        """Format spike information for alert"""
        pass


class IConfiguration(ABC):
    """Interface for configuration management"""
    
    @abstractmethod
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value"""
        pass
    
    @abstractmethod
    def set(self, key: str, value: Any) -> None:
        """Set configuration value"""
        pass
    
    @abstractmethod
    def load_from_file(self, file_path: str) -> bool:
        """Load configuration from file"""
        pass
    
    @abstractmethod
    def save_to_file(self, file_path: str) -> bool:
        """Save configuration to file"""
        pass


class ILogger(Protocol):
    """Protocol for logging operations"""
    
    def info(self, message: str) -> None: ...
    def debug(self, message: str) -> None: ...
    def warning(self, message: str) -> None: ...
    def error(self, message: str) -> None: ...
    def critical(self, message: str) -> None: ...


class ICache(ABC):
    """Interface for caching operations"""
    
    @abstractmethod
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache"""
        pass
    
    @abstractmethod
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """Set value in cache with optional TTL"""
        pass
    
    @abstractmethod
    def delete(self, key: str) -> bool:
        """Delete value from cache"""
        pass
    
    @abstractmethod
    def clear(self) -> None:
        """Clear all cache"""
        pass


class IValidator(ABC):
    """Interface for data validation"""
    
    @abstractmethod
    def validate(self, data: Any) -> tuple[bool, Optional[str]]:
        """Validate data and return (is_valid, error_message)"""
        pass


class IMetricsCollector(ABC):
    """Interface for metrics collection"""
    
    @abstractmethod
    def record_metric(self, metric_name: str, value: float, tags: Optional[Dict] = None) -> None:
        """Record a metric"""
        pass
    
    @abstractmethod
    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get summary of collected metrics"""
        pass
