"""
Analysis Engine Implementation
Responsible for analyzing delivery data and detecting spikes
"""

import pandas as pd
import numpy as np
from datetime import date, datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from loguru import logger
from scipy import stats

from src.interfaces import IAnalysisEngine, DeliverySpike, IFilterStrategy


class DeliveryAnalysisEngine(IAnalysisEngine):
    """
    Main analysis engine for delivery data
    Single Responsibility: Analyze delivery patterns and detect spikes
    """
    
    def __init__(self, filters: Optional[List[IFilterStrategy]] = None):
        """
        Initialize analysis engine
        
        Args:
            filters: Optional list of filter strategies to apply
        """
        self.filters = filters or []
        logger.info(f"Analysis engine initialized with {len(self.filters)} filters")
    
    def calculate_average_delivery(self, 
                                  symbol: str, 
                                  data: pd.DataFrame, 
                                  lookback_days: int) -> float:
        """
        Calculate average delivery for a symbol
        
        Args:
            symbol: Stock symbol
            data: Historical data
            lookback_days: Number of days to look back
            
        Returns:
            Average delivery quantity
        """
        # Filter data for the symbol
        symbol_data = data[data['symbol'] == symbol].copy()
        
        if symbol_data.empty:
            return 0.0
        
        # Sort by date and get last N days
        symbol_data = symbol_data.sort_values('date')
        recent_data = symbol_data.tail(lookback_days)
        
        if 'delivery_qty' not in recent_data.columns:
            return 0.0
        
        # Calculate average, excluding zeros for more accurate spike detection
        non_zero_deliveries = recent_data[recent_data['delivery_qty'] > 0]['delivery_qty']
        
        if len(non_zero_deliveries) == 0:
            return 0.0
        
        return non_zero_deliveries.mean()
    
    def detect_spikes(self, 
                     data: pd.DataFrame, 
                     lookback_days: int, 
                     spike_multiplier: float) -> List[DeliverySpike]:
        """
        Detect delivery spikes in the data
        
        Args:
            data: DataFrame with stock data
            lookback_days: Number of days for average calculation
            spike_multiplier: Minimum ratio for spike detection
            
        Returns:
            List of detected spikes
        """
        spikes = []
        
        # Apply filters first
        filtered_data = self._apply_filters(data)
        
        if filtered_data.empty:
            logger.warning("No data after applying filters")
            return spikes
        
        # Get unique dates and symbols
        dates = filtered_data['date'].unique()
        latest_date = max(dates)
        
        # Get data for the latest date
        latest_data = filtered_data[filtered_data['date'] == latest_date]
        
        # Historical data for average calculation
        historical_data = filtered_data[filtered_data['date'] < latest_date]
        
        for _, stock in latest_data.iterrows():
            symbol = stock['symbol']
            current_delivery = stock.get('delivery_qty', 0)
            
            if current_delivery == 0:
                continue
            
            # Calculate average delivery
            avg_delivery = self.calculate_average_delivery(
                symbol, 
                historical_data, 
                lookback_days
            )
            
            if avg_delivery == 0:
                continue
            
            # Calculate spike ratio
            spike_ratio = current_delivery / avg_delivery
            
            if spike_ratio >= spike_multiplier:
                # Calculate additional metrics
                price_change = self._calculate_price_change(symbol, filtered_data)
                volume_change = self._calculate_volume_change(symbol, filtered_data)
                
                spike = DeliverySpike(
                    symbol=symbol,
                    spike_date=latest_date,
                    current_delivery=current_delivery,
                    avg_delivery=avg_delivery,
                    spike_ratio=spike_ratio,
                    price_change=price_change,
                    volume_change=volume_change
                )
                
                spikes.append(spike)
                logger.info(f"Spike detected: {spike}")
        
        # Sort by spike ratio (highest first)
        spikes.sort(key=lambda x: x.spike_ratio, reverse=True)
        
        return spikes
    
    def analyze_trends(self, 
                      symbol: str, 
                      data: pd.DataFrame, 
                      window: int = 5) -> Dict[str, Any]:
        """
        Analyze delivery trends for a symbol
        
        Args:
            symbol: Stock symbol
            data: Historical data
            window: Window size for trend calculation
            
        Returns:
            Dictionary with trend analysis
        """
        # Filter data for the symbol
        symbol_data = data[data['symbol'] == symbol].copy()
        
        if symbol_data.empty or len(symbol_data) < window:
            return {
                'symbol': symbol,
                'trend': 'insufficient_data',
                'trend_strength': 0.0,
                'recent_deliveries': [],
                'statistics': {}
            }
        
        # Sort by date
        symbol_data = symbol_data.sort_values('date')
        
        # Get recent delivery data
        recent_data = symbol_data.tail(window * 2)  # Get more data for better analysis
        
        if 'delivery_qty' not in recent_data.columns:
            return {
                'symbol': symbol,
                'trend': 'no_delivery_data',
                'trend_strength': 0.0,
                'recent_deliveries': [],
                'statistics': {}
            }
        
        # Calculate moving averages
        recent_data['ma_short'] = recent_data['delivery_qty'].rolling(window=window).mean()
        recent_data['ma_long'] = recent_data['delivery_qty'].rolling(window=window*2).mean()
        
        # Determine trend
        trend = self._determine_trend(recent_data)
        
        # Calculate trend strength
        trend_strength = self._calculate_trend_strength(recent_data)
        
        # Calculate statistics
        statistics = self._calculate_statistics(recent_data)
        
        # Get correlation with price
        price_correlation = self._calculate_price_delivery_correlation(recent_data)
        
        return {
            'symbol': symbol,
            'trend': trend,
            'trend_strength': trend_strength,
            'recent_deliveries': recent_data['delivery_qty'].tail(window).tolist(),
            'recent_dates': recent_data['date'].tail(window).tolist(),
            'moving_average_short': recent_data['ma_short'].iloc[-1] if not recent_data['ma_short'].isna().all() else 0,
            'moving_average_long': recent_data['ma_long'].iloc[-1] if not recent_data['ma_long'].isna().all() else 0,
            'statistics': statistics,
            'price_correlation': price_correlation,
            'volatility': statistics.get('std', 0) / statistics.get('mean', 1) if statistics.get('mean', 0) > 0 else 0
        }
    
    def _apply_filters(self, data: pd.DataFrame) -> pd.DataFrame:
        """Apply all configured filters to the data"""
        filtered_data = data.copy()
        
        for filter_strategy in self.filters:
            filtered_data = filter_strategy.filter(filtered_data)
            logger.debug(f"Applied filter: {filter_strategy.get_description()}")
        
        return filtered_data
    
    def _calculate_price_change(self, symbol: str, data: pd.DataFrame) -> float:
        """Calculate price change for a symbol"""
        symbol_data = data[data['symbol'] == symbol].sort_values('date')
        
        if len(symbol_data) < 2:
            return 0.0
        
        latest = symbol_data.iloc[-1]
        previous = symbol_data.iloc[-2]
        
        if 'close' in latest and 'close' in previous:
            prev_close = previous['close']
            if prev_close > 0:
                return ((latest['close'] - prev_close) / prev_close) * 100
        
        return 0.0
    
    def _calculate_volume_change(self, symbol: str, data: pd.DataFrame) -> float:
        """Calculate volume change for a symbol"""
        symbol_data = data[data['symbol'] == symbol].sort_values('date')
        
        if len(symbol_data) < 2:
            return 0.0
        
        latest = symbol_data.iloc[-1]
        
        # Calculate average volume (excluding latest)
        historical_volumes = symbol_data.iloc[:-1]['volume']
        avg_volume = historical_volumes.mean()
        
        if avg_volume > 0:
            return ((latest['volume'] - avg_volume) / avg_volume) * 100
        
        return 0.0
    
    def _determine_trend(self, data: pd.DataFrame) -> str:
        """Determine the trend direction"""
        if len(data) < 3:
            return 'insufficient_data'
        
        # Get last few delivery quantities
        recent_deliveries = data['delivery_qty'].tail(5).values
        
        # Linear regression to determine trend
        x = np.arange(len(recent_deliveries))
        slope, _, _, _, _ = stats.linregress(x, recent_deliveries)
        
        # Calculate percentage of increase/decrease
        avg_delivery = recent_deliveries.mean()
        if avg_delivery > 0:
            trend_percent = (slope / avg_delivery) * 100
            
            if trend_percent > 10:
                return 'strongly_rising'
            elif trend_percent > 2:
                return 'rising'
            elif trend_percent < -10:
                return 'strongly_falling'
            elif trend_percent < -2:
                return 'falling'
        
        return 'sideways'
    
    def _calculate_trend_strength(self, data: pd.DataFrame) -> float:
        """Calculate the strength of the trend (0-100)"""
        if len(data) < 3:
            return 0.0
        
        recent_deliveries = data['delivery_qty'].tail(10).values
        
        if len(recent_deliveries) < 3:
            return 0.0
        
        # Calculate R-squared of linear regression
        x = np.arange(len(recent_deliveries))
        slope, intercept, r_value, _, _ = stats.linregress(x, recent_deliveries)
        
        # R-squared indicates how well the trend line fits
        r_squared = r_value ** 2
        
        # Convert to percentage
        return r_squared * 100
    
    def _calculate_statistics(self, data: pd.DataFrame) -> Dict[str, float]:
        """Calculate statistical measures"""
        delivery_data = data['delivery_qty'].dropna()
        
        if len(delivery_data) == 0:
            return {}
        
        return {
            'mean': delivery_data.mean(),
            'median': delivery_data.median(),
            'std': delivery_data.std(),
            'min': delivery_data.min(),
            'max': delivery_data.max(),
            'q25': delivery_data.quantile(0.25),
            'q75': delivery_data.quantile(0.75),
            'iqr': delivery_data.quantile(0.75) - delivery_data.quantile(0.25),
            'skewness': delivery_data.skew(),
            'kurtosis': delivery_data.kurtosis()
        }
    
    def _calculate_price_delivery_correlation(self, data: pd.DataFrame) -> float:
        """Calculate correlation between price and delivery"""
        if 'close' not in data.columns or 'delivery_qty' not in data.columns:
            return 0.0
        
        # Remove any NaN values
        clean_data = data[['close', 'delivery_qty']].dropna()
        
        if len(clean_data) < 3:
            return 0.0
        
        correlation = clean_data['close'].corr(clean_data['delivery_qty'])
        
        return correlation if not np.isnan(correlation) else 0.0


class AdvancedSpikeDetector(IAnalysisEngine):
    """
    Advanced spike detection using statistical methods
    Extends the base analysis with more sophisticated algorithms
    """
    
    def __init__(self, z_score_threshold: float = 3.0):
        """
        Initialize advanced spike detector
        
        Args:
            z_score_threshold: Z-score threshold for outlier detection
        """
        self.z_score_threshold = z_score_threshold
        logger.info(f"Advanced spike detector initialized with z-score threshold: {z_score_threshold}")
    
    def calculate_average_delivery(self, 
                                  symbol: str, 
                                  data: pd.DataFrame, 
                                  lookback_days: int) -> float:
        """Calculate robust average using median"""
        symbol_data = data[data['symbol'] == symbol].copy()
        
        if symbol_data.empty:
            return 0.0
        
        symbol_data = symbol_data.sort_values('date').tail(lookback_days)
        
        if 'delivery_qty' not in symbol_data.columns:
            return 0.0
        
        # Use median for robustness against outliers
        return symbol_data['delivery_qty'].median()
    
    def detect_spikes(self, 
                     data: pd.DataFrame, 
                     lookback_days: int, 
                     spike_multiplier: float) -> List[DeliverySpike]:
        """
        Detect spikes using z-score method
        
        Args:
            data: DataFrame with stock data
            lookback_days: Number of days for average calculation
            spike_multiplier: Minimum ratio for spike detection
            
        Returns:
            List of detected spikes
        """
        spikes = []
        
        # Group by symbol
        for symbol in data['symbol'].unique():
            symbol_data = data[data['symbol'] == symbol].copy()
            symbol_data = symbol_data.sort_values('date')
            
            if len(symbol_data) < lookback_days + 1:
                continue
            
            # Calculate z-scores
            symbol_data['z_score'] = self._calculate_z_scores(symbol_data['delivery_qty'])
            
            # Get latest data point
            latest = symbol_data.iloc[-1]
            
            # Check for spike using both z-score and multiplier
            if abs(latest['z_score']) >= self.z_score_threshold:
                avg_delivery = self.calculate_average_delivery(symbol, symbol_data.iloc[:-1], lookback_days)
                
                if avg_delivery > 0:
                    spike_ratio = latest['delivery_qty'] / avg_delivery
                    
                    if spike_ratio >= spike_multiplier:
                        spike = DeliverySpike(
                            symbol=symbol,
                            spike_date=latest['date'],
                            current_delivery=latest['delivery_qty'],
                            avg_delivery=avg_delivery,
                            spike_ratio=spike_ratio,
                            price_change=self._calculate_price_change_pct(symbol_data),
                            volume_change=self._calculate_volume_change_pct(symbol_data)
                        )
                        spikes.append(spike)
        
        return sorted(spikes, key=lambda x: x.spike_ratio, reverse=True)
    
    def analyze_trends(self, 
                      symbol: str, 
                      data: pd.DataFrame, 
                      window: int = 5) -> Dict[str, Any]:
        """Analyze trends with additional statistical measures"""
        base_analysis = DeliveryAnalysisEngine().analyze_trends(symbol, data, window)
        
        # Add advanced metrics
        symbol_data = data[data['symbol'] == symbol].copy()
        
        if not symbol_data.empty and 'delivery_qty' in symbol_data.columns:
            # Calculate momentum
            momentum = self._calculate_momentum(symbol_data['delivery_qty'])
            
            # Calculate relative strength
            rs = self._calculate_relative_strength(symbol_data)
            
            base_analysis['momentum'] = momentum
            base_analysis['relative_strength'] = rs
            base_analysis['z_scores'] = self._calculate_z_scores(symbol_data['delivery_qty'].tail(window)).tolist()
        
        return base_analysis
    
    def _calculate_z_scores(self, series: pd.Series) -> pd.Series:
        """Calculate z-scores for a series"""
        mean = series.mean()
        std = series.std()
        
        if std == 0:
            return pd.Series([0] * len(series), index=series.index)
        
        return (series - mean) / std
    
    def _calculate_momentum(self, series: pd.Series, period: int = 10) -> float:
        """Calculate momentum indicator"""
        if len(series) < period + 1:
            return 0.0
        
        current = series.iloc[-1]
        past = series.iloc[-period-1]
        
        if past == 0:
            return 0.0
        
        return ((current - past) / past) * 100
    
    def _calculate_relative_strength(self, data: pd.DataFrame) -> float:
        """Calculate relative strength of delivery vs volume"""
        if 'delivery_qty' not in data.columns or 'volume' not in data.columns:
            return 0.0
        
        recent_data = data.tail(14)  # 14 days for RSI-like calculation
        
        if len(recent_data) < 2:
            return 0.0
        
        # Calculate delivery percentage changes
        delivery_pct = recent_data['delivery_percent'] if 'delivery_percent' in recent_data.columns else (recent_data['delivery_qty'] / recent_data['volume'] * 100)
        
        # Calculate gains and losses
        changes = delivery_pct.diff()
        gains = changes.where(changes > 0, 0)
        losses = -changes.where(changes < 0, 0)
        
        avg_gain = gains.mean()
        avg_loss = losses.mean()
        
        if avg_loss == 0:
            return 100.0
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    def _calculate_price_change_pct(self, data: pd.DataFrame) -> float:
        """Calculate percentage price change"""
        if len(data) < 2 or 'close' not in data.columns:
            return 0.0
        
        latest = data.iloc[-1]['close']
        previous = data.iloc[-2]['close']
        
        if previous == 0:
            return 0.0
        
        return ((latest - previous) / previous) * 100
    
    def _calculate_volume_change_pct(self, data: pd.DataFrame) -> float:
        """Calculate percentage volume change"""
        if len(data) < 2 or 'volume' not in data.columns:
            return 0.0
        
        recent_volume = data.iloc[-1]['volume']
        avg_volume = data.iloc[:-1]['volume'].mean()
        
        if avg_volume == 0:
            return 0.0
        
        return ((recent_volume - avg_volume) / avg_volume) * 100
