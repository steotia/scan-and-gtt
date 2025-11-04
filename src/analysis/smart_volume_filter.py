"""
Smart Volume Filter with Price Brackets
Full implementation with historical analysis and index adjustments
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from datetime import date, timedelta
from loguru import logger

from src.interfaces import IFilterStrategy


@dataclass
class VolumeMetrics:
    """Data class for volume metrics"""
    symbol: str
    current_price: float
    current_volume: int
    current_turnover: float
    avg_volume: float
    avg_price: float
    avg_turnover: float
    relative_volume: float
    turnover_ratio: float
    bracket: str
    passes_filter: bool
    failure_reason: Optional[str] = None


class SmartVolumeFilter(IFilterStrategy):
    """
    Advanced volume filter using price-aware brackets
    Adjusts requirements based on stock price ranges and historical patterns
    """
    
    # Default bracket configuration
    DEFAULT_BRACKETS = {
        'penny': {
            'price_range': (0, 50),
            'min_turnover': 2_00_00_000,      # ₹2 Crores
            'min_avg_turnover': 1_00_00_000,  # ₹1 Crore avg
            'min_rvol': 2.0,                  # 2x normal volume
            'min_turnover_ratio': 1.5,        # Today/Avg turnover
            'description': 'Penny stocks (<₹50)'
        },
        'low': {
            'price_range': (50, 500),
            'min_turnover': 5_00_00_000,      # ₹5 Crores
            'min_avg_turnover': 2_00_00_000,  # ₹2 Crores avg
            'min_rvol': 1.5,                  # 1.5x normal
            'min_turnover_ratio': 1.3,
            'description': 'Low-price stocks (₹50-500)'
        },
        'mid': {
            'price_range': (500, 2000),
            'min_turnover': 10_00_00_000,     # ₹10 Crores
            'min_avg_turnover': 5_00_00_000,  # ₹5 Crores avg
            'min_rvol': 1.3,                  # 1.3x normal
            'min_turnover_ratio': 1.2,
            'description': 'Mid-price stocks (₹500-2000)'
        },
        'high': {
            'price_range': (2000, 10000),
            'min_turnover': 10_00_00_000,     # ₹10 Crores
            'min_avg_turnover': 5_00_00_000,  # ₹5 Crores avg
            'min_rvol': 1.2,                  # 1.2x normal
            'min_turnover_ratio': 1.1,
            'description': 'High-price stocks (₹2000-10000)'
        },
        'ultra_high': {
            'price_range': (10000, float('inf')),
            'min_turnover': 5_00_00_000,      # ₹5 Crores (lower OK)
            'min_avg_turnover': 2_00_00_000,  # ₹2 Crores avg
            'min_rvol': 1.1,                  # 1.1x normal
            'min_turnover_ratio': 1.0,        # Any increase is OK
            'description': 'Ultra high-price stocks (>₹10000)'
        }
    }
    
    # Index-specific multipliers
    INDEX_MULTIPLIERS = {
        'NIFTY_50': 2.0,         # Strict for large caps
        'NIFTY_NEXT_50': 1.5,    
        'NIFTY_100': 1.5,
        'LARGECAP': 1.5,
        'NIFTY_MIDCAP': 1.0,     # Normal for mid caps
        'MIDCAP': 1.0,
        'NIFTY_SMALLCAP': 0.7,   # Relaxed for small caps
        'SMALLCAP': 0.7,
        'NIFTY_BANK': 1.8,       # Banks need high liquidity
        'NIFTY_IT': 1.3,
        'NIFTY_PHARMA': 1.2,
        'NIFTY_AUTO': 1.2,
        'NIFTY_FMCG': 1.1,
        'NIFTY_METAL': 1.0,
        'NIFTY_REALTY': 0.8,
        'NIFTY_ENERGY': 1.3
    }
    
    def __init__(self, 
                 brackets: Optional[Dict] = None,
                 lookback_days: int = 20,
                 index_filter: Optional[str] = None,
                 mode: str = 'normal',
                 enable_logging: bool = True):
        """
        Initialize Smart Volume Filter
        
        Args:
            brackets: Custom bracket configuration (overrides defaults)
            lookback_days: Days to calculate averages
            index_filter: Current index being filtered
            mode: 'conservative', 'normal', or 'aggressive'
            enable_logging: Whether to log detailed metrics
        """
        self.brackets = brackets or self.DEFAULT_BRACKETS.copy()
        self.lookback_days = lookback_days
        self.index_filter = index_filter
        self.mode = mode
        self.enable_logging = enable_logging
        
        # Adjust brackets based on mode
        self._adjust_for_mode()
        
        # Store metrics for analysis
        self.last_metrics: List[VolumeMetrics] = []
        
    def filter(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Apply smart volume filter based on price brackets
        
        Args:
            data: DataFrame with columns: symbol, date, close, volume, vwap, delivery_qty
            
        Returns:
            Filtered DataFrame
        """
        # Validate required columns
        required = ['symbol', 'date', 'close', 'volume']
        if not all(col in data.columns for col in required):
            logger.warning(f"Missing required columns. Need: {required}")
            return data
        
        # Use vwap if available, otherwise use close
        if 'vwap' not in data.columns:
            data['vwap'] = data['close']
        
        # Get the latest date in the data
        latest_date = data['date'].max()
        cutoff_date = latest_date - timedelta(days=self.lookback_days + 10)
        
        # Process each symbol
        filtered_symbols = []
        self.last_metrics = []
        
        # Group by symbol for efficiency
        for symbol, stock_data in data.groupby('symbol'):
            metrics = self._analyze_stock(symbol, stock_data, latest_date, cutoff_date)
            self.last_metrics.append(metrics)
            
            if metrics.passes_filter:
                filtered_symbols.append(symbol)
        
        # Log summary statistics
        self._log_filter_summary()
        
        # Return filtered data for the latest date
        latest_data = data[data['date'] == latest_date]
        return latest_data[latest_data['symbol'].isin(filtered_symbols)]
    
    def _analyze_stock(self, symbol: str, stock_data: pd.DataFrame, 
                       latest_date: date, cutoff_date: date) -> VolumeMetrics:
        """
        Analyze a single stock's volume metrics
        """
        # Get latest day's data
        latest_data = stock_data[stock_data['date'] == latest_date]
        
        if latest_data.empty:
            return VolumeMetrics(
                symbol=symbol,
                current_price=0,
                current_volume=0,
                current_turnover=0,
                avg_volume=0,
                avg_price=0,
                avg_turnover=0,
                relative_volume=0,
                turnover_ratio=0,
                bracket='unknown',
                passes_filter=False,
                failure_reason='No data for latest date'
            )
        
        latest = latest_data.iloc[0]
        current_price = latest['close']
        current_volume = latest['volume']
        current_vwap = latest['vwap']
        current_turnover = current_volume * current_vwap
        
        # Get historical data (excluding latest)
        historical = stock_data[
            (stock_data['date'] < latest_date) & 
            (stock_data['date'] >= cutoff_date)
        ]
        
        # Need minimum history for reliable averages
        if len(historical) < 5:
            return VolumeMetrics(
                symbol=symbol,
                current_price=current_price,
                current_volume=current_volume,
                current_turnover=current_turnover,
                avg_volume=0,
                avg_price=0,
                avg_turnover=0,
                relative_volume=0,
                turnover_ratio=0,
                bracket=self._get_bracket_name(current_price),
                passes_filter=False,
                failure_reason=f'Insufficient history ({len(historical)} days)'
            )
        
        # Calculate averages
        avg_volume = historical['volume'].mean()
        avg_price = historical['vwap'].mean()
        avg_turnover = avg_volume * avg_price
        
        # Calculate relative metrics
        relative_volume = current_volume / avg_volume if avg_volume > 0 else 0
        turnover_ratio = current_turnover / avg_turnover if avg_turnover > 0 else 0
        
        # Get price bracket
        bracket_name = self._get_bracket_name(current_price)
        bracket = self.brackets[bracket_name]
        
        # Apply index multiplier if applicable
        multiplier = self._get_index_multiplier()
        
        # Check all conditions
        passes, reason = self._check_conditions(
            bracket, current_turnover, avg_turnover, 
            relative_volume, turnover_ratio, multiplier
        )
        
        return VolumeMetrics(
            symbol=symbol,
            current_price=current_price,
            current_volume=current_volume,
            current_turnover=current_turnover,
            avg_volume=avg_volume,
            avg_price=avg_price,
            avg_turnover=avg_turnover,
            relative_volume=relative_volume,
            turnover_ratio=turnover_ratio,
            bracket=bracket_name,
            passes_filter=passes,
            failure_reason=None if passes else reason
        )
    
    def _get_bracket_name(self, price: float) -> str:
        """Get the appropriate bracket name for a price"""
        for name, bracket in self.brackets.items():
            min_price, max_price = bracket['price_range']
            if min_price <= price < max_price:
                return name
        return 'mid'  # Default fallback
    
    def _get_index_multiplier(self) -> float:
        """Get multiplier based on current index filter"""
        if self.index_filter and self.index_filter in self.INDEX_MULTIPLIERS:
            return self.INDEX_MULTIPLIERS[self.index_filter]
        return 1.0
    
    def _check_conditions(self, bracket: dict, current_turnover: float,
                         avg_turnover: float, rvol: float, 
                         turnover_ratio: float, multiplier: float) -> Tuple[bool, Optional[str]]:
        """
        Check if stock passes all conditions for its bracket
        
        Returns:
            (passes, failure_reason)
        """
        # Apply multiplier to thresholds
        min_turnover = bracket['min_turnover'] * multiplier
        min_avg_turnover = bracket['min_avg_turnover'] * multiplier
        min_rvol = bracket['min_rvol']
        min_turnover_ratio = bracket['min_turnover_ratio']
        
        # Check each condition
        if current_turnover < min_turnover:
            return False, f"Current turnover ₹{current_turnover/1_00_000:.1f}L < ₹{min_turnover/1_00_000:.1f}L"
        
        if avg_turnover < min_avg_turnover:
            return False, f"Avg turnover ₹{avg_turnover/1_00_000:.1f}L < ₹{min_avg_turnover/1_00_000:.1f}L"
        
        if rvol < min_rvol:
            return False, f"RVOL {rvol:.2f}x < {min_rvol}x"
        
        if turnover_ratio < min_turnover_ratio:
            return False, f"Turnover ratio {turnover_ratio:.2f}x < {min_turnover_ratio}x"
        
        return True, None
    
    def _adjust_for_mode(self):
        """Adjust brackets based on mode (conservative/normal/aggressive)"""
        if self.mode == 'conservative':
            # Increase all requirements by 50%
            for bracket in self.brackets.values():
                bracket['min_turnover'] = int(bracket['min_turnover'] * 1.5)
                bracket['min_avg_turnover'] = int(bracket['min_avg_turnover'] * 1.5)
                bracket['min_rvol'] = bracket['min_rvol'] * 1.2
                
        elif self.mode == 'aggressive':
            # Decrease requirements by 30%
            for bracket in self.brackets.values():
                bracket['min_turnover'] = int(bracket['min_turnover'] * 0.7)
                bracket['min_avg_turnover'] = int(bracket['min_avg_turnover'] * 0.7)
                bracket['min_rvol'] = max(1.0, bracket['min_rvol'] * 0.8)
    
    def _log_filter_summary(self):
        """Log summary statistics of the filtering"""
        if not self.enable_logging or not self.last_metrics:
            return
        
        total = len(self.last_metrics)
        passed = sum(1 for m in self.last_metrics if m.passes_filter)
        
        # Group by bracket
        bracket_stats = {}
        for metric in self.last_metrics:
            if metric.bracket not in bracket_stats:
                bracket_stats[metric.bracket] = {'total': 0, 'passed': 0}
            bracket_stats[metric.bracket]['total'] += 1
            if metric.passes_filter:
                bracket_stats[metric.bracket]['passed'] += 1
        
        logger.info(f"Smart Volume Filter Summary:")
        logger.info(f"  Total stocks: {total}, Passed: {passed} ({passed/total*100:.1f}%)")
        logger.info(f"  Mode: {self.mode}, Index: {self.index_filter or 'None'}")
        
        for bracket, stats in bracket_stats.items():
            if stats['total'] > 0:
                pct = stats['passed'] / stats['total'] * 100
                logger.info(f"  {bracket}: {stats['passed']}/{stats['total']} ({pct:.1f}%)")
        
        # Log top failures if in debug mode
        if logger._core.min_level <= 10:  # DEBUG level
            failures = [m for m in self.last_metrics if not m.passes_filter][:5]
            if failures:
                logger.debug("Top 5 filtered stocks:")
                for m in failures:
                    logger.debug(f"  {m.symbol}: {m.failure_reason}")
    
    def get_description(self) -> str:
        """Get filter description"""
        index_part = f" [{self.index_filter}]" if self.index_filter else ""
        return f"Smart Volume Filter ({self.mode}){index_part}"
    
    def get_metrics(self) -> pd.DataFrame:
        """
        Get detailed metrics from last filtering operation
        
        Returns:
            DataFrame with all volume metrics
        """
        if not self.last_metrics:
            return pd.DataFrame()
        
        metrics_data = []
        for m in self.last_metrics:
            metrics_data.append({
                'symbol': m.symbol,
                'price': m.current_price,
                'volume': m.current_volume,
                'turnover_cr': m.current_turnover / 1_00_00_000,
                'avg_turnover_cr': m.avg_turnover / 1_00_00_000,
                'rvol': m.relative_volume,
                'turnover_ratio': m.turnover_ratio,
                'bracket': m.bracket,
                'passed': m.passes_filter,
                'reason': m.failure_reason or 'Passed'
            })
        
        df = pd.DataFrame(metrics_data)
        return df.sort_values('turnover_cr', ascending=False)
    
    def get_statistics(self) -> Dict:
        """Get detailed statistics about the filtering"""
        if not self.last_metrics:
            return {}
        
        passed_metrics = [m for m in self.last_metrics if m.passes_filter]
        failed_metrics = [m for m in self.last_metrics if not m.passes_filter]
        
        stats = {
            'total_stocks': len(self.last_metrics),
            'passed': len(passed_metrics),
            'failed': len(failed_metrics),
            'pass_rate': len(passed_metrics) / len(self.last_metrics) * 100 if self.last_metrics else 0,
            'avg_rvol_passed': np.mean([m.relative_volume for m in passed_metrics]) if passed_metrics else 0,
            'avg_rvol_failed': np.mean([m.relative_volume for m in failed_metrics]) if failed_metrics else 0,
            'total_turnover_passed_cr': sum(m.current_turnover for m in passed_metrics) / 1_00_00_000,
            'total_turnover_failed_cr': sum(m.current_turnover for m in failed_metrics) / 1_00_00_000,
        }
        
        # Failure reasons breakdown
        failure_reasons = {}
        for m in failed_metrics:
            reason_type = m.failure_reason.split('<')[0].strip() if m.failure_reason else 'Unknown'
            failure_reasons[reason_type] = failure_reasons.get(reason_type, 0) + 1
        stats['failure_reasons'] = failure_reasons
        
        return stats


class SmartDeliveryVolumeFilter(IFilterStrategy):
    """
    Combined filter that correlates delivery spikes with volume patterns
    High Delivery + High Volume = Strong accumulation signal
    """
    
    def __init__(self,
                 volume_filter: Optional[SmartVolumeFilter] = None,
                 min_delivery_percent: float = 40.0,
                 delivery_rvol_correlation: float = 0.5):
        """
        Initialize combined delivery-volume filter
        
        Args:
            volume_filter: Smart volume filter instance
            min_delivery_percent: Minimum delivery percentage
            delivery_rvol_correlation: Min correlation between delivery% and RVOL
        """
        self.volume_filter = volume_filter or SmartVolumeFilter()
        self.min_delivery_percent = min_delivery_percent
        self.delivery_rvol_correlation = delivery_rvol_correlation
        
    def filter(self, data: pd.DataFrame) -> pd.DataFrame:
        """Apply combined delivery and volume filter"""
        
        # First apply volume filter
        volume_filtered = self.volume_filter.filter(data)
        
        if volume_filtered.empty or 'delivery_percent' not in volume_filtered.columns:
            return volume_filtered
        
        # Then filter by delivery percentage
        delivery_filtered = volume_filtered[
            volume_filtered['delivery_percent'] >= self.min_delivery_percent
        ]
        
        # Advanced: Check correlation between delivery and volume
        if self.delivery_rvol_correlation > 0:
            delivery_filtered = self._check_correlation(delivery_filtered)
        
        logger.info(f"Delivery-Volume Filter: {len(volume_filtered)} -> {len(delivery_filtered)} stocks")
        
        return delivery_filtered
    
    def _check_correlation(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Check if high delivery correlates with high volume
        Filters out stocks where delivery spike might be manipulation
        """
        filtered_symbols = []
        
        for symbol in data['symbol'].unique():
            stock_data = data[data['symbol'] == symbol]
            
            # Get volume metrics for this stock
            metrics = next((m for m in self.volume_filter.last_metrics 
                          if m.symbol == symbol), None)
            
            if metrics and metrics.relative_volume > 1.0:
                # High delivery should come with high volume
                delivery_pct = stock_data['delivery_percent'].iloc[0]
                
                # Simple correlation check
                if delivery_pct > 60 and metrics.relative_volume > 1.5:
                    filtered_symbols.append(symbol)  # Strong signal
                elif delivery_pct > 40 and metrics.relative_volume > 1.2:
                    filtered_symbols.append(symbol)  # Good signal
                elif metrics.relative_volume > 2.0:
                    filtered_symbols.append(symbol)  # Volume spike alone
        
        return data[data['symbol'].isin(filtered_symbols)]
    
    def get_description(self) -> str:
        """Get filter description"""
        return (f"Smart Delivery-Volume Filter "
                f"(Del>{self.min_delivery_percent}%, {self.volume_filter.get_description()})")


# Factory function for easy creation
def create_smart_filter(
    mode: str = 'normal',
    index: Optional[str] = None,
    lookback_days: int = 20,
    with_delivery: bool = False
) -> IFilterStrategy:
    """
    Factory function to create smart filters
    
    Args:
        mode: 'conservative', 'normal', or 'aggressive'
        index: Index being filtered (for multipliers)
        lookback_days: Days for average calculation
        with_delivery: Include delivery correlation
        
    Returns:
        Configured filter instance
    """
    volume_filter = SmartVolumeFilter(
        mode=mode,
        index_filter=index,
        lookback_days=lookback_days
    )
    
    if with_delivery:
        return SmartDeliveryVolumeFilter(
            volume_filter=volume_filter,
            min_delivery_percent=40.0 if mode == 'normal' else 50.0 if mode == 'conservative' else 30.0
        )
    
    return volume_filter