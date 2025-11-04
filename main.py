"""
NSE Delivery Tracker Main Application - Refactored
Entry point for the delivery spike detection system
"""

import asyncio
import click
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Optional, List
from loguru import logger
import sys
import pandas as pd

# Configure logging
logger.remove()
logger.add(
    sys.stdout,
    colorize=True,
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level}</level> | <cyan>{message}</cyan>"
)
logger.add(
    "logs/nse_tracker_{time}.log",
    rotation="1 day",
    retention="30 days",
    level="DEBUG"
)

# Import components
from src.config.manager import ConfigurationManager, AppConfig
from src.data.fetcher import NSEDataFetcherFactory, DataSource
from src.data.repository import DataRepositoryFactory
from src.analysis.engine import DeliveryAnalysisEngine, AdvancedSpikeDetector
from src.analysis.filters import (
    FilterFactory, VolumeFilter, DeliveryPercentFilter, 
    IndexFilter, PriceRangeFilter, CompositeFilter
)
from src.reporting.excel_generator import ExcelReportGenerator
from src.interfaces import IDataFetcher, IDataRepository, IAnalysisEngine, IReportGenerator


class NSEDeliveryTracker:
    """
    Main application class
    Follows Dependency Injection pattern
    """
    
    def __init__(self, 
                 config: AppConfig,
                 data_fetcher: IDataFetcher,
                 data_repository: IDataRepository,
                 analysis_engine: IAnalysisEngine,
                 report_generator: IReportGenerator):
        """
        Initialize the tracker with injected dependencies
        
        Args:
            config: Application configuration
            data_fetcher: Data fetching implementation
            data_repository: Data storage implementation
            analysis_engine: Analysis engine implementation
            report_generator: Report generation implementation
        """
        self.config = config
        self.data_fetcher = data_fetcher
        self.data_repository = data_repository
        self.analysis_engine = analysis_engine
        self.report_generator = report_generator
        
        logger.info("NSE Delivery Tracker initialized")
    
    async def run_analysis(self, 
                          analysis_date: Optional[date] = None,
                          fetch_historical: bool = True) -> dict:
        """
        Run the complete analysis pipeline
        
        Args:
            analysis_date: Date to analyze (default: today)
            fetch_historical: Whether to fetch historical data
            
        Returns:
            Analysis results dictionary
        """
        if analysis_date is None:
            analysis_date = date.today()
            # If today is weekend, use last Friday
            if analysis_date.weekday() >= 5:
                days_to_friday = (analysis_date.weekday() - 4) % 7
                analysis_date = analysis_date - timedelta(days=days_to_friday)
        
        logger.info(f"Starting analysis for {analysis_date}")
        
        try:
            # Step 1: Fetch or load data
            data = await self._fetch_or_load_data(analysis_date, fetch_historical)
            
            if data.empty:
                logger.error("No data available for analysis")
                return {}
            
            # Step 2: Detect spikes
            logger.info("Detecting delivery spikes...")
            spikes = self.analysis_engine.detect_spikes(
                data,
                self.config.lookback_days,
                self.config.spike_multiplier
            )
            
            logger.info(f"Found {len(spikes)} delivery spikes")
            
            # Step 3: Analyze trends for spike stocks
            logger.info("Analyzing trends...")
            trends = {}
            for spike in spikes[:20]:  # Analyze top 20 spikes
                trend_analysis = self.analysis_engine.analyze_trends(
                    spike.symbol,
                    data,
                    window=5
                )
                trends[spike.symbol] = trend_analysis
            
            # Step 4: Calculate statistics
            logger.info("Calculating statistics...")
            statistics = self._calculate_statistics(data, spikes)
            
            # Step 5: Prepare results
            results = {
                'analysis_date': analysis_date,
                'total_stocks': len(data['symbol'].unique()) if 'symbol' in data.columns else 0,
                'spikes': spikes,
                'trends': trends,
                'statistics': statistics,
                'raw_data': data[data['date'] == analysis_date] if 'date' in data.columns else data,
                'lookback_days': self.config.lookback_days,
                'spike_multiplier': self.config.spike_multiplier,
                'filters_description': self._get_filters_description()
            }
            
            logger.info("Analysis completed successfully")
            return results
            
        except Exception as e:
            logger.error(f"Error during analysis: {e}")
            raise
    
    async def _fetch_or_load_data(self, 
                                 analysis_date: date, 
                                 fetch_historical: bool) -> pd.DataFrame:
        """
        Fetch or load data for analysis
        
        Args:
            analysis_date: Date to analyze
            fetch_historical: Whether to fetch historical data
            
        Returns:
            Combined DataFrame with historical data
        """
        # Calculate date range
        start_date = analysis_date - timedelta(days=self.config.lookback_days + 30)
        end_date = analysis_date
        
        all_data = []
        
        # Check what data we already have
        existing_dates = self.data_repository.get_available_dates()
        
        # Determine dates to fetch
        current_date = start_date
        dates_to_fetch = []
        
        while current_date <= end_date:
            if current_date.weekday() < 5:  # Weekday
                if current_date not in existing_dates and fetch_historical:
                    dates_to_fetch.append(current_date)
                elif current_date in existing_dates:
                    # Load from repository
                    data = self.data_repository.load(current_date)
                    if data is not None:
                        data['date'] = current_date
                        all_data.append(data)
            
            current_date += timedelta(days=1)
        
        # Fetch missing data
        if dates_to_fetch:
            logger.info(f"Fetching data for {len(dates_to_fetch)} days...")
            
            for fetch_date in dates_to_fetch[-self.config.lookback_days-5:]:  # Limit fetching
                logger.info(f"Fetching {fetch_date}...")
                daily_data = await self.data_fetcher.fetch_daily_data(fetch_date)
                
                if not daily_data.empty:
                    # Save to repository
                    self.data_repository.save(daily_data, fetch_date)
                    daily_data['date'] = fetch_date
                    all_data.append(daily_data)
                
                # Rate limiting
                await asyncio.sleep(self.config.rate_limit_delay)
        
        # Combine all data
        if all_data:
            combined_data = pd.concat(all_data, ignore_index=True)
            logger.info(f"Loaded {len(combined_data)} records for analysis")
            return combined_data
        
        return pd.DataFrame()
    
    def _calculate_statistics(self, data: pd.DataFrame, spikes: list) -> dict:
        """Calculate statistical measures"""
        latest_data = data[data['date'] == data['date'].max()] if 'date' in data.columns else data
        
        statistics = {
            'market': {
                'total_volume': latest_data['volume'].sum() if 'volume' in latest_data.columns else 0,
                'total_delivery': latest_data['delivery_qty'].sum() if 'delivery_qty' in latest_data.columns else 0,
                'avg_delivery_percent': latest_data['delivery_percent'].mean() if 'delivery_percent' in latest_data.columns else 0,
                'stocks_above_avg': len(latest_data[latest_data['delivery_percent'] > latest_data['delivery_percent'].mean()]) if 'delivery_percent' in latest_data.columns else 0
            }
        }
        
        if 'delivery_qty' in latest_data.columns:
            delivery_data = latest_data['delivery_qty'].dropna()
            statistics['distribution'] = {
                'mean': delivery_data.mean(),
                'median': delivery_data.median(),
                'std': delivery_data.std(),
                'q25': delivery_data.quantile(0.25),
                'q75': delivery_data.quantile(0.75),
                'skewness': delivery_data.skew(),
                'kurtosis': delivery_data.kurtosis()
            }
        
        # Highest and lowest delivery stocks
        if 'delivery_percent' in latest_data.columns:
            sorted_by_delivery = latest_data.sort_values('delivery_percent', ascending=False)
            if not sorted_by_delivery.empty:
                statistics['market']['highest_delivery_stock'] = sorted_by_delivery.iloc[0]['symbol'] if 'symbol' in sorted_by_delivery.columns else 'N/A'
                statistics['market']['lowest_delivery_stock'] = sorted_by_delivery.iloc[-1]['symbol'] if 'symbol' in sorted_by_delivery.columns else 'N/A'
        
        return statistics
    
    def _get_filters_description(self) -> str:
        """Get description of applied filters"""
        if hasattr(self.analysis_engine, 'filters') and self.analysis_engine.filters:
            descriptions = [f.get_description() for f in self.analysis_engine.filters]
            return " | ".join(descriptions)
        return "None"
    
    def generate_report(self, analysis_results: dict, output_path: Optional[str] = None) -> bool:
        """
        Generate report from analysis results
        
        Args:
            analysis_results: Analysis results dictionary
            output_path: Path to save report
            
        Returns:
            True if successful
        """
        if not output_path:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = f"{self.config.reports_directory}/nse_delivery_report_{timestamp}.xlsx"
        
        logger.info(f"Generating report: {output_path}")
        
        return self.report_generator.generate(analysis_results, output_path)


def create_filters(config: AppConfig) -> List[IFilterStrategy]:
    """
    Create filter list based on configuration
    Single place to create filters - no duplication
    
    Args:
        config: Application configuration
        
    Returns:
        List of filter strategies
    """
    filters = []
    
    if config.min_volume > 0:
        filters.append(VolumeFilter(config.min_volume))
        
    if config.min_delivery_percent > 0:
        filters.append(DeliveryPercentFilter(config.min_delivery_percent))
        
    if config.index_filter and config.index_filter != "ALL":
        filters.append(IndexFilter(config.index_filter))
        
    if config.price_range:
        filters.append(PriceRangeFilter(
            config.price_range.get("min", 0),
            config.price_range.get("max", float('inf'))
        ))
    
    if config.sectors:
        from src.analysis.filters import SectorFilter
        filters.append(SectorFilter(config.sectors))
    
    return filters


def create_app(config: AppConfig) -> NSEDeliveryTracker:
    """
    Factory function to create the application with all dependencies
    Now takes AppConfig directly instead of loading it
    
    Args:
        config: Application configuration object
        
    Returns:
        Configured NSEDeliveryTracker instance
    """
    # # Create data fetcher
    # data_fetcher = NSEDataFetcherFactory.create(
    #     DataSource[config.data_source],
    #     cache=None
    # )

    # CORRECT - Use the working jugaad-data fetcher
    from src.data.fetcher_delivery import DeliveryDataFetcher
    data_fetcher = DeliveryDataFetcher(cache_dir=config.data_directory)
    
    # Create data repository
    data_repository = DataRepositoryFactory.create(
        config.storage_type,
        base_path=config.data_directory
    )
    
    # Create filters using the centralized function
    filters = create_filters(config)
    
    # Create analysis engine
    if config.debug_mode:
        analysis_engine = AdvancedSpikeDetector(z_score_threshold=3.0)
    else:
        analysis_engine = DeliveryAnalysisEngine(filters=filters)
    
    # Create report generator
    report_generator = ExcelReportGenerator()
    
    # Create and return application
    return NSEDeliveryTracker(
        config=config,
        data_fetcher=data_fetcher,
        data_repository=data_repository,
        analysis_engine=analysis_engine,
        report_generator=report_generator
    )


@click.command()
@click.option('--date', '-d', type=click.DateTime(formats=['%Y-%m-%d']), 
              help='Analysis date (YYYY-MM-DD)')
@click.option('--lookback', '-l', type=int, help='Lookback days for average calculation')
@click.option('--multiplier', '-m', type=float, help='Spike multiplier threshold')
@click.option('--index', '-i', 
              type=click.Choice(['ALL', 'LARGECAP', 'MIDCAP', 'SMALLCAP',
                                'NIFTY_50', 'NIFTY_NEXT_50', 'NIFTY_100', 
                                'NIFTY_500', 'NIFTY_MIDCAP_150', 'NIFTY_SMALLCAP_250',
                                'NIFTY_BANK', 'NIFTY_IT', 'NIFTY_PHARMA', 'NIFTY_AUTO',
                                'NIFTY_FMCG', 'NIFTY_METAL', 'NIFTY_REALTY', 'NIFTY_ENERGY']), 
              help='Index filter')
@click.option('--config', '-c', default='config.yaml', help='Configuration file path')
@click.option('--output', '-o', help='Output report path')
@click.option('--no-fetch', is_flag=True, help='Use only existing data, do not fetch new data')
@click.option('--debug', is_flag=True, help='Enable debug mode')
def main(date, lookback, multiplier, index, config, output, no_fetch, debug):
    """
    NSE Delivery Tracker - Detect unusual delivery spikes in NSE stocks
    """
    # Enable debug logging if requested
    if debug:
        logger.remove()
        logger.add(sys.stdout, level="DEBUG")
    
    # Load configuration
    config_manager = ConfigurationManager(config)
    app_config = config_manager.get_config()
    
    # Override configuration with CLI arguments
    # This happens BEFORE creating the app
    if lookback:
        app_config.lookback_days = lookback
    if multiplier:
        app_config.spike_multiplier = multiplier
    if index:
        app_config.index_filter = index
    
    # Now create app with the updated configuration
    app = create_app(app_config)
    
    # Parse date
    analysis_date = date.date() if date else None
    
    # Run analysis
    logger.info("Starting NSE Delivery Tracker...")
    
    try:
        # Run async analysis
        results = asyncio.run(app.run_analysis(
            analysis_date=analysis_date,
            fetch_historical=not no_fetch
        ))
        
        if results:
            # Generate report
            success = app.generate_report(results, output)
            
            if success:
                logger.success("Analysis completed successfully!")
                
                # Print summary
                spikes = results.get('spikes', [])
                if spikes:
                    click.echo("\n" + "="*50)
                    click.echo("TOP 10 DELIVERY SPIKES")
                    click.echo("="*50)
                    
                    for i, spike in enumerate(spikes[:10], 1):
                        click.echo(f"{i}. {spike.symbol}: {spike.spike_ratio:.1f}x spike "
                                 f"(Delivery: {spike.current_delivery:,} vs Avg: {spike.avg_delivery:,.0f})")
                    
                    click.echo("="*50)
                else:
                    click.echo("No delivery spikes found matching the criteria.")
            else:
                logger.error("Failed to generate report")
                sys.exit(1)
        else:
            logger.error("No analysis results generated")
            sys.exit(1)
            
    except Exception as e:
        logger.error(f"Application error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()