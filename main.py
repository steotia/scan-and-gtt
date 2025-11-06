"""
NSE Delivery Tracker Main Application
Entry point for the delivery spike detection system
WITH SMART VOLUME FILTER INTEGRATION
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
from src.data.fetcher_delivery import DeliveryDataFetcher
from src.data.repository import DataRepositoryFactory
from src.analysis.engine import DeliveryAnalysisEngine, AdvancedSpikeDetector
from src.analysis.filters import (
    VolumeFilter, DeliveryPercentFilter, 
    IndexFilter, PriceRangeFilter, CompositeFilter, SectorFilter
)
from src.analysis.smart_volume_filter import (
    SmartVolumeFilter,
    SmartDeliveryVolumeFilter,
    create_smart_filter
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


def create_filters(config: AppConfig, mode: str = 'normal') -> List:
    """
    Create filter list based on configuration
    Single place to create filters - no duplication
    Now includes Smart Volume Filter
    
    Args:
        config: Application configuration
        mode: Filter mode ('conservative', 'normal', 'aggressive')
        
    Returns:
        List of filter strategies
    """
    filters = []
    
    # Smart Volume Filter (NEW - Add this first for efficiency)
    if hasattr(config, 'use_smart_volume_filter') and config.use_smart_volume_filter:
        smart_filter = SmartVolumeFilter(
            mode=mode,
            index_filter=config.index_filter if hasattr(config, 'index_filter') else None,
            lookback_days=config.lookback_days,
            enable_logging=config.debug_mode if hasattr(config, 'debug_mode') else False
        )
        filters.append(smart_filter)
        logger.info(f"Added Smart Volume Filter (mode: {mode})")
    elif config.min_volume > 0:
        # Only use simple volume filter if smart filter is disabled
        filters.append(VolumeFilter(config.min_volume))
        logger.info(f"Added Simple Volume Filter (min: {config.min_volume})")
    
    if config.min_delivery_percent > 0:
        filters.append(DeliveryPercentFilter(config.min_delivery_percent))
        
    if hasattr(config, 'index_filter') and config.index_filter and config.index_filter != "ALL":
        filters.append(IndexFilter(config.index_filter))
        
    if hasattr(config, 'price_range') and config.price_range:
        filters.append(PriceRangeFilter(
            config.price_range.get("min", 0),
            config.price_range.get("max", float('inf'))
        ))
    
    if hasattr(config, 'sectors') and config.sectors:
        filters.append(SectorFilter(config.sectors))
    
    # Optional: Combined Delivery-Volume Filter
    if hasattr(config, 'use_delivery_volume_correlation') and config.use_delivery_volume_correlation:
        delivery_volume_filter = SmartDeliveryVolumeFilter(
            volume_filter=smart_filter if hasattr(config, 'use_smart_volume_filter') and config.use_smart_volume_filter else None,
            min_delivery_percent=config.min_delivery_percent
        )
        filters.append(delivery_volume_filter)
        logger.info("Added Delivery-Volume Correlation Filter")
    
    return filters


def create_app(config: AppConfig, mode: str = 'normal') -> NSEDeliveryTracker:
    """
    Factory function to create the application with all dependencies
    Now takes AppConfig directly and supports filter modes
    
    Args:
        config: Application configuration object
        mode: Filter mode ('conservative', 'normal', 'aggressive')
        
    Returns:
        Configured NSEDeliveryTracker instance
    """
    # Create data fetcher using the working delivery fetcher
    data_fetcher = DeliveryDataFetcher(cache_dir=config.data_directory)
    
    # Create data repository
    data_repository = DataRepositoryFactory.create(
        config.storage_type,
        base_path=config.data_directory
    )
    
    # Create filters using the centralized function with mode
    filters = create_filters(config, mode)
    
    # Create analysis engine
    if hasattr(config, 'debug_mode') and config.debug_mode:
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

def print_analysis_dashboard(results: dict, app):
    """
    Print comprehensive analysis dashboard with spikes and filter details
    """
    try:
        from tabulate import tabulate
    except ImportError:
        # Fallback if tabulate not installed
        tabulate = None
    import pandas as pd
    
    # ANSI color codes for terminal
    RESET = "\033[0m"
    BOLD = "\033[1m"
    RED = "\033[91m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    BLUE = "\033[94m"
    CYAN = "\033[96m"
    
    print("\n" + "="*100)
    print(f"{BOLD}{CYAN}📊 NSE DELIVERY ANALYSIS DASHBOARD{RESET}")
    print("="*100)
    
    # 1. Summary Statistics
    print(f"\n{BOLD}📈 SUMMARY STATISTICS{RESET}")
    print("-"*50)
    
    analysis_date = results.get('analysis_date', 'N/A')
    total_stocks = results.get('total_stocks', 0)
    spikes = results.get('spikes', [])
    spike_multiplier = results.get('spike_multiplier', 5.0)
    lookback_days = results.get('lookback_days', 20)
    filters_desc = results.get('filters_description', 'None')
    
    summary_data = [
        ["Analysis Date", analysis_date],
        ["Total Stocks Analyzed", total_stocks],
        ["Delivery Spikes Found", len(spikes)],
        ["Spike Threshold", f"{spike_multiplier}x"],
        ["Lookback Period", f"{lookback_days} days"],
        ["Active Filters", filters_desc]
    ]
    
    for item in summary_data:
        print(f"  {item[0]:25} : {BOLD}{item[1]}{RESET}")
    
    # 2. Smart Volume Filter Metrics (if available)
    if hasattr(app.analysis_engine, 'filters'):
        for filter_obj in app.analysis_engine.filters:
            if hasattr(filter_obj, 'get_metrics'):  # SmartVolumeFilter
                print(f"\n{BOLD}🎯 SMART VOLUME FILTER METRICS{RESET}")
                print("-"*50)
                
                try:
                    metrics_df = filter_obj.get_metrics()
                    if not metrics_df.empty:
                        # Show passed stocks
                        passed = metrics_df[metrics_df['passed'] == True].head(10)
                        if not passed.empty:
                            print(f"\n{GREEN}✅ Top Stocks Passing Volume Filter:{RESET}")
                            if tabulate:
                                # Fixed: Use proper float formatting
                                table_data = []
                                for _, row in passed.iterrows():
                                    table_data.append([
                                        row['symbol'],
                                        f"{row['price']:.2f}",
                                        f"{row['turnover_cr']:.2f}",
                                        f"{row['rvol']:.2f}",
                                        row['bracket']
                                    ])
                                print(tabulate(
                                    table_data,
                                    headers=['Symbol', 'Price', 'Turnover(Cr)', 'RVOL', 'Bracket'],
                                    tablefmt='simple'
                                ))
                            else:
                                # Fallback without tabulate
                                for _, row in passed.iterrows():
                                    print(f"  {row['symbol']:12} | Price: ₹{row['price']:8.2f} | "
                                          f"TO: ₹{row['turnover_cr']:6.2f}Cr | RVOL: {row['rvol']:.2f}x | "
                                          f"{row['bracket']}")
                        
                        # Show failed stocks
                        failed = metrics_df[metrics_df['passed'] == False].head(5)
                        if not failed.empty:
                            print(f"\n{RED}❌ Sample Stocks Filtered Out:{RESET}")
                            if tabulate:
                                table_data = []
                                for _, row in failed.iterrows():
                                    table_data.append([
                                        row['symbol'],
                                        f"{row['price']:.2f}",
                                        f"{row['turnover_cr']:.2f}",
                                        row['reason'][:50]  # Truncate long reasons
                                    ])
                                print(tabulate(
                                    table_data,
                                    headers=['Symbol', 'Price', 'Turnover(Cr)', 'Reason'],
                                    tablefmt='simple'
                                ))
                            else:
                                for _, row in failed.iterrows():
                                    print(f"  {row['symbol']:12} | ₹{row['price']:8.2f} | {row['reason'][:50]}")
                    
                    # Get statistics
                    if hasattr(filter_obj, 'get_statistics'):
                        stats = filter_obj.get_statistics()
                        if stats:
                            print(f"\n{YELLOW}📊 Filter Statistics:{RESET}")
                            print(f"  Total Stocks: {stats.get('total_stocks', 0)}")
                            print(f"  Passed: {stats.get('passed', 0)}")
                            print(f"  Failed: {stats.get('failed', 0)}")
                            print(f"  Pass Rate: {stats.get('pass_rate', 0):.1f}%")
                            print(f"  Avg RVOL (Passed): {stats.get('avg_rvol_passed', 0):.2f}x")
                            print(f"  Total Turnover (Passed): ₹{stats.get('total_turnover_passed_cr', 0):.1f} Cr")
                            
                            # Show failure reasons
                            if 'failure_reasons' in stats and stats['failure_reasons']:
                                print(f"\n{YELLOW}Failure Reasons:{RESET}")
                                for reason, count in list(stats['failure_reasons'].items())[:5]:
                                    print(f"  • {reason}: {count} stocks")
                except Exception as e:
                    print(f"  Error displaying filter metrics: {e}")
    
    # 3. Top Delivery Spikes
    print(f"\n{BOLD}🚀 TOP DELIVERY SPIKES{RESET}")
    print("-"*50)
    
    if spikes:
        if tabulate:
            spike_data = []
            for i, spike in enumerate(spikes[:15], 1):
                spike_data.append([
                    f"{i}",
                    spike.symbol,
                    f"{spike.spike_ratio:.1f}x",
                    f"{spike.current_delivery:,}",
                    f"{spike.avg_delivery:,.0f}",
                    f"{spike.price_change:+.1f}%",
                    f"{spike.volume_change:+.1f}%"
                ])
            
            headers = ["#", "Symbol", "Spike", "Current Del", "Avg Del", "Price Δ", "Volume Δ"]
            print(tabulate(spike_data, headers=headers, tablefmt='simple'))
        else:
            # Fallback without tabulate
            for i, spike in enumerate(spikes[:15], 1):
                print(f"{i:2}. {spike.symbol:12} | Spike: {spike.spike_ratio:5.1f}x | "
                      f"Del: {spike.current_delivery:10,} | Avg: {spike.avg_delivery:10,.0f}")
        
        # Highlight extreme spikes
        extreme_spikes = [s for s in spikes if s.spike_ratio >= 10]
        if extreme_spikes:
            print(f"\n{RED}{BOLD}🔥 EXTREME SPIKES (>10x):{RESET}")
            for spike in extreme_spikes[:5]:
                print(f"  • {spike.symbol}: {spike.spike_ratio:.1f}x spike!")
    else:
        print(f"{YELLOW}No delivery spikes found with current filters.{RESET}")
        print("\nPossible reasons:")
        print("  • Spike multiplier too high (try --multiplier 2 or 3)")
        print("  • Smart volume filter too strict (try --mode aggressive)")
        print("  • Not enough historical data for comparison")
        print("  • Try disabling smart volume: --no-smart-volume")
    
    # 4. Market Overview (if raw data available)
    raw_data = results.get('raw_data')
    if raw_data is not None and not raw_data.empty:
        print(f"\n{BOLD}🌍 MARKET OVERVIEW{RESET}")
        print("-"*50)
        
        print(f"Latest Date Stocks: {len(raw_data)}")
        
        if 'delivery_percent' in raw_data.columns:
            # Top delivery stocks
            top_delivery = raw_data.nlargest(5, 'delivery_percent')[['symbol', 'delivery_qty', 'delivery_percent']]
            print(f"\n{CYAN}Highest Delivery % Stocks:{RESET}")
            for _, row in top_delivery.iterrows():
                print(f"  • {row['symbol']:12} : {row['delivery_qty']:12,} ({row['delivery_percent']:.1f}%)")
    
    # 5. Debug Information
    print(f"\n{BOLD}🔍 DEBUG INFORMATION{RESET}")
    print("-"*50)
    print(f"Total records loaded: {results.get('total_stocks', 0)}")
    print(f"After filters: Check logs above")
    print(f"Date being analyzed: {results.get('analysis_date', 'Unknown')}")
    
    # 6. Recommendations
    print(f"\n{BOLD}💡 RECOMMENDATIONS{RESET}")
    print("-"*50)
    
    if len(spikes) == 0:
        print(f"{YELLOW}No spikes found. Try these commands:{RESET}")
        print(f"  1. Lower multiplier: python main.py --date {results.get('analysis_date')} --multiplier 2")
        print(f"  2. Aggressive mode: python main.py --date {results.get('analysis_date')} --mode aggressive --multiplier 3")
        print(f"  3. No smart filter: python main.py --date {results.get('analysis_date')} --no-smart-volume --multiplier 2")
        print(f"  4. All stocks, minimal filter: python main.py --date {results.get('analysis_date')} --index ALL --no-smart-volume --multiplier 2")
    elif len(spikes) < 5:
        print(f"{YELLOW}Few spikes found. To find more:{RESET}")
        print("  1. Lower spike threshold: --multiplier 3")
        print("  2. Use aggressive mode: --mode aggressive")
    else:
        print(f"{GREEN}Good number of spikes found!{RESET}")
        print("  Review top 5 for opportunities")
    
    print("\n" + "="*100)
    print(f"{BOLD}📝 Report: {results.get('report_path', 'reports/')}{RESET}")
    print("="*100 + "\n")


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
@click.option('--mode', type=click.Choice(['conservative', 'normal', 'aggressive']),
              default='normal', help='Filter mode for smart volume filter')
@click.option('--smart-volume/--no-smart-volume', default=True,
              help='Use smart volume filter (default: enabled)')
@click.option('--config', '-c', default='config.yaml', help='Configuration file path')
@click.option('--output', '-o', help='Output report path')
@click.option('--no-fetch', is_flag=True, help='Use only existing data, do not fetch new data')
@click.option('--debug', is_flag=True, help='Enable debug mode')
def main(date, lookback, multiplier, index, mode, smart_volume, config, output, no_fetch, debug):
    """
    NSE Delivery Tracker - Detect unusual delivery spikes in NSE stocks
    Now with Smart Volume Filtering and Enhanced Dashboard
    """
    # Enable debug logging if requested
    if debug:
        logger.remove()
        logger.add(sys.stdout, level="DEBUG")
    
    # Load configuration
    config_manager = ConfigurationManager(config)
    app_config = config_manager.get_config()
    
    # Override configuration with CLI arguments
    if lookback:
        app_config.lookback_days = lookback
    if multiplier:
        app_config.spike_multiplier = multiplier
    if index:
        app_config.index_filter = index
    
    # Add smart volume filter settings
    app_config.use_smart_volume_filter = smart_volume
    app_config.filter_mode = mode
    app_config.debug_mode = debug
    
    # Log configuration
    logger.info(f"Configuration: Mode={mode}, Smart Volume={smart_volume}, "
                f"Multiplier={app_config.spike_multiplier}, Index={app_config.index_filter}")
    
    # Create app with the updated configuration and mode
    app = create_app(app_config, mode)
    
    # Parse date
    analysis_date = date.date() if date else None
    
    # Run analysis
    logger.info("Starting NSE Delivery Tracker...")
    logger.info(f"Mode: {mode}, Smart Volume: {'Enabled' if smart_volume else 'Disabled'}")
    
    try:
        # Run async analysis
        results = asyncio.run(app.run_analysis(
            analysis_date=analysis_date,
            fetch_historical=not no_fetch
        ))
        
        if results:
            # Store report path in results for dashboard
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            if not output:
                output = f"{app_config.reports_directory}/nse_delivery_report_{timestamp}.xlsx"
            results['report_path'] = output
            
            # Generate Excel report
            success = app.generate_report(results, output)
            
            if success:
                logger.success("Analysis completed successfully!")
                
                # Print the enhanced dashboard
                print_analysis_dashboard(results, app)
                
                # Also print simple summary if no dashboard
                spikes = results.get('spikes', [])
                if not spikes:
                    print("\n" + "="*60)
                    print("ℹ️  NO SPIKES FOUND - TROUBLESHOOTING TIPS")
                    print("="*60)
                    print(f"Current settings:")
                    print(f"  - Spike Multiplier: {app_config.spike_multiplier}x")
                    print(f"  - Mode: {mode}")
                    print(f"  - Smart Volume: {smart_volume}")
                    print(f"  - Index Filter: {app_config.index_filter}")
                    print(f"\nTry these commands:")
                    print(f"  1. Lower threshold: python main.py --date {analysis_date} --multiplier 2")
                    print(f"  2. Aggressive mode: python main.py --date {analysis_date} --mode aggressive --multiplier 3")
                    print(f"  3. No filters: python main.py --date {analysis_date} --no-smart-volume --multiplier 2")
                    print(f"  4. All stocks: python main.py --date {analysis_date} --index ALL --multiplier 2")
                    print("="*60)
            else:
                logger.error("Failed to generate report")
                sys.exit(1)
        else:
            logger.error("No analysis results generated")
            logger.info("Check if data exists for the selected date")
            sys.exit(1)
            
    except Exception as e:
        logger.error(f"Application error: {e}")
        if debug:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
