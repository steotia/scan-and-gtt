#!/usr/bin/env python
"""
Export ALL stocks with smart filter analysis and Market Cap Classification
Shows what the smart filter is doing for every stock with Large/Mid/Small cap info
"""

import pandas as pd
import asyncio
from datetime import date, datetime
from pathlib import Path
import xlsxwriter
from main import create_app
from src.config.manager import ConfigurationManager
from src.analysis.smart_volume_filter import SmartVolumeFilter
from src.analysis.filters import IndexFilter


def get_market_cap_category(symbol):
    """
    Determine if a stock is Large Cap, Mid Cap, or Small Cap
    Returns: (category, index_list)
    """
    # Check Large Cap
    if symbol in IndexFilter.NIFTY_50:
        return 'LARGE CAP', 'NIFTY 50'
    elif symbol in IndexFilter.NIFTY_NEXT_50:
        return 'LARGE CAP', 'NIFTY NEXT 50'
    
    # Check Mid Cap
    elif symbol in IndexFilter.NIFTY_MIDCAP_150:
        return 'MID CAP', 'NIFTY MIDCAP 150'
    
    # Check Small Cap
    elif symbol in IndexFilter.NIFTY_SMALLCAP_250:
        return 'SMALL CAP', 'NIFTY SMALLCAP 250'
    
    # Check Sectoral indices for unclassified
    else:
        indices = []
        if symbol in IndexFilter.NIFTY_BANK:
            indices.append('NIFTY BANK')
        if symbol in IndexFilter.NIFTY_IT:
            indices.append('NIFTY IT')
        if symbol in IndexFilter.NIFTY_PHARMA:
            indices.append('NIFTY PHARMA')
        if symbol in IndexFilter.NIFTY_AUTO:
            indices.append('NIFTY AUTO')
        if symbol in IndexFilter.NIFTY_FMCG:
            indices.append('NIFTY FMCG')
        if symbol in IndexFilter.NIFTY_METAL:
            indices.append('NIFTY METAL')
        if symbol in IndexFilter.NIFTY_REALTY:
            indices.append('NIFTY REALTY')
        if symbol in IndexFilter.NIFTY_ENERGY:
            indices.append('NIFTY ENERGY')
        
        if indices:
            # Try to determine based on common knowledge
            if symbol in ['HDFCBANK', 'ICICIBANK', 'KOTAKBANK', 'AXISBANK', 'SBIN']:
                return 'LARGE CAP', ', '.join(indices)
            else:
                return 'MID CAP', ', '.join(indices)
        
        # Default to MICRO/UNKNOWN
        return 'MICRO CAP', 'UNLISTED/OTHER'


async def analyze_all_stocks_with_filter(analysis_date=None, output_file=None):
    """
    Analyze ALL stocks and show smart filter decisions with Market Cap
    """
    
    if analysis_date is None:
        analysis_date = date(2025, 11, 4)
    
    if output_file is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f"smart_filter_analysis_with_mcap_{timestamp}.xlsx"
    
    print("="*80)
    print(f"SMART FILTER ANALYSIS WITH MARKET CAP - {analysis_date}")
    print("="*80)
    
    # Load configuration
    config_manager = ConfigurationManager("config.yaml")
    app_config = config_manager.get_config()
    
    # Create app WITHOUT filters first to get all data
    app_config.min_volume = 0
    app_config.min_delivery_percent = 0
    app_config.index_filter = "ALL"
    
    app = create_app(app_config, "normal")
    
    # Fetch all data
    print("\n1. Fetching all data (no filters)...")
    all_data = await app._fetch_or_load_data(analysis_date, fetch_historical=False)
    
    if all_data.empty:
        print("❌ No data found!")
        return
    
    # Get latest date data
    latest_date = all_data['date'].max()
    latest_data = all_data[all_data['date'] == latest_date].copy()
    
    print(f"   ✓ Found {len(latest_data)} stocks for {latest_date}")
    
    # Add market cap classification
    print("\n2. Adding Market Cap Classification...")
    for idx, row in latest_data.iterrows():
        mcap_category, index_membership = get_market_cap_category(row['symbol'])
        latest_data.at[idx, 'market_cap'] = mcap_category
        latest_data.at[idx, 'index_membership'] = index_membership
    
    # Count by market cap
    mcap_counts = latest_data['market_cap'].value_counts()
    print("\n   Market Cap Distribution:")
    for mcap, count in mcap_counts.items():
        print(f"   • {mcap:12}: {count:4} stocks")
    
    # Now apply smart filter analysis to each stock
    print("\n3. Analyzing each stock with Smart Volume Filter...")
    
    # Create smart filters with different modes
    filters = {
        'conservative': SmartVolumeFilter(mode='conservative', lookback_days=20, enable_logging=False),
        'normal': SmartVolumeFilter(mode='normal', lookback_days=20, enable_logging=False),
        'aggressive': SmartVolumeFilter(mode='aggressive', lookback_days=20, enable_logging=False)
    }
    
    # Analyze each stock
    analysis_results = []
    
    for filter_mode, smart_filter in filters.items():
        print(f"\n   Analyzing with {filter_mode} mode...")
        
        # Apply filter to get metrics
        filtered = smart_filter.filter(all_data)
        
        # Get the detailed metrics
        if smart_filter.last_metrics:
            for metric in smart_filter.last_metrics:
                # Find the stock in latest data to get all info
                stock_data = latest_data[latest_data['symbol'] == metric.symbol]
                
                if not stock_data.empty:
                    stock = stock_data.iloc[0]
                    
                    analysis_results.append({
                        'mode': filter_mode,
                        'symbol': metric.symbol,
                        'market_cap': stock.get('market_cap', 'UNKNOWN'),
                        'index_membership': stock.get('index_membership', 'UNKNOWN'),
                        'close': stock.get('close', metric.current_price),
                        'volume': stock.get('volume', metric.current_volume),
                        'delivery_qty': stock.get('delivery_qty', 0),
                        'delivery_percent': stock.get('delivery_percent', 0),
                        'current_turnover_cr': metric.current_turnover / 1_00_00_000,
                        'avg_turnover_cr': metric.avg_turnover / 1_00_00_000,
                        'rvol': metric.relative_volume,
                        'turnover_ratio': metric.turnover_ratio,
                        'price_bracket': metric.bracket,
                        'passes_filter': metric.passes_filter,
                        'filter_reason': metric.failure_reason or 'PASSED',
                        'vwap': stock.get('vwap', stock.get('close', 0))
                    })
    
    # Convert to DataFrame
    results_df = pd.DataFrame(analysis_results)
    
    # Calculate spike ratios (need historical data)
    print("\n4. Calculating delivery spike ratios...")
    
    # Get historical data for spike calculation
    historical_data = all_data[all_data['date'] < latest_date]
    
    spike_data = []
    for symbol in latest_data['symbol'].unique():
        try:
            # Get market cap for this symbol
            mcap_info = latest_data[latest_data['symbol'] == symbol].iloc[0]
            market_cap = mcap_info.get('market_cap', 'UNKNOWN')
            index_membership = mcap_info.get('index_membership', 'UNKNOWN')
            
            # Current delivery
            current = latest_data[latest_data['symbol'] == symbol].iloc[0]
            current_delivery = current.get('delivery_qty', 0)
            
            # Historical average
            hist = historical_data[historical_data['symbol'] == symbol]
            if len(hist) >= 5 and current_delivery > 0:
                avg_delivery = hist['delivery_qty'].mean()
                if avg_delivery > 0:
                    spike_ratio = current_delivery / avg_delivery
                else:
                    spike_ratio = 0
            else:
                spike_ratio = 0
                avg_delivery = 0
            
            spike_data.append({
                'symbol': symbol,
                'market_cap': market_cap,
                'index_membership': index_membership,
                'current_delivery': current_delivery,
                'avg_delivery': avg_delivery,
                'spike_ratio': spike_ratio,
                'is_spike_5x': spike_ratio >= 5,
                'is_spike_3x': spike_ratio >= 3,
                'is_spike_2x': spike_ratio >= 2
            })
        except:
            continue
    
    spike_df = pd.DataFrame(spike_data)
    
    # Merge spike data with filter results
    if not results_df.empty and not spike_df.empty:
        # Drop market_cap columns from spike_df to avoid duplication
        spike_df_merge = spike_df.drop(['market_cap', 'index_membership'], axis=1)
        results_df = results_df.merge(spike_df_merge, on='symbol', how='left')
    
    # Create Excel with multiple sheets
    print(f"\n5. Creating Excel report: {output_file}")
    
    with pd.ExcelWriter(output_file, engine='xlsxwriter') as writer:
        workbook = writer.book
        
        # Format definitions
        header_format = workbook.add_format({
            'bold': True,
            'bg_color': '#4A90E2',
            'font_color': 'white',
            'border': 1
        })
        
        largecap_format = workbook.add_format({
            'bg_color': '#E8F5E9',  # Light green
            'border': 1
        })
        
        midcap_format = workbook.add_format({
            'bg_color': '#FFF3E0',  # Light orange
            'border': 1
        })
        
        smallcap_format = workbook.add_format({
            'bg_color': '#FCE4EC',  # Light pink
            'border': 1
        })
        
        # 1. Summary Sheet with Market Cap breakdown
        summary_data = []
        for mode in ['conservative', 'normal', 'aggressive']:
            mode_data = results_df[results_df['mode'] == mode]
            if not mode_data.empty:
                # Overall stats
                passed = mode_data['passes_filter'].sum()
                failed = len(mode_data) - passed
                pass_rate = (passed / len(mode_data)) * 100
                
                # Market cap breakdown
                for mcap in ['LARGE CAP', 'MID CAP', 'SMALL CAP', 'MICRO CAP']:
                    mcap_data = mode_data[mode_data['market_cap'] == mcap]
                    if not mcap_data.empty:
                        mcap_passed = mcap_data['passes_filter'].sum()
                        mcap_total = len(mcap_data)
                        mcap_pass_rate = (mcap_passed / mcap_total * 100) if mcap_total > 0 else 0
                        
                        # Spikes in this category
                        if 'is_spike_3x' in mcap_data.columns:
                            mcap_spikes = mcap_data['is_spike_3x'].sum()
                            mcap_spikes_passed = mcap_data[mcap_data['passes_filter'] & mcap_data['is_spike_3x']].shape[0]
                        else:
                            mcap_spikes = 0
                            mcap_spikes_passed = 0
                        
                        summary_data.append({
                            'Mode': mode.upper(),
                            'Market Cap': mcap,
                            'Total Stocks': mcap_total,
                            'Passed Filter': mcap_passed,
                            'Pass Rate %': mcap_pass_rate,
                            '3x Spikes': mcap_spikes,
                            '3x Spikes Passed': mcap_spikes_passed
                        })
        
        summary_df = pd.DataFrame(summary_data)
        summary_df.to_excel(writer, sheet_name='Summary by Market Cap', index=False)
        
        # 2. Normal Mode Results (Most Important)
        normal_df = results_df[results_df['mode'] == 'normal'].copy()
        if not normal_df.empty:
            # Sort by spike ratio, then market cap
            normal_df['mcap_order'] = normal_df['market_cap'].map({
                'LARGE CAP': 1, 'MID CAP': 2, 'SMALL CAP': 3, 'MICRO CAP': 4
            })
            normal_df.sort_values(['spike_ratio', 'mcap_order'], ascending=[False, True], inplace=True)
            normal_df.drop('mcap_order', axis=1, inplace=True)
            
            # Reorder columns for better readability
            column_order = ['symbol', 'market_cap', 'index_membership', 'spike_ratio', 'passes_filter', 
                           'close', 'volume', 'delivery_percent', 'current_turnover_cr', 
                           'rvol', 'price_bracket', 'filter_reason']
            
            # Only include columns that exist
            column_order = [col for col in column_order if col in normal_df.columns]
            other_cols = [col for col in normal_df.columns if col not in column_order]
            normal_df = normal_df[column_order + other_cols]
            
            normal_df.to_excel(writer, sheet_name='Normal Mode', index=False)
        
        # 3. Large Cap Analysis
        largecap_df = results_df[results_df['market_cap'] == 'LARGE CAP'].copy()
        if not largecap_df.empty:
            largecap_df.sort_values(['spike_ratio', 'mode'], ascending=[False, True], inplace=True)
            largecap_df.to_excel(writer, sheet_name='Large Cap Stocks', index=False)
        
        # 4. Mid Cap Analysis
        midcap_df = results_df[results_df['market_cap'] == 'MID CAP'].copy()
        if not midcap_df.empty:
            midcap_df.sort_values(['spike_ratio', 'mode'], ascending=[False, True], inplace=True)
            midcap_df.to_excel(writer, sheet_name='Mid Cap Stocks', index=False)
        
        # 5. Small Cap Analysis
        smallcap_df = results_df[results_df['market_cap'] == 'SMALL CAP'].copy()
        if not smallcap_df.empty:
            smallcap_df.sort_values(['spike_ratio', 'mode'], ascending=[False, True], inplace=True)
            smallcap_df.to_excel(writer, sheet_name='Small Cap Stocks', index=False)
        
        # 6. Top Spikes by Market Cap
        if not spike_df.empty:
            spike_df.sort_values('spike_ratio', ascending=False, inplace=True)
            
            # Add which modes passed the filter
            for idx, row in spike_df.iterrows():
                symbol = row['symbol']
                passes_modes = []
                for mode in ['conservative', 'normal', 'aggressive']:
                    mode_data = results_df[(results_df['symbol'] == symbol) & (results_df['mode'] == mode)]
                    if not mode_data.empty and mode_data.iloc[0]['passes_filter']:
                        passes_modes.append(mode[0].upper())  # C/N/A
                spike_df.at[idx, 'passes_modes'] = ''.join(passes_modes) if passes_modes else 'NONE'
            
            # Reorder columns
            spike_cols = ['symbol', 'market_cap', 'index_membership', 'spike_ratio', 
                         'passes_modes', 'current_delivery', 'avg_delivery']
            spike_df = spike_df[[col for col in spike_cols if col in spike_df.columns]]
            
            spike_df.to_excel(writer, sheet_name='All Spikes by MCap', index=False)
    
    print(f"\n✅ Analysis complete! Report saved to: {output_file}")
    
    # Print summary
    print("\n" + "="*80)
    print("SUMMARY BY MARKET CAP")
    print("="*80)
    
    print("\nFilter Pass Rates by Market Cap (Normal Mode):")
    normal_data = results_df[results_df['mode'] == 'normal']
    for mcap in ['LARGE CAP', 'MID CAP', 'SMALL CAP', 'MICRO CAP']:
        mcap_data = normal_data[normal_data['market_cap'] == mcap]
        if not mcap_data.empty:
            passed = mcap_data['passes_filter'].sum()
            total = len(mcap_data)
            print(f"  {mcap:12}: {passed:3}/{total:3} passed ({passed/total*100:5.1f}%)")
    
    print("\nTop 10 Delivery Spikes by Market Cap:")
    if not spike_df.empty:
        for idx, row in spike_df.head(10).iterrows():
            passes = row.get('passes_modes', 'NONE')
            print(f"  {row['symbol']:12} ({row['market_cap']:10}): {row['spike_ratio']:6.2f}x | Passes: {passes}")
    
    print("\nMarket Cap Performance:")
    for mcap in ['LARGE CAP', 'MID CAP', 'SMALL CAP']:
        mcap_spikes = spike_df[(spike_df['market_cap'] == mcap) & (spike_df['spike_ratio'] >= 3)]
        if not mcap_spikes.empty:
            print(f"\n  {mcap} with 3x+ spikes: {len(mcap_spikes)} stocks")
            for _, row in mcap_spikes.head(3).iterrows():
                print(f"    • {row['symbol']}: {row['spike_ratio']:.2f}x ({row['index_membership']})")
    
    return results_df


if __name__ == "__main__":
    import sys
    
    # Parse command line arguments
    if len(sys.argv) > 1:
        analysis_date = date.fromisoformat(sys.argv[1])
    else:
        analysis_date = date(2025, 11, 4)
    
    print(f"Analyzing {analysis_date} with Market Cap classification...")
    
    # Run analysis
    results = asyncio.run(analyze_all_stocks_with_filter(analysis_date))
    
    print("\n📊 Open the Excel file to see:")
    print("  • Summary by Market Cap: Breakdown by Large/Mid/Small")
    print("  • Normal Mode: All stocks with MCap info")
    print("  • Large/Mid/Small Cap sheets: Filtered by market cap")
    print("  • All Spikes by MCap: Shows market cap for every spike")
    print("\n✨ New columns added:")
    print("  • market_cap: LARGE CAP / MID CAP / SMALL CAP / MICRO CAP")
    print("  • index_membership: NIFTY 50, NIFTY MIDCAP 150, etc.")