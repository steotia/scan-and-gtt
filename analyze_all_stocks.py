#!/usr/bin/env python
"""
COMPLETE UPDATED VERSION of analyze_all_stocks_with_mcap.py
Includes:
- Step 1: 2 decimal places for all numbers
- Step 2: Report named by analysis date
- Step 3: Delivery/volume/turnover change percentages
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
        analysis_date = date(2025, 11, 7)
    
    # STEP 2 CHANGE: Use analysis date in filename
    if output_file is None:
        date_str = analysis_date.strftime("%Y-%m-%d")
        output_file = f"analysis_{date_str}.xlsx"
    
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
                    
                    # STEP 1 CHANGE: Round all numbers to 2 decimal places
                    analysis_results.append({
                        'mode': filter_mode,
                        'symbol': metric.symbol,
                        'market_cap': stock.get('market_cap', 'UNKNOWN'),
                        'index_membership': stock.get('index_membership', 'UNKNOWN'),
                        'close': round(stock.get('close', metric.current_price), 2),
                        'volume': int(stock.get('volume', metric.current_volume)),
                        'delivery_qty': int(stock.get('delivery_qty', 0)),
                        'delivery_percent': round(stock.get('delivery_percent', 0), 2),
                        'current_turnover_cr': round(metric.current_turnover / 1_00_00_000, 2),
                        'avg_turnover_cr': round(metric.avg_turnover / 1_00_00_000, 2),
                        'rvol': round(metric.relative_volume, 2),
                        'turnover_ratio': round(metric.turnover_ratio, 2),
                        'price_bracket': metric.bracket,
                        'passes_filter': metric.passes_filter,
                        'filter_reason': metric.failure_reason or 'PASSED',
                        'vwap': round(stock.get('vwap', stock.get('close', 0)), 2)
                    })
    
    # Convert to DataFrame
    results_df = pd.DataFrame(analysis_results)
    
    # STEP 3 CHANGE: Calculate delivery/volume/turnover changes
    print("\n4. Calculating delivery spike ratios and changes...")
    
    # Get historical data for spike calculation
    historical_data = all_data[all_data['date'] < latest_date]
    
    spike_data = []
    for symbol in latest_data['symbol'].unique():
        try:
            # Get market cap for this symbol
            mcap_info = latest_data[latest_data['symbol'] == symbol].iloc[0]
            market_cap = mcap_info.get('market_cap', 'UNKNOWN')
            index_membership = mcap_info.get('index_membership', 'UNKNOWN')
            
            # Current values
            current = latest_data[latest_data['symbol'] == symbol].iloc[0]
            current_delivery = current.get('delivery_qty', 0)
            current_volume = current.get('volume', 0)
            current_close = current.get('close', 0)
            current_turnover = (current_close * current_volume) / 1_00_00_000  # In crores
            
            # Historical averages
            hist = historical_data[historical_data['symbol'] == symbol]
            if len(hist) >= 5:
                avg_delivery = hist['delivery_qty'].mean()
                avg_volume = hist['volume'].mean()
                avg_close = hist['close'].mean()
                avg_turnover = (avg_close * avg_volume) / 1_00_00_000  # In crores
                
                # Calculate ratios and changes
                if avg_delivery > 0:
                    spike_ratio = current_delivery / avg_delivery
                    delivery_change_pct = ((current_delivery - avg_delivery) / avg_delivery) * 100
                else:
                    spike_ratio = 0
                    delivery_change_pct = 0
                
                if avg_volume > 0:
                    volume_change_pct = ((current_volume - avg_volume) / avg_volume) * 100
                else:
                    volume_change_pct = 0
                
                if avg_turnover > 0:
                    turnover_change_pct = ((current_turnover - avg_turnover) / avg_turnover) * 100
                else:
                    turnover_change_pct = 0
            else:
                spike_ratio = 0
                avg_delivery = 0
                avg_volume = 0
                avg_turnover = 0
                delivery_change_pct = 0
                volume_change_pct = 0
                turnover_change_pct = 0
            
            spike_data.append({
                'symbol': symbol,
                'market_cap': market_cap,
                'index_membership': index_membership,
                'current_delivery': int(current_delivery),
                'avg_delivery': round(avg_delivery, 0),
                'spike_ratio': round(spike_ratio, 2),
                'delivery_change_pct': round(delivery_change_pct, 2),
                'current_volume': int(current_volume),
                'avg_volume': round(avg_volume, 0),
                'volume_change_pct': round(volume_change_pct, 2),
                'current_turnover_cr': round(current_turnover, 2),
                'avg_turnover_cr': round(avg_turnover, 2),
                'turnover_change_pct': round(turnover_change_pct, 2),
                'is_spike_5x': spike_ratio >= 5,
                'is_spike_3x': spike_ratio >= 3,
                'is_spike_2x': spike_ratio >= 2
            })
        except Exception as e:
            continue
    
    spike_df = pd.DataFrame(spike_data)
    
    # Merge spike data with filter results
    if not results_df.empty and not spike_df.empty:
        # Select columns to merge (avoid duplicates)
        merge_cols = ['symbol', 'spike_ratio', 'delivery_change_pct', 'volume_change_pct', 
                     'turnover_change_pct', 'current_volume', 'avg_volume', 
                     'current_turnover_cr', 'avg_turnover_cr', 
                     'is_spike_5x', 'is_spike_3x', 'is_spike_2x']
        spike_df_merge = spike_df[merge_cols]
        results_df = results_df.merge(spike_df_merge, on='symbol', how='left', suffixes=('', '_spike'))
    
    # Create Excel with multiple sheets
    print(f"\n5. Creating Excel report: {output_file}")
    
    with pd.ExcelWriter(output_file, engine='xlsxwriter') as writer:
        workbook = writer.book
        
        # Format definitions
        header_format = workbook.add_format({
            'bold': True,
            'bg_color': '#4A90E2',
            'font_color': 'white',
            'border': 1,
            'align': 'center'
        })
        
        # Number formats
        decimal2_format = workbook.add_format({'num_format': '0.00', 'border': 1})
        integer_format = workbook.add_format({'num_format': '#,##0', 'border': 1})
        percent_format = workbook.add_format({'num_format': '0.00%', 'border': 1})
        
        # Color formats for different market caps
        largecap_format = workbook.add_format({'bg_color': '#E8F5E9', 'border': 1})
        midcap_format = workbook.add_format({'bg_color': '#FFF3E0', 'border': 1})
        smallcap_format = workbook.add_format({'bg_color': '#FCE4EC', 'border': 1})
        
        # 1. Summary Sheet with Market Cap breakdown
        summary_data = []
        for mode in ['conservative', 'normal', 'aggressive']:
            mode_data = results_df[results_df['mode'] == mode]
            if not mode_data.empty:
                for mcap in ['LARGE CAP', 'MID CAP', 'SMALL CAP', 'MICRO CAP']:
                    mcap_data = mode_data[mode_data['market_cap'] == mcap]
                    if not mcap_data.empty:
                        mcap_passed = mcap_data['passes_filter'].sum()
                        mcap_total = len(mcap_data)
                        mcap_pass_rate = (mcap_passed / mcap_total * 100) if mcap_total > 0 else 0
                        
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
                            'Pass Rate %': round(mcap_pass_rate, 2),
                            '3x Spikes': mcap_spikes,
                            '3x Spikes Passed': mcap_spikes_passed
                        })
        
        summary_df = pd.DataFrame(summary_data)
        summary_df.to_excel(writer, sheet_name='Summary by Market Cap', index=False)
        
        # Apply formatting to Summary sheet
        worksheet = writer.sheets['Summary by Market Cap']
        for col_num, col_name in enumerate(summary_df.columns):
            worksheet.write(0, col_num, col_name, header_format)
            if 'Rate' in col_name:
                worksheet.set_column(col_num, col_num, 12, decimal2_format)
            elif 'Total' in col_name or 'Passed' in col_name or 'Spikes' in col_name:
                worksheet.set_column(col_num, col_num, 12, integer_format)
        
        # 2. Normal Mode Results (Most Important)
        normal_df = results_df[results_df['mode'] == 'normal'].copy()
        if not normal_df.empty:
            # Sort by spike ratio, then market cap
            normal_df['mcap_order'] = normal_df['market_cap'].map({
                'LARGE CAP': 1, 'MID CAP': 2, 'SMALL CAP': 3, 'MICRO CAP': 4
            })
            normal_df.sort_values(['spike_ratio', 'mcap_order'], ascending=[False, True], inplace=True)
            normal_df.drop('mcap_order', axis=1, inplace=True)
            
            # Write to Excel
            normal_df.to_excel(writer, sheet_name='Normal Mode', index=False)
            
            # Format Normal Mode sheet
            worksheet = writer.sheets['Normal Mode']
            for col_num, col_name in enumerate(normal_df.columns):
                worksheet.write(0, col_num, col_name, header_format)
        
        # 3. Top Spikes with Changes
        if not spike_df.empty:
            # Sort by spike ratio and get top 50
            top_spikes = spike_df.sort_values('spike_ratio', ascending=False).head(50)
            
            # Reorder columns for clarity
            spike_cols_order = [
                'symbol', 'market_cap', 'index_membership',
                'spike_ratio', 
                'delivery_change_pct', 'volume_change_pct', 'turnover_change_pct',
                'current_delivery', 'avg_delivery',
                'current_volume', 'avg_volume',
                'current_turnover_cr', 'avg_turnover_cr',
                'is_spike_3x', 'is_spike_2x'
            ]
            
            # Only include columns that exist
            spike_cols_order = [col for col in spike_cols_order if col in top_spikes.columns]
            top_spikes = top_spikes[spike_cols_order]
            
            top_spikes.to_excel(writer, sheet_name='Top Spikes with Changes', index=False)
            
            # Format Top Spikes sheet
            worksheet = writer.sheets['Top Spikes with Changes']
            for col_num, col_name in enumerate(top_spikes.columns):
                worksheet.write(0, col_num, col_name, header_format)
                if 'pct' in col_name or 'percent' in col_name:
                    worksheet.set_column(col_num, col_num, 15, decimal2_format)
                elif 'ratio' in col_name or '_cr' in col_name:
                    worksheet.set_column(col_num, col_num, 12, decimal2_format)
                elif 'volume' in col_name or 'delivery' in col_name:
                    worksheet.set_column(col_num, col_num, 14, integer_format)
    
    print(f"\n✅ Analysis complete! Report saved to: {output_file}")
    
    # Print summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    
    print("\nTop 10 Delivery Spikes with Changes:")
    if not spike_df.empty:
        print(f"{'Symbol':12} {'MCap':10} {'Spike':8} {'Del Δ%':10} {'Vol Δ%':10} {'TO Δ%':10}")
        print("-"*70)
        for idx, row in spike_df.head(10).iterrows():
            print(f"{row['symbol']:12} {row['market_cap']:10} "
                  f"{row['spike_ratio']:8.2f}x "
                  f"{row['delivery_change_pct']:+9.2f}% "
                  f"{row['volume_change_pct']:+9.2f}% "
                  f"{row['turnover_change_pct']:+9.2f}%")
    
    return results_df


if __name__ == "__main__":
    import sys
    
    # Parse command line arguments
    if len(sys.argv) > 1:
        analysis_date = date.fromisoformat(sys.argv[1])
    else:
        analysis_date = date(2025, 11, 7)
    
    print(f"Analyzing {analysis_date} with Market Cap classification...")
    
    # Run analysis
    results = asyncio.run(analyze_all_stocks_with_filter(analysis_date))
    
    print("\n📊 Report Features:")
    print("  ✅ Numbers rounded to 2 decimal places")
    print("  ✅ Report named with analysis date")
    print("  ✅ Delivery/Volume/Turnover change % added")
    print("  ✅ Market cap classification included")
