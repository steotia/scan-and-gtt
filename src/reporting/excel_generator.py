"""
Excel Report Generator Implementation
Responsible for generating Excel reports with charts
"""

import pandas as pd
from pathlib import Path
from datetime import datetime, date
from typing import Dict, Any, List, Optional
import xlsxwriter
from loguru import logger
import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64

from src.interfaces import IReportGenerator, DeliverySpike


class ExcelReportGenerator(IReportGenerator):
    """
    Excel report generator with charts and formatting
    Single Responsibility: Generate Excel reports
    """
    
    def __init__(self, template_path: Optional[str] = None):
        """
        Initialize Excel report generator
        
        Args:
            template_path: Optional path to Excel template
        """
        self.template_path = template_path
        self.workbook = None
        self.charts = []
        logger.info("Excel report generator initialized")
    
    def generate(self, 
                analysis_results: Dict[str, Any], 
                output_path: str) -> bool:
        """
        Generate Excel report from analysis results
        
        Args:
            analysis_results: Dictionary with analysis results
            output_path: Path to save the report
            
        Returns:
            True if successful
        """
        try:
            # Ensure output directory exists
            output_file = Path(output_path)
            output_file.parent.mkdir(parents=True, exist_ok=True)
            
            # Create workbook
            self.workbook = xlsxwriter.Workbook(str(output_file))
            
            # Add formats
            formats = self._create_formats()
            
            # Add worksheets
            self._add_summary_sheet(analysis_results, formats)
            self._add_spikes_sheet(analysis_results.get('spikes', []), formats)
            self._add_trends_sheet(analysis_results.get('trends', {}), formats)
            self._add_statistics_sheet(analysis_results.get('statistics', {}), formats)
            self._add_raw_data_sheet(analysis_results.get('raw_data', pd.DataFrame()), formats)
            
            # Add charts
            self._add_charts_sheet(analysis_results, formats)
            
            # Close workbook
            self.workbook.close()
            
            logger.info(f"Report generated successfully: {output_file}")
            return True
            
        except Exception as e:
            logger.error(f"Error generating Excel report: {e}")
            return False
    
    def add_chart(self, chart_data: pd.DataFrame, chart_type: str) -> None:
        """
        Add chart to the report
        
        Args:
            chart_data: Data for the chart
            chart_type: Type of chart
        """
        self.charts.append({
            'data': chart_data,
            'type': chart_type
        })
    
    def _create_formats(self) -> Dict[str, Any]:
        """Create cell formats for the workbook"""
        formats = {
            'header': self.workbook.add_format({
                'bold': True,
                'bg_color': '#4A90E2',
                'font_color': 'white',
                'align': 'center',
                'valign': 'vcenter',
                'border': 1,
                'font_size': 12
            }),
            'title': self.workbook.add_format({
                'bold': True,
                'font_size': 16,
                'align': 'center',
                'valign': 'vcenter'
            }),
            'subtitle': self.workbook.add_format({
                'bold': True,
                'font_size': 14,
                'align': 'left'
            }),
            'date': self.workbook.add_format({
                'num_format': 'yyyy-mm-dd',
                'align': 'center'
            }),
            'number': self.workbook.add_format({
                'num_format': '#,##0',
                'align': 'right'
            }),
            'decimal': self.workbook.add_format({
                'num_format': '#,##0.00',
                'align': 'right'
            }),
            'percent': self.workbook.add_format({
                'num_format': '0.00%',
                'align': 'right'
            }),
            'currency': self.workbook.add_format({
                'num_format': '₹#,##0.00',
                'align': 'right'
            }),
            'spike': self.workbook.add_format({
                'bold': True,
                'bg_color': '#FFE4E1',
                'font_color': '#8B0000',
                'align': 'center',
                'border': 1
            }),
            'positive': self.workbook.add_format({
                'font_color': '#008000',
                'align': 'right'
            }),
            'negative': self.workbook.add_format({
                'font_color': '#FF0000',
                'align': 'right'
            })
        }
        return formats
    
    def _add_summary_sheet(self, analysis_results: Dict[str, Any], formats: Dict):
        """Add summary worksheet"""
        worksheet = self.workbook.add_worksheet('Summary')
        
        # Title
        worksheet.merge_range('A1:H1', 'NSE Delivery Analysis Report', formats['title'])
        worksheet.merge_range('A2:H2', f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M')}", formats['subtitle'])
        
        row = 4
        
        # Key metrics
        worksheet.write(row, 0, 'Key Metrics', formats['subtitle'])
        row += 1
        
        metrics = [
            ('Analysis Date', analysis_results.get('analysis_date', date.today().isoformat())),
            ('Total Stocks Analyzed', analysis_results.get('total_stocks', 0)),
            ('Stocks with Delivery Spikes', len(analysis_results.get('spikes', []))),
            ('Spike Threshold', f"{analysis_results.get('spike_multiplier', 5)}x"),
            ('Lookback Period', f"{analysis_results.get('lookback_days', 20)} days"),
            ('Filters Applied', analysis_results.get('filters_description', 'None'))
        ]
        
        for metric_name, metric_value in metrics:
            worksheet.write(row, 0, metric_name, formats['header'])
            worksheet.write(row, 1, str(metric_value))
            row += 1
        
        # Top spikes
        row += 2
        worksheet.write(row, 0, 'Top 10 Delivery Spikes', formats['subtitle'])
        row += 1
        
        # Headers
        headers = ['Symbol', 'Spike Ratio', 'Current Delivery', 'Avg Delivery', 
                  'Price Change %', 'Volume Change %']
        for col, header in enumerate(headers):
            worksheet.write(row, col, header, formats['header'])
        
        row += 1
        
        # Data
        spikes = analysis_results.get('spikes', [])[:10]
        for spike in spikes:
            worksheet.write(row, 0, spike.symbol)
            worksheet.write(row, 1, f"{spike.spike_ratio:.1f}x", formats['spike'])
            worksheet.write(row, 2, spike.current_delivery, formats['number'])
            worksheet.write(row, 3, spike.avg_delivery, formats['decimal'])
            
            # Price change with color
            price_format = formats['positive'] if spike.price_change >= 0 else formats['negative']
            worksheet.write(row, 4, spike.price_change / 100, price_format)
            
            # Volume change with color
            vol_format = formats['positive'] if spike.volume_change >= 0 else formats['negative']
            worksheet.write(row, 5, spike.volume_change / 100, vol_format)
            
            row += 1
        
        # Adjust column widths
        worksheet.set_column('A:A', 15)
        worksheet.set_column('B:F', 18)
    
    def _add_spikes_sheet(self, spikes: List[DeliverySpike], formats: Dict):
        """Add detailed spikes worksheet"""
        worksheet = self.workbook.add_worksheet('Delivery Spikes')
        
        # Title
        worksheet.merge_range('A1:H1', 'Detailed Delivery Spikes Analysis', formats['title'])
        
        row = 3
        
        # Headers
        headers = ['Rank', 'Symbol', 'Date', 'Spike Ratio', 'Current Delivery', 
                  'Average Delivery', 'Price Change %', 'Volume Change %']
        
        for col, header in enumerate(headers):
            worksheet.write(row, col, header, formats['header'])
        
        row += 1
        
        # Data
        for rank, spike in enumerate(spikes, 1):
            worksheet.write(row, 0, rank, formats['number'])
            worksheet.write(row, 1, spike.symbol)
            worksheet.write(row, 2, spike.spike_date, formats['date'])
            
            # Spike ratio with conditional formatting
            spike_cell_format = formats['spike'] if spike.spike_ratio >= 10 else formats['decimal']
            worksheet.write(row, 3, spike.spike_ratio, spike_cell_format)
            
            worksheet.write(row, 4, spike.current_delivery, formats['number'])
            worksheet.write(row, 5, spike.avg_delivery, formats['decimal'])
            
            # Price change
            price_format = formats['positive'] if spike.price_change >= 0 else formats['negative']
            worksheet.write(row, 6, spike.price_change / 100, price_format)
            
            # Volume change
            vol_format = formats['positive'] if spike.volume_change >= 0 else formats['negative']
            worksheet.write(row, 7, spike.volume_change / 100, vol_format)
            
            row += 1
        
        # Add conditional formatting
        if spikes:
            worksheet.conditional_format(f'D5:D{row}', {
                'type': 'data_bar',
                'bar_color': '#FF6347'
            })
        
        # Adjust column widths
        worksheet.set_column('A:A', 8)
        worksheet.set_column('B:B', 15)
        worksheet.set_column('C:C', 12)
        worksheet.set_column('D:H', 18)
    
    def _add_trends_sheet(self, trends: Dict[str, Any], formats: Dict):
        """Add trends analysis worksheet"""
        worksheet = self.workbook.add_worksheet('Trends')
        
        # Title
        worksheet.merge_range('A1:G1', 'Delivery Trends Analysis', formats['title'])
        
        row = 3
        
        if not trends:
            worksheet.write(row, 0, "No trend data available")
            return
        
        # Headers
        headers = ['Symbol', 'Trend', 'Trend Strength', 'MA Short', 'MA Long', 
                  'Price Correlation', 'Volatility']
        
        for col, header in enumerate(headers):
            worksheet.write(row, col, header, formats['header'])
        
        row += 1
        
        # Data
        for symbol, trend_data in trends.items():
            if isinstance(trend_data, dict):
                worksheet.write(row, 0, symbol)
                worksheet.write(row, 1, trend_data.get('trend', 'Unknown'))
                worksheet.write(row, 2, trend_data.get('trend_strength', 0) / 100, formats['percent'])
                worksheet.write(row, 3, trend_data.get('moving_average_short', 0), formats['number'])
                worksheet.write(row, 4, trend_data.get('moving_average_long', 0), formats['number'])
                worksheet.write(row, 5, trend_data.get('price_correlation', 0), formats['decimal'])
                worksheet.write(row, 6, trend_data.get('volatility', 0), formats['decimal'])
                row += 1
        
        # Adjust column widths
        worksheet.set_column('A:A', 15)
        worksheet.set_column('B:G', 18)
    
    def _add_statistics_sheet(self, statistics: Dict[str, Any], formats: Dict):
        """Add statistics worksheet"""
        worksheet = self.workbook.add_worksheet('Statistics')
        
        # Title
        worksheet.merge_range('A1:E1', 'Statistical Analysis', formats['title'])
        
        row = 3
        
        # Market Statistics
        worksheet.write(row, 0, 'Market Statistics', formats['subtitle'])
        row += 1
        
        market_stats = statistics.get('market', {})
        stat_items = [
            ('Total Trading Volume', market_stats.get('total_volume', 0)),
            ('Total Delivery Quantity', market_stats.get('total_delivery', 0)),
            ('Average Delivery %', market_stats.get('avg_delivery_percent', 0)),
            ('Stocks Above Average', market_stats.get('stocks_above_avg', 0)),
            ('Highest Delivery Stock', market_stats.get('highest_delivery_stock', 'N/A')),
            ('Lowest Delivery Stock', market_stats.get('lowest_delivery_stock', 'N/A'))
        ]
        
        for stat_name, stat_value in stat_items:
            worksheet.write(row, 0, stat_name, formats['header'])
            if isinstance(stat_value, (int, float)):
                if 'percent' in stat_name.lower():
                    worksheet.write(row, 1, stat_value / 100, formats['percent'])
                else:
                    worksheet.write(row, 1, stat_value, formats['number'])
            else:
                worksheet.write(row, 1, str(stat_value))
            row += 1
        
        # Distribution Statistics
        row += 2
        worksheet.write(row, 0, 'Delivery Distribution', formats['subtitle'])
        row += 1
        
        distribution = statistics.get('distribution', {})
        dist_items = [
            ('Mean', distribution.get('mean', 0)),
            ('Median', distribution.get('median', 0)),
            ('Standard Deviation', distribution.get('std', 0)),
            ('25th Percentile', distribution.get('q25', 0)),
            ('75th Percentile', distribution.get('q75', 0)),
            ('Skewness', distribution.get('skewness', 0)),
            ('Kurtosis', distribution.get('kurtosis', 0))
        ]
        
        for stat_name, stat_value in dist_items:
            worksheet.write(row, 0, stat_name, formats['header'])
            worksheet.write(row, 1, stat_value, formats['decimal'])
            row += 1
        
        # Adjust column widths
        worksheet.set_column('A:A', 25)
        worksheet.set_column('B:B', 20)
    
    def _add_raw_data_sheet(self, data: pd.DataFrame, formats: Dict):
        """Add raw data worksheet"""
        if data.empty:
            return
        
        worksheet = self.workbook.add_worksheet('Raw Data')
        
        # Title
        worksheet.merge_range('A1:L1', 'Raw Stock Data', formats['title'])
        
        row = 3
        
        # Headers
        for col, header in enumerate(data.columns):
            worksheet.write(row, col, header, formats['header'])
        
        row += 1
        
        # Data (limit to 1000 rows for performance)
        for idx, data_row in data.head(1000).iterrows():
            for col, value in enumerate(data_row):
                if pd.isna(value):
                    worksheet.write(row, col, '')
                elif isinstance(value, (int, float)):
                    if col < len(data.columns) and 'percent' in data.columns[col].lower():
                        worksheet.write(row, col, value / 100, formats['percent'])
                    else:
                        worksheet.write(row, col, value, formats['number'])
                else:
                    worksheet.write(row, col, str(value))
            row += 1
        
        # Add autofilter
        worksheet.autofilter(3, 0, row - 1, len(data.columns) - 1)
        
        # Adjust column widths
        for col in range(len(data.columns)):
            worksheet.set_column(col, col, 15)
    
    def _add_charts_sheet(self, analysis_results: Dict[str, Any], formats: Dict):
        """Add charts worksheet"""
        worksheet = self.workbook.add_worksheet('Charts')
        
        # Title
        worksheet.merge_range('A1:H1', 'Visual Analysis', formats['title'])
        
        # Get spikes data
        spikes = analysis_results.get('spikes', [])
        if not spikes:
            worksheet.write(3, 0, "No data available for charts")
            return
        
        # Chart 1: Top 10 Spikes Bar Chart
        chart1 = self.workbook.add_chart({'type': 'column'})
        
        # Prepare data for chart
        worksheet.write(3, 10, 'Symbol', formats['header'])
        worksheet.write(3, 11, 'Spike Ratio', formats['header'])
        
        for i, spike in enumerate(spikes[:10]):
            worksheet.write(4 + i, 10, spike.symbol)
            worksheet.write(4 + i, 11, spike.spike_ratio)
        
        # Configure chart
        chart1.add_series({
            'categories': ['Charts', 4, 10, 13, 10],
            'values': ['Charts', 4, 11, 13, 11],
            'name': 'Spike Ratio',
            'data_labels': {'value': True, 'num_format': '0.0x'}
        })
        
        chart1.set_title({'name': 'Top 10 Delivery Spikes'})
        chart1.set_x_axis({'name': 'Symbol'})
        chart1.set_y_axis({'name': 'Spike Ratio (X times average)'})
        chart1.set_size({'width': 720, 'height': 480})
        
        worksheet.insert_chart('A4', chart1)
        
        # Chart 2: Price vs Volume Change Scatter
        if len(spikes) > 1:
            chart2 = self.workbook.add_chart({'type': 'scatter'})
            
            # Prepare data
            worksheet.write(3, 13, 'Price Change %', formats['header'])
            worksheet.write(3, 14, 'Volume Change %', formats['header'])
            
            for i, spike in enumerate(spikes[:20]):
                worksheet.write(4 + i, 13, spike.price_change)
                worksheet.write(4 + i, 14, spike.volume_change)
            
            chart2.add_series({
                'categories': ['Charts', 4, 13, min(23, 3 + len(spikes)), 13],
                'values': ['Charts', 4, 14, min(23, 3 + len(spikes)), 14],
                'name': 'Price vs Volume Change',
                'marker': {'type': 'circle', 'size': 8}
            })
            
            chart2.set_title({'name': 'Price Change vs Volume Change'})
            chart2.set_x_axis({'name': 'Price Change %'})
            chart2.set_y_axis({'name': 'Volume Change %'})
            chart2.set_size({'width': 720, 'height': 480})
            
            worksheet.insert_chart('A30', chart2)
    
    def _add_charts_sheet(self, analysis_results: Dict[str, Any], formats: Dict):
        """Add charts worksheet"""
        worksheet = self.workbook.add_worksheet('Charts')
        
        # Title
        worksheet.merge_range('A1:H1', 'Visual Analysis', formats['title'])
        
        # Always create the ChartData worksheet (fix for chart_data not defined error)
        chart_data = self.workbook.add_worksheet('ChartData')
        chart_data.hide()
        
        # Chart 1: Top 10 Spikes Bar Chart
        spikes = analysis_results.get('spikes', [])[:10]
        if spikes:
            # Write data
            chart_data.write(0, 0, 'Symbol')
            chart_data.write(0, 1, 'Spike Ratio')
            
            for i, spike in enumerate(spikes):
                chart_data.write(i + 1, 0, spike.symbol)
                chart_data.write(i + 1, 1, spike.spike_ratio)
            
            # Create bar chart
            chart = self.workbook.add_chart({'type': 'bar'})
            chart.add_series({
                'name': 'Spike Ratio',
                'categories': ['ChartData', 1, 0, len(spikes), 0],
                'values': ['ChartData', 1, 1, len(spikes), 1],
                'fill': {'color': '#FF6347'},
                'border': {'color': '#8B0000'}
            })
            
            chart.set_title({'name': 'Top 10 Delivery Spikes'})
            chart.set_x_axis({'name': 'Spike Ratio (times)'})
            chart.set_y_axis({'name': 'Stock Symbol'})
            chart.set_size({'width': 720, 'height': 400})
            
            worksheet.insert_chart('B3', chart)
        else:
            # No spikes found message
            worksheet.write(3, 1, 'No delivery spikes found for chart generation')
        
        # Chart 2: Delivery Distribution Histogram
        if 'raw_data' in analysis_results and not analysis_results['raw_data'].empty:
            raw_data = analysis_results['raw_data']
            
            if 'delivery_percent' in raw_data.columns:
                try:
                    # Create histogram chart
                    hist_chart = self.workbook.add_chart({'type': 'column'})
                    
                    # Bin the data
                    bins = [0, 20, 40, 60, 80, 100]
                    labels = ['0-20%', '20-40%', '40-60%', '60-80%', '80-100%']
                    
                    # Count occurrences in each bin
                    counts = pd.cut(raw_data['delivery_percent'], bins=bins).value_counts().sort_index()
                    
                    # Write histogram data
                    for i, (label, count) in enumerate(zip(labels, counts)):
                        chart_data.write(i + 15, 0, label)
                        chart_data.write(i + 15, 1, count)
                    
                    hist_chart.add_series({
                        'name': 'Stock Count',
                        'categories': ['ChartData', 15, 0, 19, 0],
                        'values': ['ChartData', 15, 1, 19, 1],
                        'fill': {'color': '#4A90E2'}
                    })
                    
                    hist_chart.set_title({'name': 'Delivery Percentage Distribution'})
                    hist_chart.set_x_axis({'name': 'Delivery Percentage Range'})
                    hist_chart.set_y_axis({'name': 'Number of Stocks'})
                    hist_chart.set_size({'width': 720, 'height': 400})
                    
                    worksheet.insert_chart('B25', hist_chart)
                except Exception as e:
                    logger.warning(f"Could not create histogram: {e}")
                    worksheet.write(25, 1, 'Histogram data not available')