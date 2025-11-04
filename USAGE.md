# NSE Delivery Tracker - Usage Guide

## Quick Start

### 1. Installation

```bash
# Clone or download the project
cd nse_delivery_tracker

# Run setup script (Mac/Linux)
./setup.sh

# Or manually setup
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Basic Usage

```bash
# Run analysis for today
python main.py

# Run for specific date
python main.py --date 2024-01-15

# Custom parameters
python main.py --lookback 30 --multiplier 3.0
```

## Configuration

### Using config.yaml

Edit `config.yaml` to customize default settings:

```yaml
# Key settings to adjust
lookback_days: 20        # Days to calculate average
spike_multiplier: 5.0     # Minimum spike ratio (5x = 500% of average)
min_volume: 100000        # Minimum volume filter
min_delivery_percent: 30  # Minimum delivery % filter
index_filter: NIFTY_50    # Filter specific index
```

### Using Environment Variables

Create a `.env` file or export variables:

```bash
export NSE_LOOKBACK_DAYS=30
export NSE_SPIKE_MULTIPLIER=3.0
export NSE_DEBUG_MODE=true
```

## Command Line Options

```bash
python main.py [OPTIONS]

Options:
  -d, --date TEXT           Analysis date (YYYY-MM-DD)
  -l, --lookback INTEGER    Lookback days for average calculation
  -m, --multiplier FLOAT    Spike multiplier threshold
  -i, --index [ALL|NIFTY_50|NIFTY_100]  Index filter
  -c, --config TEXT         Configuration file path
  -o, --output TEXT         Output report path
  --no-fetch               Use only existing data
  --debug                  Enable debug mode
  --help                   Show help message
```

## Examples

### 1. Detect High Delivery Spikes (10x)

```bash
python main.py --multiplier 10.0
```

### 2. Analyze NIFTY 50 Stocks Only

```bash
python main.py --index NIFTY_50
```

### 3. Extended Historical Analysis

```bash
python main.py --lookback 60 --date 2024-01-15
```

### 4. Quick Analysis Without Fetching New Data

```bash
python main.py --no-fetch
```

### 5. Custom Output Location

```bash
python main.py --output /path/to/reports/analysis.xlsx
```

## Understanding the Output

### Excel Report Sheets

1. **Summary**: Overview of analysis with top 10 spikes
2. **Delivery Spikes**: Detailed list of all detected spikes
3. **Trends**: Trend analysis for spike stocks
4. **Statistics**: Market statistics and distributions
5. **Raw Data**: Complete dataset used for analysis
6. **Charts**: Visual representations

### Key Metrics

- **Spike Ratio**: Current delivery ÷ Average delivery
- **Delivery %**: (Delivered Quantity ÷ Traded Quantity) × 100
- **Trend**: Rising/Falling/Sideways based on recent patterns
- **Price Correlation**: Correlation between delivery and price

## Scheduling Daily Runs

### Using Cron (Linux/Mac)

```bash
# Edit crontab
crontab -e

# Add daily run at 6:30 PM (after market close)
30 18 * * 1-5 cd /path/to/nse_delivery_tracker && ./venv/bin/python main.py
```

### Using Task Scheduler (Windows)

1. Open Task Scheduler
2. Create Basic Task
3. Set trigger: Daily at 6:30 PM
4. Set action: Start program
5. Program: `C:\path\to\venv\Scripts\python.exe`
6. Arguments: `main.py`
7. Start in: `C:\path\to\nse_delivery_tracker`

## Filtering Strategies

### Quality Stocks Filter

```yaml
# In config.yaml
min_volume: 500000
min_delivery_percent: 40
index_filter: NIFTY_100
```

### Sector-Specific Analysis

```yaml
# In config.yaml
sectors:
  - BANKING
  - IT
  - PHARMA
```

### Market Cap Filter

```yaml
# In config.yaml
market_cap:
  - LARGE
  - MID
```

## Advanced Usage

### Custom Analysis Pipeline

```python
from main import create_app
import asyncio

# Create app with custom config
app = create_app('custom_config.yaml')

# Override settings programmatically
app.config.lookback_days = 45
app.config.spike_multiplier = 7.0

# Run analysis
results = asyncio.run(app.run_analysis())

# Process results
for spike in results['spikes'][:5]:
    print(f"{spike.symbol}: {spike.spike_ratio:.1f}x spike")
```

## Troubleshooting

### Common Issues

1. **No data fetched**: 
   - Check internet connection
   - Verify NSE website is accessible
   - Try with `--debug` flag

2. **Empty results**:
   - Reduce spike_multiplier (try 2.0 or 3.0)
   - Reduce min_delivery_percent
   - Check if analyzing weekend date

3. **Import errors**:
   - Ensure virtual environment is activated
   - Run `pip install -r requirements.txt`

## Data Interpretation Guide

### What to Look For

1. **Spike Ratio > 5x**: Unusual institutional interest
2. **Rising Trend + High Delivery**: Accumulation pattern
3. **Price Correlation > 0.7**: Strong price-delivery relationship
4. **Consistent High Delivery**: Long-term investor interest

### Red Flags

- Spike with declining price: Possible distribution
- Low volume with high delivery %: Thin trading
- Isolated spike without trend: One-time event

## Performance Optimization

1. **Use SQLite for large datasets**
2. **Enable caching for repeat queries**
3. **Limit historical fetch to required period**
4. **Use index filters to reduce data size**
