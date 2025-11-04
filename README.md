# NSE Delivery Tracker

A modular Python application for tracking NSE (National Stock Exchange) delivery data, detecting unusual spikes, and analyzing delivery trends.

## Features

- **Automatic Data Download**: Fetches daily bhavcopy data from NSE
- **Spike Detection**: Identifies stocks with delivery quantities ≥5X their historical average
- **Trend Analysis**: Tracks rising delivery patterns
- **Configurable Analysis**: Customizable lookback periods and spike multipliers
- **Excel Reports**: Generates detailed analysis reports with charts
- **Email Alerts**: Sends notifications for significant delivery spikes
- **Modular Architecture**: Following SOLID principles for maintainability

## Architecture

The system follows SOLID principles:
- **S**ingle Responsibility: Each class has one clear purpose
- **O**pen/Closed: Easy to extend without modifying existing code
- **L**iskov Substitution: Interfaces allow swapping implementations
- **I**nterface Segregation: Small, focused interfaces
- **D**ependency Inversion: Depends on abstractions, not concretions

## Project Structure

```
nse_delivery_tracker/
├── src/
│   ├── __init__.py
│   ├── interfaces/          # Abstract base classes
│   ├── data/               # Data fetching and storage
│   ├── analysis/           # Analysis engines
│   ├── reporting/          # Report generation
│   ├── notification/       # Alert systems
│   └── config/            # Configuration management
├── tests/                  # Unit tests
├── data/                   # Data storage
├── reports/               # Generated reports
├── logs/                  # Application logs
├── config.yaml            # Configuration file
├── main.py               # Application entry point
└── requirements.txt      # Dependencies
```

## Installation

1. Clone the repository
2. Create a virtual environment:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Configure settings in `config.yaml`
5. Run the application:
   ```bash
   python main.py
   ```

## Usage

### Basic Usage
```python
python main.py --date today
```

### Custom Analysis
```python
python main.py --lookback 30 --multiplier 5 --index NIFTY50
```

### Schedule Daily Runs
```bash
# Add to crontab
0 18 * * 1-5 cd /path/to/nse_delivery_tracker && ./venv/bin/python main.py
```

## Configuration

Edit `config.yaml` to customize:
- Lookback periods
- Spike multipliers
- Email settings
- Report formats
- Index filters

## Testing

Run tests with:
```bash
pytest tests/
```

## License

MIT License
