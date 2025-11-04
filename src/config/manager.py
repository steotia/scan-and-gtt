"""
Configuration Management Implementation
Responsible for managing application configuration
"""

import yaml
import json
from pathlib import Path
from typing import Any, Dict, Optional
from dataclasses import dataclass, field, asdict
from loguru import logger
import os
from dotenv import load_dotenv

from src.interfaces import IConfiguration


@dataclass
class AppConfig:
    """Application configuration data class"""
    
    # Data fetching
    data_source: str = "NSE_BHAVCOPY"
    retry_count: int = 3
    rate_limit_delay: float = 1.0
    
    # Analysis
    lookback_days: int = 20
    spike_multiplier: float = 5.0
    min_volume: int = 100000
    min_delivery_percent: float = 30.0
    price_range: Dict[str, float] = field(default_factory=lambda: {"min": 10.0, "max": 50000.0})
    
    # Storage
    storage_type: str = "csv"  # csv or sqlite
    data_directory: str = "data"
    reports_directory: str = "reports"
    logs_directory: str = "logs"
    
    # Filtering
    index_filter: str = "ALL"  # ALL, NIFTY_50, NIFTY_100
    sectors: list = field(default_factory=list)
    market_cap: list = field(default_factory=lambda: ["LARGE", "MID"])
    
    # Reporting
    report_format: str = "excel"  # excel, html, pdf
    include_charts: bool = True
    max_spikes_in_report: int = 50
    
    # Email notifications
    email_enabled: bool = False
    smtp_server: str = ""
    smtp_port: int = 587
    email_from: str = ""
    email_to: list = field(default_factory=list)
    email_password: str = ""
    
    # Scheduling
    schedule_enabled: bool = False
    schedule_time: str = "18:00"  # Run at 6 PM
    schedule_days: list = field(default_factory=lambda: ["MON", "TUE", "WED", "THU", "FRI"])
    
    # Advanced
    use_cache: bool = True
    cache_ttl: int = 3600  # seconds
    parallel_processing: bool = True
    max_workers: int = 4
    debug_mode: bool = False
    
    # API Keys (loaded from environment)
    api_keys: Dict[str, str] = field(default_factory=dict)


class YAMLConfiguration(IConfiguration):
    """
    YAML-based configuration implementation
    Single Responsibility: Manage YAML configuration files
    """
    
    def __init__(self, config_file: str = "config.yaml"):
        """
        Initialize YAML configuration
        
        Args:
            config_file: Path to configuration file
        """
        self.config_file = Path(config_file)
        self.config = AppConfig()
        
        # Load environment variables
        load_dotenv()
        
        # Load configuration from file if exists
        if self.config_file.exists():
            self.load_from_file(str(self.config_file))
        else:
            logger.warning(f"Config file {config_file} not found, using defaults")
            self._create_default_config()
        
        # Override with environment variables
        self._load_env_overrides()
        
        logger.info("Configuration loaded successfully")
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value
        
        Args:
            key: Configuration key (supports dot notation)
            default: Default value if key not found
            
        Returns:
            Configuration value
        """
        # Support dot notation (e.g., "email.smtp_server")
        keys = key.split('.')
        value = asdict(self.config)
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        
        return value
    
    def set(self, key: str, value: Any) -> None:
        """
        Set configuration value
        
        Args:
            key: Configuration key (supports dot notation)
            value: Value to set
        """
        keys = key.split('.')
        
        if len(keys) == 1:
            if hasattr(self.config, key):
                setattr(self.config, key, value)
        else:
            # Handle nested keys
            config_dict = asdict(self.config)
            current = config_dict
            
            for k in keys[:-1]:
                if k not in current:
                    current[k] = {}
                current = current[k]
            
            current[keys[-1]] = value
            
            # Update config object
            self._update_from_dict(config_dict)
    
    def load_from_file(self, file_path: str) -> bool:
        """
        Load configuration from YAML file
        
        Args:
            file_path: Path to configuration file
            
        Returns:
            True if successful
        """
        try:
            with open(file_path, 'r') as f:
                data = yaml.safe_load(f) or {}
            
            self._update_from_dict(data)
            logger.info(f"Configuration loaded from {file_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error loading configuration: {e}")
            return False
    
    def save_to_file(self, file_path: str) -> bool:
        """
        Save configuration to YAML file
        
        Args:
            file_path: Path to save configuration
            
        Returns:
            True if successful
        """
        try:
            config_dict = asdict(self.config)
            
            # Remove sensitive information
            if 'email_password' in config_dict:
                config_dict['email_password'] = '***'
            if 'api_keys' in config_dict:
                config_dict['api_keys'] = {k: '***' for k in config_dict['api_keys']}
            
            with open(file_path, 'w') as f:
                yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False)
            
            logger.info(f"Configuration saved to {file_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving configuration: {e}")
            return False
    
    def _update_from_dict(self, data: Dict[str, Any]):
        """Update config object from dictionary"""
        for key, value in data.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)
    
    def _load_env_overrides(self):
        """Load configuration overrides from environment variables"""
        env_mappings = {
            'NSE_DATA_SOURCE': 'data_source',
            'NSE_LOOKBACK_DAYS': 'lookback_days',
            'NSE_SPIKE_MULTIPLIER': 'spike_multiplier',
            'NSE_EMAIL_ENABLED': 'email_enabled',
            'NSE_SMTP_SERVER': 'smtp_server',
            'NSE_SMTP_PORT': 'smtp_port',
            'NSE_EMAIL_FROM': 'email_from',
            'NSE_EMAIL_PASSWORD': 'email_password',
            'NSE_DEBUG_MODE': 'debug_mode'
        }
        
        for env_key, config_key in env_mappings.items():
            env_value = os.getenv(env_key)
            if env_value is not None:
                # Convert to appropriate type
                if config_key in ['lookback_days', 'smtp_port']:
                    env_value = int(env_value)
                elif config_key == 'spike_multiplier':
                    env_value = float(env_value)
                elif config_key in ['email_enabled', 'debug_mode']:
                    env_value = env_value.lower() in ['true', '1', 'yes']
                
                setattr(self.config, config_key, env_value)
                logger.debug(f"Override {config_key} from environment")
    
    def _create_default_config(self):
        """Create default configuration file"""
        default_config = """
# NSE Delivery Tracker Configuration

# Data Fetching
data_source: NSE_BHAVCOPY
retry_count: 3
rate_limit_delay: 1.0

# Analysis Parameters
lookback_days: 20
spike_multiplier: 5.0
min_volume: 100000
min_delivery_percent: 30.0
price_range:
  min: 10.0
  max: 50000.0

# Storage
storage_type: csv  # Options: csv, sqlite
data_directory: data
reports_directory: reports
logs_directory: logs

# Filtering
index_filter: ALL  # Options: ALL, NIFTY_50, NIFTY_100
sectors: []  # Leave empty for all sectors
market_cap:
  - LARGE
  - MID

# Reporting
report_format: excel
include_charts: true
max_spikes_in_report: 50

# Email Notifications
email_enabled: false
smtp_server: smtp.gmail.com
smtp_port: 587
email_from: ""
email_to: []
# email_password: Set via NSE_EMAIL_PASSWORD environment variable

# Scheduling
schedule_enabled: false
schedule_time: "18:00"
schedule_days:
  - MON
  - TUE
  - WED
  - THU
  - FRI

# Advanced
use_cache: true
cache_ttl: 3600
parallel_processing: true
max_workers: 4
debug_mode: false
"""
        
        try:
            self.config_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self.config_file, 'w') as f:
                f.write(default_config.strip())
            logger.info(f"Created default configuration at {self.config_file}")
        except Exception as e:
            logger.error(f"Could not create default config: {e}")


class ConfigurationValidator:
    """Validates configuration values"""
    
    @staticmethod
    def validate(config: AppConfig) -> tuple[bool, list]:
        """
        Validate configuration
        
        Args:
            config: Configuration object
            
        Returns:
            Tuple of (is_valid, list_of_errors)
        """
        errors = []
        
        # Validate numeric ranges
        if config.lookback_days < 1:
            errors.append("lookback_days must be at least 1")
        
        if config.spike_multiplier < 1.0:
            errors.append("spike_multiplier must be at least 1.0")
        
        if config.min_volume < 0:
            errors.append("min_volume cannot be negative")
        
        if not (0 <= config.min_delivery_percent <= 100):
            errors.append("min_delivery_percent must be between 0 and 100")
        
        # Validate price range
        if config.price_range["min"] >= config.price_range["max"]:
            errors.append("price_range min must be less than max")
        
        # Validate storage type
        if config.storage_type not in ["csv", "sqlite"]:
            errors.append("storage_type must be 'csv' or 'sqlite'")
        
        # Validate email configuration if enabled
        if config.email_enabled:
            if not config.smtp_server:
                errors.append("smtp_server is required when email is enabled")
            if not config.email_from:
                errors.append("email_from is required when email is enabled")
            if not config.email_to:
                errors.append("email_to list is required when email is enabled")
        
        # Validate index filter
        valid_indices = ["ALL", "NIFTY_50", "NIFTY_100", "NIFTY_NEXT_50"]
        if config.index_filter not in valid_indices:
            errors.append(f"index_filter must be one of {valid_indices}")
        
        return len(errors) == 0, errors


class ConfigurationManager:
    """
    High-level configuration manager
    Facade for configuration operations
    """
    
    def __init__(self, config_file: str = "config.yaml"):
        """Initialize configuration manager"""
        self.config = YAMLConfiguration(config_file)
        self.validator = ConfigurationValidator()
        
        # Validate configuration
        is_valid, errors = self.validator.validate(self.config.config)
        if not is_valid:
            logger.warning(f"Configuration validation errors: {errors}")
    
    def get_config(self) -> AppConfig:
        """Get configuration object"""
        return self.config.config
    
    def update_config(self, updates: Dict[str, Any]) -> bool:
        """
        Update configuration with validation
        
        Args:
            updates: Dictionary of updates
            
        Returns:
            True if successful
        """
        # Apply updates
        for key, value in updates.items():
            self.config.set(key, value)
        
        # Validate
        is_valid, errors = self.validator.validate(self.config.config)
        
        if not is_valid:
            logger.error(f"Invalid configuration: {errors}")
            return False
        
        return True
    
    def save_config(self, file_path: Optional[str] = None) -> bool:
        """Save configuration to file"""
        if file_path is None:
            file_path = str(self.config.config_file)
        
        return self.config.save_to_file(file_path)
