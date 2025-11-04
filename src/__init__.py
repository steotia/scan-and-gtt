"""
NSE Delivery Tracker Package
A modular system for tracking and analyzing NSE delivery data
"""

__version__ = "1.0.0"
__author__ = "NSE Tracker Team"

# Make key components available at package level
from src.config.manager import ConfigurationManager, AppConfig
from src.interfaces import DeliverySpike, DataSource

__all__ = [
    "ConfigurationManager",
    "AppConfig", 
    "DeliverySpike",
    "DataSource"
]
