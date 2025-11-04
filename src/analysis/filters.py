"""
Enhanced Filter Strategy Implementations with Complete NSE Indices
Includes Large Cap, Mid Cap, Small Cap categorization
"""

import pandas as pd
from typing import List, Set, Dict
from abc import ABC, abstractmethod
from loguru import logger

from src.interfaces import IFilterStrategy


class VolumeFilter(IFilterStrategy):
    """Filter stocks based on minimum volume"""
    
    def __init__(self, min_volume: int = 100000):
        self.min_volume = min_volume
    
    def filter(self, data: pd.DataFrame) -> pd.DataFrame:
        if 'volume' not in data.columns:
            return data
        
        filtered = data[data['volume'] >= self.min_volume]
        logger.debug(f"Volume filter: {len(data)} -> {len(filtered)} stocks")
        return filtered
    
    def get_description(self) -> str:
        return f"Min volume: {self.min_volume:,}"


class DeliveryPercentFilter(IFilterStrategy):
    """Filter stocks based on minimum delivery percentage"""
    
    def __init__(self, min_delivery_percent: float = 30.0):
        self.min_delivery_percent = min_delivery_percent
    
    def filter(self, data: pd.DataFrame) -> pd.DataFrame:
        if 'delivery_percent' not in data.columns:
            return data
        
        filtered = data[data['delivery_percent'] >= self.min_delivery_percent]
        logger.debug(f"Delivery % filter: {len(data)} -> {len(filtered)} stocks")
        return filtered
    
    def get_description(self) -> str:
        return f"Min delivery %: {self.min_delivery_percent}%"


class PriceRangeFilter(IFilterStrategy):
    """Filter stocks based on price range"""
    
    def __init__(self, min_price: float = 10.0, max_price: float = 10000.0):
        self.min_price = min_price
        self.max_price = max_price
    
    def filter(self, data: pd.DataFrame) -> pd.DataFrame:
        if 'close' not in data.columns:
            return data
        
        filtered = data[
            (data['close'] >= self.min_price) & 
            (data['close'] <= self.max_price)
        ]
        logger.debug(f"Price filter: {len(data)} -> {len(filtered)} stocks")
        return filtered
    
    def get_description(self) -> str:
        return f"Price: ₹{self.min_price}-₹{self.max_price}"


class IndexFilter(IFilterStrategy):
    """
    Enhanced filter for NSE indices with proper market cap categorization
    """
    
    # LARGE CAP INDICES
    NIFTY_50 = [
        'RELIANCE', 'TCS', 'HDFCBANK', 'INFY', 'ICICIBANK', 'HINDUNILVR', 
        'ITC', 'SBIN', 'BAJFINANCE', 'BHARTIARTL', 'KOTAKBANK', 'LT',
        'AXISBANK', 'ASIANPAINT', 'HCLTECH', 'MARUTI', 'SUNPHARMA', 'TITAN',
        'ULTRACEMCO', 'WIPRO', 'NESTLEIND', 'TECHM', 'ONGC', 'POWERGRID',
        'NTPC', 'JSWSTEEL', 'M&M', 'TATAMOTORS', 'HDFCLIFE', 'TATASTEEL',
        'ADANIENT', 'INDUSINDBK', 'BAJAJFINSV', 'HINDALCO', 'DIVISLAB',
        'CIPLA', 'DRREDDY', 'BRITANNIA', 'SBILIFE', 'GRASIM', 'HEROMOTOCO',
        'COALINDIA', 'APOLLOHOSP', 'EICHERMOT', 'BAJAJ-AUTO', 'TATACONSUM',
        'BEL', 'ADANIPORTS', 'LTIM', 'SHRIRAMFIN'
    ]
    
    NIFTY_NEXT_50 = [
        'VEDL', 'PNB', 'INDIGO', 'BANKBARODA', 'PIDILITIND', 'ZOMATO',
        'JINDALSTEL', 'CHOLAFIN', 'HAVELLS', 'AMBUJACEM', 'SIEMENS',
        'ICICIPRULI', 'DLF', 'ICICIGI', 'GODREJCP', 'DABUR', 'BERGEPAINT',
        'INDUSTOWER', 'ADANIGREEN', 'MARICO', 'CANBK', 'BOSCHLTD', 'SRF',
        'MOTHERSON', 'TORNTPHARM', 'PAGEIND', 'TRENT', 'MCDOWELL-N',
        'PGHH', 'TVSMOTOR', 'IRCTC', 'ZYDUSLIFE', 'NAUKRI', 'GODREJPROP',
        'ADANIPOWER', 'INDHOTEL', 'POLYCAB', 'ABB', 'PERSISTENT', 'OBEROIRLTY',
        'COLPAL', 'IOC', 'BAJAJHLDNG', 'TIINDIA', 'GICRE', 'LINDEINDIA',
        'MAXHEALTH', 'DMART', 'ALKEM', 'CGPOWER'
    ]
    
    # MID CAP INDICES
    NIFTY_MIDCAP_150 = [
        # First 50 mid-caps
        'AUROPHARMA', 'BANDHANBNK', 'BANKBARODA', 'BHARATFORG', 'BIOCON',
        'CANBK', 'COFORGE', 'CONCOR', 'CUMMINSIND', 'ESCORTS', 'FEDERALBNK',
        'GAIL', 'GMRINFRA', 'GODREJIND', 'IDFCFIRSTB', 'INDHOTEL', 'JUBLFOOD',
        'L&TFH', 'LICHSGFIN', 'LUPIN', 'M&MFIN', 'MANAPPURAM', 'MUTHOOTFIN',
        'NATIONALUM', 'NAUKRI', 'NMDC', 'OBEROIRLTY', 'OFSS', 'PETRONET',
        'PFC', 'PIIND', 'PNB', 'RECLTD', 'SAIL', 'SBICARD', 'TATACOMM',
        'TATAPOWER', 'TORNTPOWER', 'TVSMOTOR', 'UBL', 'UNIONBANK', 'VOLTAS',
        
        # Additional mid-caps (next 100)
        'ACC', 'ABCAPITAL', 'ABFRL', 'ALKEM', 'APOLLOTYRE', 'ASHOKLEY',
        'ASTRAL', 'ATUL', 'AUBANK', 'AUROPHARMA', 'BALKRISIND', 'BALRAMCHIN',
        'BATAINDIA', 'BEL', 'BHEL', 'BIRLACORPN', 'CAMS', 'CANFINHOME',
        'CDSL', 'CENTRALBK', 'CENTURYPLY', 'CHAMBLFERT', 'COALINDIA', 'COROMANDEL',
        'CROMPTON', 'CUB', 'DCBBANK', 'DEEPAKNTR', 'DELTACORP', 'DIXON',
        'EMAMILTD', 'ENDURANCE', 'EQUITAS', 'EXIDEIND', 'FORTIS', 'GLENMARK',
        'GNFC', 'GODREJAGRO', 'GSFC', 'GUJGASLTD', 'HAPPSTMNDS', 'HATSUN',
        'HFCL', 'HINDZINC', 'HUDCO', 'IBULHSGFIN', 'IDBI', 'IDFCFIRSTB',
        'INDIANB', 'INDIAMART', 'INTELLECT', 'IOB', 'IPCALAB', 'ISEC',
        'JBCHEPHARM', 'JINDALSAW', 'JKCEMENT', 'JKLAKSHMI', 'JMFINANCIL',
        'JSWENERGY', 'JUSTDIAL', 'KAJARIACER', 'KANSAINER', 'KARURVYSYA',
        'KEI', 'KNRCON', 'KRBL', 'KSCL', 'LAXMIMACH', 'LEMONTREE',
        'LINDEINDIA', 'LUXIND', 'M&M', 'MAHABANK', 'MAHINDCIE', 'MAHLOG',
        'MAHSCOOTER', 'MAHSEAMLES', 'MAITHANALL', 'MASFIN', 'MCX',
        'METROPOLIS', 'MFSL', 'MINDTREE', 'MOTILALOFS', 'MPHASIS', 'MRPL',
        'NATCOPHARM', 'NBCC', 'NCC', 'NESCO', 'NETWORK18', 'NHL',
        'NLCINDIA', 'NOCIL', 'NTPC', 'OIL', 'ONGC', 'ORIENTELEC',
        'PAGEIND', 'PARAGMILK', 'PCJEWELLER', 'PEL', 'PERSISTENT', 'PGHL',
        'PHOENIXLTD', 'PNBHOUSING', 'POLYPLEX', 'POWERINDIA', 'PRESTIGE'
    ]
    
    # SMALL CAP INDICES  
    NIFTY_SMALLCAP_250 = [
        # First 100 small-caps
        '3MINDIA', '5PAISA', 'AARTIDRUGS', 'AARTIIND', 'AAVAS', 'ABMINTLTD',
        'ACCELYA', 'ADFFOODS', 'AEGISCHEM', 'AFFLE', 'AGARIND', 'AGROPHOS',
        'AJANTPHARM', 'AKZOINDIA', 'ALLCARGO', 'AMARAJABAT', 'AMBER',
        'AMRUTANJAN', 'ANANTRAJ', 'ANDHRSUGAR', 'ANGELONE', 'ANURAS',
        'APARINDS', 'APCOTEXIND', 'APLAPOLLO', 'APLLTD', 'APTUS', 'ARVINDFASN',
        'ASAHIINDIA', 'ASAHISONG', 'ASTRAZEN', 'ATGL', 'AVADHSUGAR', 'AVANTIFEED',
        'BAJAJELEC', 'BAJAJHIND', 'BALAMINES', 'BALMLAWRIE', 'BANCOINDIA',
        'BASF', 'BATAINDIA', 'BAYERCROP', 'BBL', 'BBTC', 'BCONCEPTS',
        'BDL', 'BEARDSELL', 'BEDMUTHA', 'BEML', 'BFINVEST', 'BFUTILITIE',
        'BGRENERGY', 'BHAGERIA', 'BHARATGEAR', 'BHARATRAS', 'BHARATW',
        'BHARTIARTL', 'BHEL', 'BIGBLOC', 'BIKAJI', 'BIRLACABLE', 'BLAL',
        'BLISSGVS', 'BLUEDART', 'BLUESTARCO', 'BODALCHEM', 'BOMDYEING',
        'BOROLTD', 'BOSCHINDIA', 'BRIGADE', 'BRITANNIA', 'BRNL', 'BROOKS',
        'BSE', 'BSHSL', 'BSOFT', 'BURNPUR', 'BUTTERFLY', 'BVCL', 'CAMPUS',
        'CAMS', 'CANARYS', 'CANDC', 'CANFINHOME', 'CANTABIL', 'CAPACITE',
        'CAPLIPOINT', 'CAPTRUST', 'CARBORUNIV', 'CARERATING', 'CARTRADE',
        'CARYSIL', 'CASTROLIND', 'CCHHL', 'CCL', 'CDSL', 'CEATLTD',
        
        # Next 150 small-caps
        'CELEBRITY', 'CENTENKA', 'CENTEXT', 'CENTRALBK', 'CENTRUM', 'CENTUM',
        'CERA', 'CEREBRAINT', 'CESC', 'CGCL', 'CHALET', 'CHAMBLFERT',
        'CHEMPLASTS', 'CHENNPETRO', 'CHERRYENGG', 'CHOICEIN', 'CHOLAHLDNG',
        'CIEINDIA', 'CIGNITITEC', 'CINELINE', 'CINEVISTA', 'CIPLA', 'CLEAN',
        'CLEDUCATE', 'CLNINDIA', 'CMICABLES', 'COCHINSHIP', 'COFFEEDAY',
        'COLPAL', 'COMPINFO', 'CONCORDBIO', 'CONFIPET', 'CONSOFINVT',
        'CONTROLPR', 'CORALFINAC', 'CORDSCABLE', 'COSMOFIRST', 'CRAFTSMAN',
        'CREATIVE', 'CREDITACC', 'CREST', 'CRISIL', 'CROMPTON', 'CROWN',
        'CSBBANK', 'CUPID', 'CYBERMEDIA', 'CYBERTECH', 'CYIENT', 'DAAWAT',
        'DABUR', 'DALBHARAT', 'DALMIASUG', 'DATAPATTNS', 'DATATICS', 'DBOL',
        'DBREALTY', 'DCB', 'DCBBANK', 'DCAL', 'DCMSHRIRAM', 'DCMNVL',
        'DCMFINSERV', 'DCMSHRIRAM', 'DCW', 'DECCANCE', 'DEEPAKFERT',
        'DEEPAKNTR', 'DEEPENR', 'DEEPIND', 'DELTACORP', 'DELTAMAG',
        'DENORA', 'DEVIT', 'DEVYANI', 'DGCONTENT', 'DHAMPURSUG', 'DHANBANK',
        'DHANI', 'DHANUKA', 'DHARAMSI', 'DHARSUGAR', 'DHRUV', 'DHUNINV',
        'DIAPOWER', 'DICIND', 'DIGISPICE', 'DIL', 'DISHTV', 'DIVGIITTS',
        'DIVISLAB', 'DIXON', 'DLF', 'DLINKINDIA', 'DMART', 'DMCC',
        'DNAMEDIA', 'DOLLAR', 'DOLPHIN', 'DOMS', 'DONEAR', 'DPABHUSHAN',
        'DPSCLTD', 'DPWIRES', 'DRCSYSTEMS', 'DREAMFOLKS', 'DREDGECORP',
        'DRREDDY', 'DSPIM', 'DTIL', 'DUCON', 'DWARKESH', 'DYNAMATECH',
        'DYNPRO', 'EASEMYTRIP', 'EASTSILK', 'EBBETF0431', 'ECLERX',
        'EDELWEISS', 'EDUCOMP', 'EICHERMOT', 'EIDPARRY', 'EIFFL', 'EIHAHOTELS',
        'EIHOTEL', 'EIMCOELECO', 'EKC', 'ELECON', 'ELECTCAST', 'ELECTHERM',
        'ELGIEQUIP', 'ELGIRUBCO', 'EMAMILTD', 'EMAMIPAP', 'EMAMIREAL',
        'EMBASSY', 'EMKAY', 'EMMBI', 'EMSLIMITED', 'EMUDHRA'
    ]
    
    # SECTORAL INDICES
    NIFTY_BANK = [
        'HDFCBANK', 'ICICIBANK', 'KOTAKBANK', 'AXISBANK', 'SBIN', 
        'INDUSINDBK', 'PNB', 'BANKBARODA', 'FEDERALBNK', 'IDFCFIRSTB',
        'BANDHANBNK', 'AUBANK'
    ]
    
    NIFTY_IT = [
        'TCS', 'INFY', 'HCLTECH', 'WIPRO', 'TECHM', 'LTIM', 'PERSISTENT',
        'COFORGE', 'MINDTREE', 'MPHASIS', 'LTTS', 'NAUKRI', 'OFSS'
    ]
    
    NIFTY_PHARMA = [
        'SUNPHARMA', 'DIVISLAB', 'CIPLA', 'DRREDDY', 'TORNTPHARM', 
        'ZYDUSLIFE', 'ALKEM', 'AUROPHARMA', 'LUPIN', 'GLENMARK',
        'BIOCON', 'IPCALAB', 'LAURUSLABS', 'NATCOPHARM', 'SANOFI'
    ]
    
    NIFTY_AUTO = [
        'MARUTI', 'TATAMOTORS', 'M&M', 'HEROMOTOCO', 'EICHERMOT', 
        'BAJAJ-AUTO', 'TVSMOTOR', 'MOTHERSON', 'ASHOKLEY', 'BHARATFORG',
        'APOLLOTYRE', 'BALKRISIND', 'CEAT', 'EXIDEIND', 'MRF'
    ]
    
    NIFTY_FMCG = [
        'HINDUNILVR', 'ITC', 'NESTLEIND', 'BRITANNIA', 'TATACONSUM',
        'DABUR', 'MARICO', 'GODREJCP', 'COLPAL', 'VBL', 'MCDOWELL-N',
        'UBL', 'PGHH', 'EMAMILTD', 'ZYDUSWELL', 'RADICO', 'JUBLFOOD'
    ]
    
    NIFTY_METAL = [
        'JSWSTEEL', 'TATASTEEL', 'HINDALCO', 'JINDALSTEL', 'VEDL',
        'SAIL', 'NMDC', 'NATIONALUM', 'COALINDIA', 'APLAPOLLO',
        'JSL', 'WELCORP', 'RATNAMANI', 'HINDZINC', 'MOIL'
    ]
    
    NIFTY_REALTY = [
        'DLF', 'GODREJPROP', 'OBEROIRLTY', 'PRESTIGE', 'SOBHA',
        'PHOENIXLTD', 'BRIGADE', 'MAHLIFE', 'IBREALEST', 'RAYMOND'
    ]
    
    NIFTY_ENERGY = [
        'RELIANCE', 'ONGC', 'POWERGRID', 'NTPC', 'COALINDIA', 
        'ADANIGREEN', 'ADANIPOWER', 'IOC', 'BPCL', 'GAIL',
        'TATAPOWER', 'TORNTPOWER', 'CESC', 'SJVN', 'NHPC'
    ]
    
    # Market Cap Categories
    LARGECAP = NIFTY_50 + NIFTY_NEXT_50  # Top 100
    MIDCAP = NIFTY_MIDCAP_150
    SMALLCAP = NIFTY_SMALLCAP_250
    
    # Index mapping
    INDEX_MAPPING = {
        'NIFTY_50': NIFTY_50,
        'NIFTY50': NIFTY_50,
        'NIFTY_NEXT_50': NIFTY_NEXT_50,
        'NIFTYNEXT50': NIFTY_NEXT_50,
        'NIFTY_100': NIFTY_50 + NIFTY_NEXT_50,
        'NIFTY100': NIFTY_50 + NIFTY_NEXT_50,
        'NIFTY_MIDCAP': NIFTY_MIDCAP_150,
        'NIFTY_MIDCAP_150': NIFTY_MIDCAP_150,
        'MIDCAP': NIFTY_MIDCAP_150,
        'NIFTY_SMALLCAP': NIFTY_SMALLCAP_250,
        'NIFTY_SMALLCAP_250': NIFTY_SMALLCAP_250,
        'SMALLCAP': NIFTY_SMALLCAP_250,
        'LARGECAP': LARGECAP,
        'NIFTY_BANK': NIFTY_BANK,
        'BANK': NIFTY_BANK,
        'NIFTY_IT': NIFTY_IT,
        'IT': NIFTY_IT,
        'NIFTY_PHARMA': NIFTY_PHARMA,
        'PHARMA': NIFTY_PHARMA,
        'NIFTY_AUTO': NIFTY_AUTO,
        'AUTO': NIFTY_AUTO,
        'NIFTY_FMCG': NIFTY_FMCG,
        'FMCG': NIFTY_FMCG,
        'NIFTY_METAL': NIFTY_METAL,
        'METAL': NIFTY_METAL,
        'NIFTY_REALTY': NIFTY_REALTY,
        'REALTY': NIFTY_REALTY,
        'NIFTY_ENERGY': NIFTY_ENERGY,
        'ENERGY': NIFTY_ENERGY,
        'ALL': [],  # Empty list means no filtering
        'NIFTY_500': LARGECAP + MIDCAP + SMALLCAP[:100],  # Top 500
    }
    
    def __init__(self, index_name: str = "ALL"):
        """
        Initialize index filter
        
        Args:
            index_name: Name of the index or market cap category
        """
        self.index_name = index_name.upper()
        self.symbols = self._get_index_symbols(self.index_name)
    
    def filter(self, data: pd.DataFrame) -> pd.DataFrame:
        """Apply index filter"""
        if 'symbol' not in data.columns or self.index_name == "ALL":
            return data
        
        filtered = data[data['symbol'].isin(self.symbols)]
        logger.debug(f"Index filter ({self.index_name}): {len(data)} -> {len(filtered)} stocks")
        return filtered
    
    def get_description(self) -> str:
        """Get filter description"""
        if self.index_name == "ALL":
            return "Index: ALL"
        return f"Index: {self.index_name} ({len(self.symbols)} stocks)"
    
    def _get_index_symbols(self, index_name: str) -> Set[str]:
        """Get symbols for the specified index"""
        if index_name in self.INDEX_MAPPING:
            return set(self.INDEX_MAPPING[index_name])
        
        # Log available indices if invalid index provided
        logger.warning(f"Unknown index: {index_name}. Available indices: {list(self.INDEX_MAPPING.keys())}")
        return set()
    
    @classmethod
    def get_available_indices(cls) -> List[str]:
        """Get list of all available indices"""
        return list(cls.INDEX_MAPPING.keys())
    
    @classmethod
    def get_index_info(cls, index_name: str) -> Dict[str, any]:
        """Get information about an index"""
        index_name = index_name.upper()
        if index_name not in cls.INDEX_MAPPING:
            return {"error": f"Unknown index: {index_name}"}
        
        symbols = cls.INDEX_MAPPING[index_name]
        return {
            "name": index_name,
            "count": len(symbols),
            "symbols": symbols[:10],  # First 10 as sample
            "category": cls._get_category(index_name)
        }
    
    @classmethod
    def _get_category(cls, index_name: str) -> str:
        """Get category of the index"""
        if index_name in ['NIFTY_50', 'NIFTY_NEXT_50', 'NIFTY_100', 'LARGECAP']:
            return "Large Cap"
        elif index_name in ['NIFTY_MIDCAP', 'NIFTY_MIDCAP_150', 'MIDCAP']:
            return "Mid Cap"
        elif index_name in ['NIFTY_SMALLCAP', 'NIFTY_SMALLCAP_250', 'SMALLCAP']:
            return "Small Cap"
        elif index_name in ['NIFTY_BANK', 'NIFTY_IT', 'NIFTY_PHARMA', 'NIFTY_AUTO', 
                            'NIFTY_FMCG', 'NIFTY_METAL', 'NIFTY_REALTY', 'NIFTY_ENERGY']:
            return "Sectoral"
        else:
            return "Mixed"


class MarketCapFilter(IFilterStrategy):
    """
    Filter stocks based on market capitalization category
    """
    
    def __init__(self, categories: List[str]):
        """
        Initialize market cap filter
        
        Args:
            categories: List of categories ['LARGE', 'MID', 'SMALL']
        """
        self.categories = [c.upper() for c in categories]
        self.symbols = self._get_symbols_for_categories()
    
    def filter(self, data: pd.DataFrame) -> pd.DataFrame:
        """Apply market cap filter"""
        if 'symbol' not in data.columns or not self.categories:
            return data
        
        filtered = data[data['symbol'].isin(self.symbols)]
        logger.debug(f"Market cap filter ({self.categories}): {len(data)} -> {len(filtered)} stocks")
        return filtered
    
    def get_description(self) -> str:
        """Get filter description"""
        return f"Market Cap: {', '.join(self.categories)}"
    
    def _get_symbols_for_categories(self) -> Set[str]:
        """Get symbols for selected market cap categories"""
        symbols = set()
        
        for category in self.categories:
            if category == 'LARGE':
                symbols.update(IndexFilter.LARGECAP)
            elif category == 'MID':
                symbols.update(IndexFilter.MIDCAP)
            elif category == 'SMALL':
                symbols.update(IndexFilter.SMALLCAP)
        
        return symbols


class SectorFilter(IFilterStrategy):
    """
    Filter stocks based on sector
    """
    
    SECTOR_INDICES = {
        'BANKING': 'NIFTY_BANK',
        'IT': 'NIFTY_IT',
        'PHARMA': 'NIFTY_PHARMA',
        'AUTO': 'NIFTY_AUTO',
        'FMCG': 'NIFTY_FMCG',
        'METAL': 'NIFTY_METAL',
        'REALTY': 'NIFTY_REALTY',
        'ENERGY': 'NIFTY_ENERGY'
    }
    
    def __init__(self, sectors: List[str]):
        """
        Initialize sector filter
        
        Args:
            sectors: List of sectors
        """
        self.sectors = [s.upper() for s in sectors]
        self.symbols = self._get_sector_symbols()
    
    def filter(self, data: pd.DataFrame) -> pd.DataFrame:
        """Apply sector filter"""
        if 'symbol' not in data.columns or not self.sectors:
            return data
        
        filtered = data[data['symbol'].isin(self.symbols)]
        logger.debug(f"Sector filter ({self.sectors}): {len(data)} -> {len(filtered)} stocks")
        return filtered
    
    def get_description(self) -> str:
        """Get filter description"""
        return f"Sectors: {', '.join(self.sectors)}"
    
    def _get_sector_symbols(self) -> Set[str]:
        """Get symbols for selected sectors"""
        symbols = set()
        
        for sector in self.sectors:
            if sector in self.SECTOR_INDICES:
                index_name = self.SECTOR_INDICES[sector]
                index_symbols = IndexFilter.INDEX_MAPPING.get(index_name, [])
                symbols.update(index_symbols)
        
        return symbols


class CompositeFilter(IFilterStrategy):
    """
    Composite filter that combines multiple filters
    Follows Composite pattern
    """
    
    def __init__(self, filters: List[IFilterStrategy]):
        self.filters = filters
    
    def filter(self, data: pd.DataFrame) -> pd.DataFrame:
        """Apply all filters in sequence"""
        filtered = data
        for filter_strategy in self.filters:
            filtered = filter_strategy.filter(filtered)
        return filtered
    
    def get_description(self) -> str:
        """Get combined description"""
        descriptions = [f.get_description() for f in self.filters]
        return " | ".join(descriptions)
    
    def add_filter(self, filter_strategy: IFilterStrategy):
        """Add a new filter to the composite"""
        self.filters.append(filter_strategy)
    
    def remove_filter(self, filter_strategy: IFilterStrategy):
        """Remove a filter from the composite"""
        if filter_strategy in self.filters:
            self.filters.remove(filter_strategy)


class CustomFilter(IFilterStrategy):
    """
    Custom filter with user-defined criteria
    """
    
    def __init__(self, filter_func, description: str):
        self.filter_func = filter_func
        self.description = description
    
    def filter(self, data: pd.DataFrame) -> pd.DataFrame:
        """Apply custom filter function"""
        return self.filter_func(data)
    
    def get_description(self) -> str:
        """Get filter description"""
        return self.description


# Factory for creating filter combinations
class FilterFactory:
    """Factory for creating common filter combinations"""
    
    @staticmethod
    def create_default_filters() -> List[IFilterStrategy]:
        """Create default filter set"""
        return [
            VolumeFilter(min_volume=100000),
            PriceRangeFilter(min_price=10, max_price=50000),
            DeliveryPercentFilter(min_delivery_percent=20)
        ]
    
    @staticmethod
    def create_quality_largecap_filters() -> List[IFilterStrategy]:
        """Create quality large cap filters"""
        return [
            IndexFilter("NIFTY_100"),
            VolumeFilter(min_volume=500000),
            DeliveryPercentFilter(min_delivery_percent=40)
        ]
    
    @staticmethod
    def create_midcap_opportunity_filters() -> List[IFilterStrategy]:
        """Create mid cap opportunity filters"""
        return [
            IndexFilter("MIDCAP"),
            VolumeFilter(min_volume=200000),
            DeliveryPercentFilter(min_delivery_percent=35),
            PriceRangeFilter(min_price=50, max_price=5000)
        ]
    
    @staticmethod
    def create_smallcap_value_filters() -> List[IFilterStrategy]:
        """Create small cap value filters"""
        return [
            IndexFilter("SMALLCAP"),
            VolumeFilter(min_volume=50000),
            DeliveryPercentFilter(min_delivery_percent=30),
            PriceRangeFilter(min_price=10, max_price=1000)
        ]
    
    @staticmethod
    def create_sector_filters(sector: str) -> List[IFilterStrategy]:
        """Create filters for specific sector"""
        return [
            SectorFilter([sector]),
            VolumeFilter(min_volume=100000),
            DeliveryPercentFilter(min_delivery_percent=25)
        ]
    
    @staticmethod
    def create_market_cap_filters(categories: List[str]) -> List[IFilterStrategy]:
        """Create filters for market cap categories"""
        return [
            MarketCapFilter(categories),
            VolumeFilter(min_volume=100000),
            DeliveryPercentFilter(min_delivery_percent=25)
        ]