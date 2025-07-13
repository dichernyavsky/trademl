import requests
import pandas as pd
import json
from typing import List, Dict, Optional
from datetime import datetime
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BinanceSymbolsFetcher:
    """
    Fetches and manages cryptocurrency symbols from Binance API.
    """
    
    def __init__(self):
        self.api_url = "https://api.binance.com/api/v3/exchangeInfo"
        self.symbols_cache = None
        self.last_update = None
    
    def fetch_all_symbols(self, force_refresh: bool = False) -> pd.DataFrame:
        """
        Fetch all available trading symbols from Binance.
        
        Args:
            force_refresh: Whether to force refresh the cache
            
        Returns:
            DataFrame with symbol information
        """
        if not force_refresh and self.symbols_cache is not None:
            logger.info("Using cached symbols data")
            return self.symbols_cache
        
        try:
            logger.info("Fetching symbols from Binance API...")
            response = requests.get(self.api_url, timeout=30)
            response.raise_for_status()
            data = response.json()
            
            if 'symbols' not in data:
                raise ValueError("Invalid response format from Binance API")
            
            # Convert to DataFrame
            symbols_df = pd.DataFrame(data['symbols'])
            
            # Filter for active symbols only
            active_symbols = symbols_df[symbols_df['status'] == 'TRADING'].copy()
            
            # Extract base and quote assets
            active_symbols['baseAsset'] = active_symbols['baseAsset'].str.upper()
            active_symbols['quoteAsset'] = active_symbols['quoteAsset'].str.upper()
            
            # Add symbol type
            active_symbols['symbolType'] = active_symbols['quoteAsset'].apply(self._categorize_symbol)
            
            # Cache the results
            self.symbols_cache = active_symbols
            self.last_update = datetime.now()
            
            logger.info(f"Fetched {len(active_symbols)} active trading symbols")
            return active_symbols
            
        except Exception as e:
            logger.error(f"Failed to fetch symbols: {e}")
            raise
    
    def _categorize_symbol(self, quote_asset: str) -> str:
        """Categorize symbol based on quote asset."""
        if quote_asset == 'USDT':
            return 'USDT_PAIR'
        elif quote_asset == 'BTC':
            return 'BTC_PAIR'
        elif quote_asset == 'ETH':
            return 'ETH_PAIR'
        elif quote_asset == 'BNB':
            return 'BNB_PAIR'
        elif quote_asset == 'BUSD':
            return 'BUSD_PAIR'
        elif quote_asset == 'USDC':
            return 'USDC_PAIR'
        else:
            return 'OTHER'
    
    def get_usdt_pairs(self, force_refresh: bool = False) -> List[str]:
        """
        Get all USDT trading pairs.
        
        Args:
            force_refresh: Whether to force refresh the cache
            
        Returns:
            List of USDT symbol names
        """
        symbols_df = self.fetch_all_symbols(force_refresh)
        usdt_pairs = symbols_df[symbols_df['quoteAsset'] == 'USDT']['symbol'].tolist()
        return sorted(usdt_pairs)
    
    def get_btc_pairs(self, force_refresh: bool = False) -> List[str]:
        """
        Get all BTC trading pairs.
        
        Args:
            force_refresh: Whether to force refresh the cache
            
        Returns:
            List of BTC symbol names
        """
        symbols_df = self.fetch_all_symbols(force_refresh)
        btc_pairs = symbols_df[symbols_df['quoteAsset'] == 'BTC']['symbol'].tolist()
        return sorted(btc_pairs)
    
    def get_eth_pairs(self, force_refresh: bool = False) -> List[str]:
        """
        Get all ETH trading pairs.
        
        Args:
            force_refresh: Whether to force refresh the cache
            
        Returns:
            List of ETH symbol names
        """
        symbols_df = self.fetch_all_symbols(force_refresh)
        eth_pairs = symbols_df[symbols_df['quoteAsset'] == 'ETH']['symbol'].tolist()
        return sorted(eth_pairs)
    
    def get_symbols_by_quote_asset(self, quote_asset: str, force_refresh: bool = False) -> List[str]:
        """
        Get all symbols for a specific quote asset.
        
        Args:
            quote_asset: Quote asset (e.g., 'USDT', 'BTC', 'ETH')
            force_refresh: Whether to force refresh the cache
            
        Returns:
            List of symbol names
        """
        symbols_df = self.fetch_all_symbols(force_refresh)
        pairs = symbols_df[symbols_df['quoteAsset'] == quote_asset.upper()]['symbol'].tolist()
        return sorted(pairs)
    
    def get_symbol_info(self, symbol: str, force_refresh: bool = False) -> Optional[Dict]:
        """
        Get detailed information for a specific symbol.
        
        Args:
            symbol: Symbol name (e.g., 'BTCUSDT')
            force_refresh: Whether to force refresh the cache
            
        Returns:
            Dictionary with symbol information or None if not found
        """
        symbols_df = self.fetch_all_symbols(force_refresh)
        symbol_info = symbols_df[symbols_df['symbol'] == symbol]
        
        if len(symbol_info) == 0:
            return None
        
        return symbol_info.iloc[0].to_dict()
    
    def get_popular_symbols(self, min_volume: Optional[float] = None) -> List[str]:
        """
        Get popular symbols (can be extended with volume data).
        
        Args:
            min_volume: Minimum volume threshold (if implemented)
            
        Returns:
            List of popular symbol names
        """
        # Top USDT pairs by market cap and popularity
        popular_symbols = [
            'BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'SOLUSDT', 'ADAUSDT',
            'XRPUSDT', 'DOTUSDT', 'DOGEUSDT', 'AVAXUSDT', 'MATICUSDT',
            'LINKUSDT', 'UNIUSDT', 'LTCUSDT', 'BCHUSDT', 'ATOMUSDT',
            'NEARUSDT', 'FTMUSDT', 'ALGOUSDT', 'VETUSDT', 'ICPUSDT'
        ]
        
        # Filter by available symbols
        available_symbols = set(self.get_usdt_pairs())
        return [s for s in popular_symbols if s in available_symbols]
    
    def get_top_usdt_pairs(self, count: int = 20) -> List[str]:
        """
        Get top USDT pairs by popularity and market relevance.
        
        Args:
            count: Number of top pairs to return
            
        Returns:
            List of top USDT symbol names
        """
        # Top USDT pairs by market cap and trading volume
        top_usdt_pairs = [
            'BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'SOLUSDT', 'ADAUSDT',
            'XRPUSDT', 'DOTUSDT', 'DOGEUSDT', 'AVAXUSDT', 'MATICUSDT',
            'LINKUSDT', 'UNIUSDT', 'LTCUSDT', 'BCHUSDT', 'ATOMUSDT',
            'NEARUSDT', 'FTMUSDT', 'ALGOUSDT', 'VETUSDT', 'ICPUSDT',
            'FILUSDT', 'APTUSDT', 'HBARUSDT', 'OPUSDT', 'ARBUSDT',
            'MKRUSDT', 'INJUSDT', 'RUNEUSDT', 'IMXUSDT', 'STXUSDT'
        ]
        
        # Filter by available symbols and return requested count
        available_symbols = set(self.get_usdt_pairs())
        available_top_pairs = [s for s in top_usdt_pairs if s in available_symbols]
        
        return available_top_pairs[:count]
    
    def save_symbols_to_file(self, filepath: str, force_refresh: bool = False):
        """
        Save symbols data to a file.
        
        Args:
            filepath: Path to save the file
            force_refresh: Whether to force refresh the cache
        """
        symbols_df = self.fetch_all_symbols(force_refresh)
        
        if filepath.endswith('.csv'):
            symbols_df.to_csv(filepath, index=False)
        elif filepath.endswith('.json'):
            symbols_df.to_json(filepath, orient='records', indent=2)
        elif filepath.endswith('.parquet'):
            symbols_df.to_parquet(filepath, index=False)
        else:
            raise ValueError("Unsupported file format. Use .csv, .json, or .parquet")
        
        logger.info(f"Saved {len(symbols_df)} symbols to {filepath}")
    
    def get_statistics(self, force_refresh: bool = False) -> Dict:
        """
        Get statistics about available symbols.
        
        Args:
            force_refresh: Whether to force refresh the cache
            
        Returns:
            Dictionary with statistics
        """
        symbols_df = self.fetch_all_symbols(force_refresh)
        
        stats = {
            'total_symbols': len(symbols_df),
            'quote_assets': symbols_df['quoteAsset'].value_counts().to_dict(),
            'symbol_types': symbols_df['symbolType'].value_counts().to_dict(),
            'last_update': self.last_update.isoformat() if self.last_update else None
        }
        
        return stats


# Convenience functions
def get_all_binance_symbols(force_refresh: bool = False) -> pd.DataFrame:
    """
    Get all available Binance symbols.
    
    Args:
        force_refresh: Whether to force refresh the cache
        
    Returns:
        DataFrame with symbol information
    """
    fetcher = BinanceSymbolsFetcher()
    return fetcher.fetch_all_symbols(force_refresh)


def get_usdt_pairs(force_refresh: bool = False) -> List[str]:
    """
    Get all USDT trading pairs.
    
    Args:
        force_refresh: Whether to force refresh the cache
        
    Returns:
        List of USDT symbol names
    """
    fetcher = BinanceSymbolsFetcher()
    return fetcher.get_usdt_pairs(force_refresh)


def get_popular_symbols() -> List[str]:
    """
    Get popular trading symbols.
    
    Returns:
        List of popular symbol names
    """
    fetcher = BinanceSymbolsFetcher()
    return fetcher.get_popular_symbols()


def save_symbols_list(filepath: str = "data/binance_symbols.csv", force_refresh: bool = False):
    """
    Save all symbols to a file.
    
    Args:
        filepath: Path to save the file
        force_refresh: Whether to force refresh the cache
    """
    fetcher = BinanceSymbolsFetcher()
    fetcher.save_symbols_to_file(filepath, force_refresh)


if __name__ == "__main__":
    # Example usage
    print("Fetching Binance symbols...")
    
    fetcher = BinanceSymbolsFetcher()
    
    # Get all symbols
    symbols_df = fetcher.fetch_all_symbols()
    
    # Get statistics
    stats = fetcher.get_statistics()
    
    print(f"\nSymbols Statistics:")
    print(f"Total symbols: {stats['total_symbols']}")
    print(f"Quote assets: {stats['quote_assets']}")
    
    # Get USDT pairs
    usdt_pairs = fetcher.get_usdt_pairs()
    print(f"\nUSDT pairs count: {len(usdt_pairs)}")
    print(f"First 10 USDT pairs: {usdt_pairs[:10]}")
    
    # Get popular symbols
    popular = fetcher.get_popular_symbols()
    print(f"\nPopular symbols: {popular}")
    
    # Save to file
    fetcher.save_symbols_to_file("data/binance_symbols.csv") 