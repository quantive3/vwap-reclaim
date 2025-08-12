# === DataLoader Class Module ===
# Contains the DataLoader class
from strategy.data import (
    load_spy_data, 
    load_chain_data, 
    load_option_data
)

class DataLoader:
    """
    Data loading class that bundles SPY, option chain, and option price data loading.
    This is a pure wrapper around existing functions to maintain exact functional equivalence.
    """
    
    def __init__(self, api_key, cache_dir, params, debug_mode=False, silent_mode=False):
        """
        Initialize the DataLoader with configuration parameters.
        
        Args:
            api_key (str): Polygon API key
            cache_dir (str): Base cache directory path
            params (dict): Strategy parameters dictionary
            debug_mode (bool): Whether to enable debug output
            silent_mode (bool): Whether to suppress non-debug output
        """
        self.api_key = api_key
        self.cache_dir = cache_dir
        self.params = params
        self.debug_mode = debug_mode
        self.silent_mode = silent_mode
    
    def load_spy(self, date):
        """
        Load SPY price data for a given date.
        
        Args:
            date (str): Date string in format 'YYYY-MM-DD'
            
        Returns:
            pd.DataFrame: DataFrame with SPY price and VWAP data for the specified date
        """
        return load_spy_data(date, self.cache_dir, self.api_key, self.params, debug_mode=self.debug_mode)
    
    def load_chain(self, date):
        """
        Load option chain data for a given date.
        
        Args:
            date (str): Date string in format 'YYYY-MM-DD'
            
        Returns:
            pd.DataFrame: DataFrame with option chain data for the specified date
        """
        return load_chain_data(date, self.cache_dir, self.api_key, self.params, debug_mode=self.debug_mode)
    
    def load_option(self, ticker, date, df_rth_filled, signal_idx=None):
        """
        Load and process option price data for a given option ticker and date.
        
        Args:
            ticker (str): The option ticker symbol to load data for
            date (str): Date string in format 'YYYY-MM-DD'
            df_rth_filled (pd.DataFrame): DataFrame with SPY price data for timestamp alignment
            signal_idx (int, optional): Optional signal index for debug output
            
        Returns:
            tuple: (df_option_aligned, option_entry_price, status) where:
                - df_option_aligned: DataFrame with aligned option price data
                - option_entry_price: The entry price for the option (or None if no valid price found)
                - status: Dictionary with status information
        """
        return load_option_data(ticker, date, self.cache_dir, df_rth_filled, self.api_key, self.params, signal_idx=signal_idx)