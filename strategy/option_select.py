# === Option Selection Module ===
# This module contains the option selection logic extracted from main.py
# It is a straight extraction and must remain behaviorally identical 
# to the original implementation unless intentionally changed.
import pandas as pd
import numpy as np

# Module-level placeholders for injected dependencies
_track_issue = None
_DEBUG_MODE = None

# Setter functions for dependency injection
def set_track_issue_function(func):
    """
    Set the track_issue function.
    
    Args:
        func: Function for tracking issues
    """
    global _track_issue
    _track_issue = func

def set_debug_mode(debug_mode):
    """
    Set the debug mode flag.
    
    Args:
        debug_mode: Boolean debug mode flag
    """
    global _DEBUG_MODE
    _DEBUG_MODE = debug_mode

def select_option_contract(entry_signal, df_chain, spy_price, params):
    """
    Select the appropriate option contract based on stretch direction and selection mode logic.
    
    Parameters:
    - entry_signal: DataFrame row with entry signal information
    - df_chain: DataFrame with option chain data
    - spy_price: Current SPY price at entry
    - params: Strategy parameters including option_selection_mode and strikes_depth
    
    Returns:
    - selected_contract: Dictionary with selected contract details
    """
    # Determine option type based on stretch direction
    stretch_direction = entry_signal['stretch_label']
    
    # Below VWAP stretch → buy call option
    # Above VWAP stretch → buy put option
    option_type = 'call' if stretch_direction == 'below' else 'put'
    
    # Get current date for filtering same-day expiry options
    current_date = entry_signal['ts_raw'].strftime('%Y-%m-%d')
    
    # Filter chain to only include the desired option type AND same-day expiry
    filtered_chain = df_chain[
        (df_chain['option_type'] == option_type) & 
        (df_chain['expiration_date'] == current_date)
    ].copy()
    
    if filtered_chain.empty:
        if params['debug_mode']:
            if not params.get('silent_mode', False):
                print(f"⚠️ No same-day expiry {option_type} options available in the chain")
            
        # If no same-day expiry options, check if we allow non-same-day expiry as fallback
        if not params.get('require_same_day_expiry', True):
            filtered_chain = df_chain[df_chain['option_type'] == option_type].copy()
            
            if filtered_chain.empty:
                if params['debug_mode']:
                    if not params.get('silent_mode', False):
                        print(f"⚠️ No {option_type} options available in the chain at all")
                return None
            else:
                if params['debug_mode']:
                    if not params.get('silent_mode', False):
                        print(f"ℹ️ Using non-same-day expiry options as fallback")
        else:
            if params['debug_mode']:
                if not params.get('silent_mode', False):
                    print(f"⚠️ Same-day expiry required but none available - skipping")
            return None
    
    # Calculate absolute difference between each strike and current price for ATM selection
    filtered_chain['abs_diff'] = (filtered_chain['strike_price'] - spy_price).abs()
    
    # Sort by absolute difference to find closest to ATM
    atm_chain = filtered_chain.sort_values('abs_diff')
    
    # Get option selection mode (itm, otm, or atm)
    option_selection_mode = params.get('option_selection_mode', 'itm').lower()
    
    # Get the target strike depth (how many strikes from ATM to go)
    strikes_depth = params.get('strikes_depth', 1)  # Default to 1 if not specified
    
    # Use selection mode logic paths
    if option_selection_mode == 'itm':
        # ===== ITM SELECTION LOGIC =====
        if option_type == 'call':
            # For calls, ITM means strike < price
            itm_contracts = atm_chain[atm_chain['strike_price'] < spy_price]
            if not itm_contracts.empty:
                # Sort by strike price in descending order (highest strike first)
                itm_sorted = itm_contracts.sort_values('strike_price', ascending=False)
                
                # Get the nth ITM strike based on depth parameter
                target_idx = min(strikes_depth - 1, len(itm_sorted) - 1)
                selected_contract = itm_sorted.iloc[target_idx]
                selection_mode_used = 'itm'
            else:
                # If no ITM contracts, fall back to ATM
                selected_contract = atm_chain.iloc[0]
                selection_mode_used = 'atm_fallback'
        else:  # put
            # For puts, ITM means strike > price
            itm_contracts = atm_chain[atm_chain['strike_price'] > spy_price]
            if not itm_contracts.empty:
                # Sort by strike price in ascending order (lowest strike first)
                itm_sorted = itm_contracts.sort_values('strike_price', ascending=True)
                
                # Get the nth ITM strike based on depth parameter
                target_idx = min(strikes_depth - 1, len(itm_sorted) - 1)
                selected_contract = itm_sorted.iloc[target_idx]
                selection_mode_used = 'itm'
            else:
                # If no ITM contracts, fall back to ATM
                selected_contract = atm_chain.iloc[0]
                selection_mode_used = 'atm_fallback'
    
    elif option_selection_mode == 'otm':
        # ===== OTM SELECTION LOGIC =====
        if option_type == 'call':
            # For calls, OTM means strike > price
            otm_contracts = atm_chain[atm_chain['strike_price'] > spy_price]
            if not otm_contracts.empty:
                # Sort by strike price in ascending order (lowest strike first)
                otm_sorted = otm_contracts.sort_values('strike_price', ascending=True)
                
                # Get the nth OTM strike based on depth parameter
                target_idx = min(strikes_depth - 1, len(otm_sorted) - 1)
                selected_contract = otm_sorted.iloc[target_idx]
                selection_mode_used = 'otm'
            else:
                # If no OTM contracts, fall back to ATM
                selected_contract = atm_chain.iloc[0]
                selection_mode_used = 'atm_fallback'
        else:  # put
            # For puts, OTM means strike < price
            otm_contracts = atm_chain[atm_chain['strike_price'] < spy_price]
            if not otm_contracts.empty:
                # Sort by strike price in descending order (highest strike first)
                otm_sorted = otm_contracts.sort_values('strike_price', ascending=False)
                
                # Get the nth OTM strike based on depth parameter
                target_idx = min(strikes_depth - 1, len(otm_sorted) - 1)
                selected_contract = otm_sorted.iloc[target_idx]
                selection_mode_used = 'otm'
            else:
                # If no OTM contracts, fall back to ATM
                selected_contract = atm_chain.iloc[0]
                selection_mode_used = 'atm_fallback'
    
    elif option_selection_mode == 'atm':
        # ===== ATM SELECTION LOGIC =====
        # Only select exact price match for ATM (strict definition)
        exact_match = atm_chain[atm_chain['strike_price'] == spy_price]
        if not exact_match.empty:
            selected_contract = exact_match.iloc[0]
            selection_mode_used = 'atm_exact'
        else:
            # No exact ATM match found, return None to skip this contract
            if params['debug_mode']:
                if not params.get('silent_mode', False):
                    print(f"⚠️ ATM mode requested but no exact match to {spy_price} found - skipping")
            return None
    
    else:
        # Invalid selection mode, default to ITM
        if params['debug_mode']:
            if not params.get('silent_mode', False):
                print(f"⚠️ Invalid option_selection_mode: {option_selection_mode}. Defaulting to 'itm'")
        
        # Reuse ITM logic as default case
        if option_type == 'call':
            itm_contracts = atm_chain[atm_chain['strike_price'] < spy_price]
            if not itm_contracts.empty:
                itm_sorted = itm_contracts.sort_values('strike_price', ascending=False)
                target_idx = min(strikes_depth - 1, len(itm_sorted) - 1)
                selected_contract = itm_sorted.iloc[target_idx]
                selection_mode_used = 'itm_default'
            else:
                selected_contract = atm_chain.iloc[0]
                selection_mode_used = 'atm_fallback'
        else:  # put
            itm_contracts = atm_chain[atm_chain['strike_price'] > spy_price]
            if not itm_contracts.empty:
                itm_sorted = itm_contracts.sort_values('strike_price', ascending=True)
                target_idx = min(strikes_depth - 1, len(itm_sorted) - 1)
                selected_contract = itm_sorted.iloc[target_idx]
                selection_mode_used = 'itm_default'
            else:
                selected_contract = atm_chain.iloc[0]
                selection_mode_used = 'atm_fallback'
    
    # Define moneyness states for the contract
    is_atm = selected_contract['strike_price'] == spy_price
    is_itm = ((option_type == 'call' and selected_contract['strike_price'] < spy_price) or
             (option_type == 'put' and selected_contract['strike_price'] > spy_price))
    is_otm = not (is_atm or is_itm)
    
    # Create a dictionary with only the necessary contract details
    contract_details = {
        'ticker': selected_contract['ticker'],
        'option_type': option_type,
        'strike_price': selected_contract['strike_price'],
        'expiration_date': selected_contract.get('expiration_date', None),
        'abs_diff': selected_contract['abs_diff'],
        'is_atm': is_atm,
        'is_itm': is_itm,
        'is_otm': is_otm,
        'is_same_day_expiry': selected_contract.get('expiration_date', '') == current_date,
        'selection_mode': option_selection_mode,
        'selection_mode_used': selection_mode_used,
        'strikes_depth': strikes_depth,
        'shares_per_contract': selected_contract.get('shares_per_contract', 100)  # Default to 100 if not available
    }
    
    # Add warnings for shares_per_contract
    # Warning for missing shares_per_contract
    if 'shares_per_contract' not in selected_contract:
        missing_shares_msg = f"Missing shares_per_contract for {contract_details['ticker']} - using default of 100"
        if params['debug_mode']:
            if not params.get('silent_mode', False):
                print(f"⚠️ {missing_shares_msg}")
        _track_issue("warnings", "shares_per_contract_missing", missing_shares_msg, date=current_date)
    
    # Warning for non-standard contract size
    if contract_details['shares_per_contract'] != 100:
        non_standard_msg = f"Non-standard contract size detected: {contract_details['shares_per_contract']} shares for {contract_details['ticker']}"
        if params['debug_mode']:
            if not params.get('silent_mode', False):
                print(f"⚠️ {non_standard_msg}")
        _track_issue("warnings", "non_standard_contract_size", non_standard_msg, date=current_date)
    
    if params['debug_mode']:
        # Determine actual moneyness status for display
        if is_atm:
            moneyness_status = "ATM"
        elif is_itm:
            moneyness_status = "ITM"
        else:
            moneyness_status = "OTM"
            
        expiry_status = "Same-day expiry" if contract_details['is_same_day_expiry'] else "Future expiry"
        
        print(f"✅ Selected {option_type.upper()} option: {contract_details['ticker']} with strike {contract_details['strike_price']} ({moneyness_status}, {expiry_status})")
        print(f"   Underlying price: {spy_price}, Strike diff: {contract_details['abs_diff']:.4f}")
        print(f"   Entry timestamp: {entry_signal['reclaim_ts']}, Expiration date: {contract_details['expiration_date']}")
    
    if _DEBUG_MODE:
        print(f"DEBUG: Stretch direction: {entry_signal['stretch_label']}")
        print(f"DEBUG: Selected option type: {option_type}")
        print(f"DEBUG: SPY price: {spy_price}, Strike: {contract_details['strike_price']}")
        print(f"DEBUG: Is ATM: {contract_details['is_atm']}, Is ITM: {contract_details['is_itm']}")
        print(f"   Selection mode: {option_selection_mode.upper()}, Actual mode used: {selection_mode_used}")
        print(f"   Strike depth: {strikes_depth} strikes from ATM")
    
    return contract_details