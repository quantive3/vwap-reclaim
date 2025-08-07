# === Signal Detection Module ===
import pandas as pd
import numpy as np
from datetime import time  # noqa: F401
import hashlib  # noqa: F401

def detect_stretch_signal(df_rth_filled, params, debug_mode=False, silent_mode=False):
    """
    Detects stretch signals when SPY price moves beyond VWAP by ±0.3%.

    Parameters:
    - df_rth_filled: DataFrame containing SPY price and VWAP data.
    - params: Dictionary of parameters including stretch threshold.
    - debug_mode: Whether to enable debug outputs.
    - silent_mode: Whether to suppress non-debug outputs.

    Returns:
    - signals: DataFrame with stretch signals.
    """
    stretch_threshold = params['stretch_threshold']
    df_rth_filled['stretch_signal'] = (
        (df_rth_filled['close'] > df_rth_filled['vwap_running'] * (1 + stretch_threshold)) |
        (df_rth_filled['close'] < df_rth_filled['vwap_running'] * (1 - stretch_threshold))
    )
    df_rth_filled['percentage_stretch'] = (
        (df_rth_filled['close'] - df_rth_filled['vwap_running']) / df_rth_filled['vwap_running']
    ) * 100
    signals = df_rth_filled[df_rth_filled['stretch_signal']].copy()
      
    # Tag each stretch signal with "above" or "below"
    signals['stretch_label'] = np.where(
        signals['close'] > signals['vwap_running'], 'above', 'below'
    )
    
    # === Filter signals to within sweepable time range ===
    entry_start = params['entry_start_time']
    entry_end = params['entry_end_time']

    # Count signals before time filtering for diagnostic purposes
    signals_before_time_filter = len(signals)

    try:
        # Convert timestamp to time objects for filtering
        signals['ts_obj'] = signals['ts_raw'].dt.time
        
        # Check for NaN or invalid timestamps before filtering
        invalid_timestamps = signals['ts_raw'].isna().sum()
        if invalid_timestamps > 0 and debug_mode:
            if not silent_mode:
                print(f"⚠️ Found {invalid_timestamps} signals with invalid timestamps before time filtering")
            
        # Apply the time filter - core logic unchanged
        signals = signals[(signals['ts_obj'] >= entry_start) & (signals['ts_obj'] <= entry_end)]
        signals.drop(columns=['ts_obj'], inplace=True)
        signals = signals[signals['ts_raw'].notna()].sort_values("ts_raw").reset_index(drop=True)
        
    except Exception as e:
        # Capture errors in timestamp conversion or filtering
        error_msg = f"❌ Error during time filtering of signals: {str(e)}"
        if not silent_mode:
            print(error_msg)
        raise ValueError(error_msg)
    
    # Count signals after time filtering for diagnostic purposes
    signals_after_time_filter = len(signals)
    signals_dropped = signals_before_time_filter - signals_after_time_filter

    # Log time filtering results without affecting logic
    if debug_mode and signals_dropped > 0:
        print(f"ℹ️ Time filtering: {signals_dropped} signals were outside the {entry_start}-{entry_end} trading window")
        print(f"ℹ️ Signals before time filter: {signals_before_time_filter}, after: {signals_after_time_filter}")

    if debug_mode:
        print("\n🧹 Post-filter signal integrity check:")
        print(f"  NaT timestamps: {signals['ts_raw'].isna().sum()}")
        print(f"  Time ordered: {signals['ts_raw'].is_monotonic_increasing}")

        # Log the first 5 stretch labels for "above" and "below"
    if debug_mode:
        print("\nFirst 5 'Above' Stretch Signals:")
        print(signals[signals['stretch_label'] == 'above'][['timestamp', 'close', 'vwap_running', 'percentage_stretch', 'stretch_label']].head())
        print("\nFirst 5 'Below' Stretch Signals:")
        print(signals[signals['stretch_label'] == 'below'][['timestamp', 'close', 'vwap_running', 'percentage_stretch', 'stretch_label']].head())
    
    # Implement cooldown logic
    last_above_signal_time = None
    last_below_signal_time = None
    cooldown_period = pd.Timedelta(seconds=params['cooldown_period_seconds'])

    filtered_signals = []
    for _, row in signals.iterrows():
        current_time = row['ts_raw']
        if row['stretch_label'] == 'above':
            if last_above_signal_time is None or (current_time - last_above_signal_time) >= cooldown_period:
                filtered_signals.append(row)
                last_above_signal_time = current_time
        elif row['stretch_label'] == 'below':
            if last_below_signal_time is None or (current_time - last_below_signal_time) >= cooldown_period:
                filtered_signals.append(row)
                last_below_signal_time = current_time

    # Convert filtered signals to DataFrame
    processed_signals_df = pd.DataFrame(filtered_signals)
    
    # Additional diagnostic for cooldown filtering
    if debug_mode:
        signals_after_cooldown = len(processed_signals_df)
        signals_dropped_by_cooldown = len(signals) - signals_after_cooldown
        if signals_dropped_by_cooldown > 0:
            print(f"ℹ️ Cooldown filtering: {signals_dropped_by_cooldown} signals were dropped due to cooldown period")
            print(f"ℹ️ Signals before cooldown: {len(signals)}, after: {signals_after_cooldown}")

    if debug_mode:
        # Log the first 5 processed 'above' stretch signals
        print("\nFirst 5 Processed 'Above' Stretch Signals:")
        print(processed_signals_df[processed_signals_df['stretch_label'] == 'above'][['ts_raw', 'close', 'vwap_running', 'percentage_stretch', 'stretch_label']].head())
        
        # Log the first 5 processed 'below' stretch signals
        print("\nFirst 5 Processed 'Below' Stretch Signals:")
        print(processed_signals_df[processed_signals_df['stretch_label'] == 'below'][['ts_raw', 'close', 'vwap_running', 'percentage_stretch', 'stretch_label']].head())

    return processed_signals_df

def detect_partial_reclaims(df_rth_filled, stretch_signals, params, debug_mode=False, silent_mode=False):
    """
    For each stretch signal, detect if a partial reclaim toward VWAP occurs within the cooldown window.
    Returns stretch signals with reclaim metadata.
    
    Parameters:
    - df_rth_filled: DataFrame containing SPY price and VWAP data.
    - stretch_signals: DataFrame with detected stretch signals.
    - params: Dictionary of parameters including reclaim threshold.
    - debug_mode: Whether to enable debug outputs.
    - silent_mode: Whether to suppress non-debug outputs.
    
    Returns:
    - DataFrame with enriched signals including reclaim metadata.
    """
    reclaim_threshold = params['reclaim_threshold']
    cooldown_seconds = params['cooldown_period_seconds']
    enriched_signals = []

    for _, row in stretch_signals.iterrows():
        stretch_time = row['ts_raw']
        label = row['stretch_label']
        vwap_at_stretch = row['vwap_running']  # noqa: F841

        # Extract the reclaim window (up to 60 seconds ahead)
        reclaim_window = df_rth_filled[
            (df_rth_filled['ts_raw'] > stretch_time) &
            (df_rth_filled['ts_raw'] <= stretch_time + pd.Timedelta(seconds=cooldown_seconds))
        ].copy()

        # Define reclaim zone based on stretch direction
        if label == 'below':
            valid_reclaims = reclaim_window[
                reclaim_window['close'] >= reclaim_window['vwap_running'] * (1 - reclaim_threshold)
            ]
        elif label == 'above':
            valid_reclaims = reclaim_window[
                reclaim_window['close'] <= reclaim_window['vwap_running'] * (1 + reclaim_threshold)
            ]
        else:
            continue  # skip malformed

        # Filter reclaim window based on direction-specific threshold
        if not valid_reclaims.empty:
            first_reclaim = valid_reclaims.iloc[0]
            row['entry_intent'] = True
            row['reclaim_ts'] = first_reclaim['ts_raw']
            row['reclaim_price'] = first_reclaim['close']
            row['vwap_at_reclaim'] = first_reclaim['vwap_running']
        else:
            row['entry_intent'] = False
            row['reclaim_ts'] = pd.NaT
            row['reclaim_price'] = np.nan
            row['vwap_at_reclaim'] = np.nan

        enriched_signals.append(row)

    return pd.DataFrame(enriched_signals)