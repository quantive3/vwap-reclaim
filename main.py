# === STEP 1: Mount Google Drive for caching ===
from google.colab import drive
drive.mount('/content/drive')

# === STEP 2: Manual API Key Input ===
API_KEY = input("🔑 Enter your Polygon API key: ").strip()

# === STEP 3: Imports and setup ===
import pandas as pd
import requests
import os
from datetime import time
import hashlib
import numpy as np

# === DEBUG FLAG ===
DEBUG_MODE = True  # Set to False to skip alignment diagnostics

start_date = "2023-01-23"
end_date = "2023-01-27"
business_days = pd.date_range(start=start_date, end=end_date, freq="B")
ticker = "SPY"

# === STEP 4: Caching paths ===
CACHE_DIR = "/content/drive/MyDrive/polygon_cache"
SPY_DIR = os.path.join(CACHE_DIR, "spy")
CHAIN_DIR = os.path.join(CACHE_DIR, "chain")
OPTION_DIR = os.path.join(CACHE_DIR, "option")

for d in [SPY_DIR, CHAIN_DIR, OPTION_DIR]:
    os.makedirs(d, exist_ok=True)

# === STEP 6: Define Parameters ===
PARAMS = {
    'stretch_threshold': 0.003,  # 0.3%
    'entry_start_time': time(9, 30),
    'entry_end_time': time(16, 0),  
}

# === STEP 7: Stretch Signal Detection ===
def detect_stretch_signal(df_rth_filled, params):
    """
    Detects stretch signals when SPY price moves beyond VWAP by ±0.3%.

    Parameters:
    - df_rth_filled: DataFrame containing SPY price and VWAP data.
    - params: Dictionary of parameters including stretch threshold.

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
    
    # Log the first 5 stretch signals
#    if DEBUG_MODE:
#        print("\nFirst 5 Stretch Signals:")
#        print(signals[['timestamp', 'close', 'vwap_running', 'percentage_stretch']].head())
    
    # Tag each stretch signal with "above" or "below"
    signals['stretch_label'] = np.where(
        signals['close'] > signals['vwap_running'], 'above', 'below'
    )
    
    # === Filter signals to within sweepable time range ===
    entry_start = params['entry_start_time']
    entry_end = params['entry_end_time']

    signals['ts_obj'] = signals['ts_raw'].dt.time
    signals = signals[(signals['ts_obj'] >= entry_start) & (signals['ts_obj'] <= entry_end)]
    signals.drop(columns=['ts_obj'], inplace=True)

        # Log the first 5 stretch labels for "above" and "below"
    if DEBUG_MODE:
        print("\nFirst 5 'Above' Stretch Signals:")
        print(signals[signals['stretch_label'] == 'above'][['timestamp', 'close', 'vwap_running', 'percentage_stretch', 'stretch_label']].head())
        print("\nFirst 5 'Below' Stretch Signals:")
        print(signals[signals['stretch_label'] == 'below'][['timestamp', 'close', 'vwap_running', 'percentage_stretch', 'stretch_label']].head())
    
    return signals

# === STEP 8: Backtest loop ===
for date_obj in business_days:
    date = date_obj.strftime("%Y-%m-%d")
    print(f"\n📅 Processing {date}...")

    try:
        # === STEP 5a: Load or pull SPY OHLCV ===
        spy_path = os.path.join(SPY_DIR, f"{ticker}_{date}.pkl")
        if os.path.exists(spy_path):
            df_rth_filled = pd.read_pickle(spy_path)
            print("📂 SPY data loaded from cache.")
        else:
            base_url = f"https://api.polygon.io/v2/aggs/ticker/{ticker}/range/1/second/{date}/{date}"
            headers = {"Authorization": f"Bearer {API_KEY}"}

            all_results = []
            cursor = None
            while True:
                url = f"{base_url}?adjusted=true&limit=50000"
                if cursor:
                    url += f"&cursor={cursor}"
                response = requests.get(url, headers=headers)
                if response.status_code != 200:
                    raise Exception(f"SPY price request failed: {response.status_code}")

                json_data = response.json()
                results = json_data.get("results", [])
                all_results.extend(results)

                if "next_url" in json_data:
                    cursor = json_data["next_url"].split("cursor=")[-1]
                else:
                    break

            if not all_results:
                print(f"⚠️ No SPY data for {date} — skipping.")
                continue

            df_raw = pd.DataFrame(all_results)
            df_raw["timestamp"] = pd.to_datetime(df_raw["t"], unit="ms", utc=True).dt.tz_convert("US/Eastern")
            df_raw.rename(columns={
                "o": "open", "h": "high", "l": "low", "c": "close",
                "v": "volume", "vw": "vw", "n": "trades"
            }, inplace=True)
            df_raw = df_raw[["timestamp", "open", "high", "low", "close", "volume", "vw", "trades"]]

            df_rth = df_raw[
                (df_raw["timestamp"].dt.time >= time(9, 30)) &
                (df_raw["timestamp"].dt.time <= time(16, 0))
            ].sort_values("timestamp").reset_index(drop=True)

            start_time = pd.Timestamp(f"{date} 09:30:00", tz="US/Eastern")
            end_time = pd.Timestamp(f"{date} 16:00:00", tz="US/Eastern")
            full_index = pd.date_range(start=start_time, end=end_time, freq="1s", tz="US/Eastern")

            df_rth_filled = df_rth.set_index("timestamp").reindex(full_index).ffill().reset_index()
            df_rth_filled.rename(columns={"index": "timestamp"}, inplace=True)
            df_rth_filled["ts_raw"] = df_rth_filled["timestamp"]
            df_rth_filled["timestamp"] = df_rth_filled["timestamp"].dt.strftime("%Y-%m-%d %H:%M:%S %Z")
            # === Updated VWAP using volume-weighted price (vw) ===
            df_rth_filled["cum_pv"] = (df_rth_filled["vw"] * df_rth_filled["volume"]).cumsum()
            df_rth_filled["cum_vol"] = df_rth_filled["volume"].cumsum()
            df_rth_filled["vwap_running"] = df_rth_filled["cum_pv"] / df_rth_filled["cum_vol"]

            if df_rth_filled["vwap_running"].isna().any():
                raise ValueError("❌ NaNs detected in vwap_running — check data or ffill logic")

            if not df_rth_filled["vwap_running"].apply(lambda x: pd.notna(x) and np.isfinite(x)).all():
                raise ValueError("❌ Non-finite values (inf/-inf) in vwap_running")



            df_rth_filled.to_pickle(spy_path)
            print("💾 SPY data pulled and cached.")

        # === STEP 5b: Load or pull option chain ===
        chain_path = os.path.join(CHAIN_DIR, f"{ticker}_chain_{date}.pkl")
        if os.path.exists(chain_path):
            df_chain = pd.read_pickle(chain_path)
            print("📂 Option chain loaded from cache.")
        else:
            def fetch_chain(contract_type):
                url = (
                    f"https://api.polygon.io/v3/reference/options/contracts"
                    f"?underlying_ticker={ticker}"
                    f"&contract_type={contract_type}"
                    f"&expiration_date={date}"
                    f"&as_of={date}"
                    f"&order=asc"
                    f"&limit=1000"
                    f"&sort=ticker"
                    f"&apiKey={API_KEY}"
                )
                resp = requests.get(url)
                if resp.status_code != 200:
                    raise Exception(f"{contract_type.upper()} request failed: {resp.status_code}")
                df = pd.DataFrame(resp.json().get("results", []))
                df["option_type"] = contract_type
                return df

            df_calls = fetch_chain("call")
            df_puts = fetch_chain("put")
            df_chain = pd.concat([df_calls, df_puts], ignore_index=True)
            df_chain["ticker_clean"] = df_chain["ticker"].str.replace("O:", "", regex=False)
            df_chain.to_pickle(chain_path)
            print("💾 Option chain pulled and cached.")

        if df_chain.empty:
            print(f"⚠️ No option chain data for {date} — skipping.")
            continue

        # === STEP 5c: Select ATM contract ===
        spy_open_price = df_rth_filled["close"].iloc[0]
        df_calls_only = df_chain[df_chain["option_type"] == "call"].copy()
        df_calls_only["abs_diff"] = (df_calls_only["strike_price"] - spy_open_price).abs()
        atm_call = df_calls_only.sort_values("abs_diff").iloc[0]
        option_ticker = atm_call["ticker"]

        # === STEP 5d: Load or pull option price data ===
        option_path = os.path.join(OPTION_DIR, f"{date}_{option_ticker.replace(':', '')}.pkl")
        if os.path.exists(option_path):
            df_option_rth = pd.read_pickle(option_path)
            print("📂 Option price data loaded from cache.")
        else:
            option_url = (
                f"https://api.polygon.io/v2/aggs/ticker/{option_ticker}/range/1/second/"
                f"{date}/{date}?adjusted=true&sort=asc&limit=50000&apiKey={API_KEY}"
            )
            resp = requests.get(option_url)
            option_results = resp.json().get("results", [])
            df_option = pd.DataFrame(option_results)

            if df_option.empty:
                print(f"⚠️ No option price data for {option_ticker} on {date} — skipping.")
                continue

            df_option["timestamp"] = pd.to_datetime(df_option["t"], unit="ms", utc=True).dt.tz_convert("US/Eastern")
            df_option.rename(columns={
                "o": "open", "h": "high", "l": "low", "c": "close",
                "v": "volume", "vw": "vwap", "n": "trades"
            }, inplace=True)
            df_option = df_option[["timestamp", "open", "high", "low", "close", "volume", "vwap", "trades"]]

            df_option_rth = df_option[
                (df_option["timestamp"].dt.time >= time(9, 30)) &
                (df_option["timestamp"].dt.time <= time(16, 0))
            ].sort_values("timestamp").reset_index(drop=True)

            df_option_rth.to_pickle(option_path)
            print("💾 Option price data pulled and cached.")

        # === STEP 5e: Timestamp alignment check ===
        df_option_aligned = df_option_rth.set_index("timestamp").reindex(df_rth_filled["ts_raw"]).ffill().reset_index()
        df_option_aligned.rename(columns={"index": "ts_raw"}, inplace=True)

        mismatch_count = (~df_option_aligned["ts_raw"].eq(df_rth_filled["ts_raw"])).sum()
        print(f"🧪 Timestamp mismatches: {mismatch_count}")

        if DEBUG_MODE:
            print(f"\n⏱️ SPY rows: {len(df_rth_filled)}")
            print(f"⏱️ OPT rows: {len(df_option_aligned)}")

            def hash_timestamps(df):
                return hashlib.md5("".join(df["ts_raw"].astype(str)).encode()).hexdigest()

            print(f"\n🔐 SPY hash:  {hash_timestamps(df_rth_filled)}")
            print(f"🔐 OPT hash:  {hash_timestamps(df_option_aligned)}")

        if mismatch_count > 0:
            print(f"⚠️ Timestamp mismatch in {mismatch_count} rows — skipping.")
            continue

        print(f"✅ {date} — Data loaded and aligned successfully.")

        # === Insert strategy logic here ===
        stretch_signals = detect_stretch_signal(df_rth_filled, PARAMS)
#        if DEBUG_MODE:
#             print(f"🔍 Detected {len(stretch_signals)} stretch signals on {date}.")

        if DEBUG_MODE:
            # Log the daily breakdown of stretch signals
            above_count = len(stretch_signals[stretch_signals['stretch_label'] == 'above'])
            below_count = len(stretch_signals[stretch_signals['stretch_label'] == 'below'])
            total_count = len(stretch_signals)
            print(f"🔍 Detected {total_count} stretch signals on {date} (Above: {above_count}, Below: {below_count}).")

    except Exception as e:
        print(f"❌ {date} — Error: {str(e)}")
        continue
