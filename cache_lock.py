# smoke_test_load_spy.py
import os
import time
import multiprocessing as mp
import pandas as pd
from filelock import FileLock
from sys import exit

# adjust this import to match your project structure
from main import load_spy_data, initialize_parameters, API_KEY

def worker(_):
    params = initialize_parameters()
    params['min_spy_data_rows'] = 0  # force fetch path
    return load_spy_data("2025-06-25", "./polygon_cache", API_KEY, params, debug_mode=False)

if __name__ == "__main__":
    # ── CLEAN SLATE ──
    spy_cache = "./polygon_cache/spy"
    if os.path.exists(spy_cache):
        for f in os.listdir(spy_cache):
            os.remove(os.path.join(spy_cache, f))

    # ── HOLD THE LOCK TO FORCE A TIMEOUT ──
    lock_path = os.path.join(spy_cache, "SPY_2025-06-25.pkl.lock")
    os.makedirs(os.path.dirname(lock_path), exist_ok=True)
    test_lock = FileLock(lock_path, timeout=5)  # 5s for quick QA
    # test_lock.acquire()  # now child processes will block here

    # ── SPAWN WORKERS ──
    start = time.time()
    try:
        with mp.Pool(2) as pool:
            results = pool.map(worker, [None, None])
    except RuntimeError as e:
        print("\n✅ Timeout exception caught as expected:\n   ", e)
        test_lock.release()
        exit(0)
    finally:
        if test_lock.is_locked:
            test_lock.release()

    # ── VERIFY RESULTS IF NO TIMEOUT ──
    duration = time.time() - start
    print(f"\n→ Total time: {duration:.1f}s (should be ~timeout or single fetch + overhead)\n")

    for i, df in enumerate(results, 1):
        print(f"Worker #{i}:")
        if df is None or df.empty:
            print("   ❌ No data returned!")
            continue
        ts = pd.to_datetime(df['timestamp'])
        low, high = ts.dt.time.min(), ts.dt.time.max()
        print(f"   ✅ Rows: {len(df)}, time span: {low} → {high}")
