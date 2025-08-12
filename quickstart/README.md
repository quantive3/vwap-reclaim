# Quickstart

Run `strategy/` with bundled synthetic data. No API key or network calls required.

## What It Does

1. Copies `quickstart/synthetic_data/` into `polygon_cache/`

2. Sets dummy credentials to avoid prompts

3. Builds a local `params` dict pinned to Jan 3–5, 2023

4. Temporarily injects those params and runs `strategy.main`

### Parameters Used

| Name | Value |
|---|---|
| Dates | 2023-01-03 to 2023-01-05 |
| Entry Window | 09:30–15:45 |
| Stretch / Reclaim | 0.003 / 0.0021 |
| Cooldown | 120s |
| Exits | TP 80%, SL -25%, Max 600s, EOD 15:54, Emergency 15:55 |
| Option Selection | `0DTE`, `itm`, `strikes_depth=1` |
| Frictions | Slippage $0.02, Latency 1s, Fees $0.65 + $0.65 one-way |

For full strategy details, see [strategy/README.md](../strategy/).

## Using Docker

You do not need this repository locally. Pull the prebuilt image and run it.

### Pull

```powershell
docker pull quantive/vwap_reclaim:latest
```

### Run

```powershell
docker run --rm -it quantive/vwap_reclaim:latest python quickstart/run_strategy.py
```


## Using Windows

### Setup

From the project root:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### Run

```powershell
python quickstart/run_strategy.py
```

## Using Mac

### Setup

From the project root:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Run

```bash
python quickstart/run_strategy.py
```

