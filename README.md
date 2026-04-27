# RINEX Spoofing Detection Tool

This tool helps you analyze GNSS RINEX navigation data for potential spoofing indicators.

It provides a simple workflow:

1. Convert RINEX files to JSON.
2. Run spoofing detection on the JSON.
3. Visualize key navigation metrics over time.

## Setup

Use Python 3.12+.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip georinex matplotlib numpy pytest
```

## How to Run

### 1) Convert RINEX to JSON

```bash
python rinex_to_json.py <path-to-nav-file> --pretty -o nav.json
```

Example:

```bash
python rinex_to_json.py brdc0010.25n --pretty -o brdc0010.json
```

### 2) Run spoofing detection

```bash
python spoof_detection.py nav.json --output findings.json
```

Optional high-rate checks:

```bash
python spoof_detection.py nav.json \
  --enable-cross-satellite-checks \
  --min-cross-satellite-correlation 0.9 \
  --max-ephemeris-age-hours 4 \
  --replay-sequence-length 4 \
  --output findings.json
```

### 3) Visualize navigation metrics

```bash
python visualize_nav.py nav.json \
  --metric SVclockBias \
  --constellation G \
  --top 5 \
  --output plots/svclockbias_gps.png
```

The generated plot helps inspect suspicious behavior over time.

## One-Command Sample Run (using repository sample data)

```bash
python spoof_detection.py combined_highrate.json --enable-cross-satellite-checks --output findings_run.json
python visualize_nav.py combined_highrate.json --metric SVclockBias --constellation G --top 5 --output plots/svclockbias_gps.png
```

## Run Tests

```bash
pytest tests/
```
