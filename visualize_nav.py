"""Quick visualizer for RINEX navigation JSON exports.

This script expects the JSON structure produced by `rinex_to_json.py`. It
creates line plots for a chosen navigation metric (e.g. `SVclockBias`) across
time, either for specific satellites or the most active satellites within a
constellation.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

try:
    import matplotlib.pyplot as plt  # type: ignore[import]
    import matplotlib.dates as mdates  # type: ignore[import]
    import numpy as np  # type: ignore[import]
except ModuleNotFoundError as exc:  # pragma: no cover - runtime guard
    raise SystemExit(
        "matplotlib and numpy are required for visualize_nav.py. Install them with `pip install matplotlib numpy`."
    ) from exc


def parse_top_argument(value: str) -> int | str:
    """Parse --top argument: accepts integer or 'all'."""
    if value.lower() == 'all':
        return 'all'
    try:
        return int(value)
    except ValueError:
        raise argparse.ArgumentTypeError(f"--top must be an integer or 'all', got: {value}")


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualise GNSS navigation parameters over time.",
    )
    parser.add_argument(
        "json_path",
        type=Path,
        help="Path to the navigation JSON produced by rinex_to_json.py",
    )
    parser.add_argument(
        "--metric",
        default="SVclockBias",
        help="Navigation field to plot (default: SVclockBias).",
    )
    parser.add_argument(
        "--constellation",
        help="Optional constellation prefix to filter satellites (e.g. G, R, E, C).",
    )
    parser.add_argument(
        "--satellites",
        nargs="+",
        help="Explicit list of satellite identifiers to plot (e.g. G01 G05).",
    )
    parser.add_argument(
        "--top",
        type=parse_top_argument,
        default=5,
        help="When satellites are not provided, plot the top-N satellites with the most data points, or 'all' to plot all satellites (default: 5).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional path to save the figure. When omitted the plot window is shown interactively.",
    )
    parser.add_argument(
        "--source",
        help="When data has a source dimension, filter to this source (e.g. bamf314x00).",
    )
    return parser.parse_args()


def load_nav_json(path: Path) -> Dict:
    if not path.exists():
        raise FileNotFoundError(f"JSON file does not exist: {path}")
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def parse_epoch(value: str) -> datetime:
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        # GeoRinex occasionally appends a trailing "Z" or lacks zero padding.
        return datetime.fromisoformat(value.replace("Z", ""))


def collect_series(
    by_time: Dict[str, Dict],
    metric: str,
    satellites_filter: Iterable[str] | None,
    constellation_filter: str | None,
    source_filter: str | None = None,
) -> Dict[str, List[Tuple[datetime, float]]]:
    satellites_filter_set = {s.upper() for s in satellites_filter} if satellites_filter else None
    constellation_filter = constellation_filter.upper() if constellation_filter else None
    source_filter = source_filter.lower() if source_filter else None

    series: Dict[str, List[Tuple[datetime, float]]] = defaultdict(list)
    for epoch_str, entry in sorted(by_time.items()):
        timestamp = parse_epoch(epoch_str)
        satellites = entry.get("satellites", {})
        for sat, payload in satellites.items():
            sat_upper = sat.upper()
            if satellites_filter_set and sat_upper not in satellites_filter_set:
                continue
            if constellation_filter and not sat_upper.startswith(constellation_filter):
                continue

            # Handle multi-dimensional data (e.g., with source dimension)
            measurements = payload.get("measurements", [])
            if measurements:
                for measurement in measurements:
                    indices = measurement.get("indices", {})
                    if source_filter and indices.get("source", "").lower() != source_filter:
                        continue
                    values = measurement.get("values", {})
                    value = values.get(metric)
                    if value is None:
                        continue
                    try:
                        numeric_value = float(value)
                        series[sat_upper].append((timestamp, numeric_value))
                    except (TypeError, ValueError):
                        continue
            else:
                # Standard 2D data (time, sv)
                values = payload.get("values") or {}
                value = values.get(metric)
                if value is None:
                    continue
                try:
                    numeric_value = float(value)
                    series[sat_upper].append((timestamp, numeric_value))
                except (TypeError, ValueError):
                    continue

    return series


def pick_top_satellites(
    series: Dict[str, List[Tuple[datetime, float]]],
    top_n: int | str,
) -> Dict[str, List[Tuple[datetime, float]]]:
    if top_n == 'all':
        return series
    if top_n <= 0:
        return series

    ranked = sorted(series.items(), key=lambda item: len(item[1]), reverse=True)
    selected = dict(ranked[:top_n])
    return selected


def plot_series(
    series: Dict[str, List[Tuple[datetime, float]]],
    metric: str,
    output: Path | None,
) -> None:
    if not series:
        raise ValueError("No data available to plot for the chosen filters/metric.")

    plt.figure(figsize=(12, 6))
    for sat, samples in sorted(series.items()):
        samples = sorted(samples, key=lambda item: item[0])
        times = [item[0] for item in samples]
        values = [item[1] for item in samples]
        
        # Detect gaps and insert NaN to break lines
        if len(times) > 1:
            # Calculate median time interval to determine gap threshold
            intervals = [
                (times[i+1] - times[i]).total_seconds()
                for i in range(len(times) - 1)
            ]
            if intervals:
                median_interval = sorted(intervals)[len(intervals) // 2]
                # Gap threshold: 3x the median interval or 1 hour, whichever is larger
                gap_threshold = max(median_interval * 3, 3600)  # 3600 seconds = 1 hour
                
                # Convert datetimes to numeric values for gap handling
                times_numeric = mdates.date2num(times)
                
                # Insert NaN values where gaps are detected
                times_with_gaps = []
                values_with_gaps = []
                for i in range(len(times_numeric)):
                    if i > 0:
                        gap = (times[i] - times[i-1]).total_seconds()
                        if gap > gap_threshold:
                            # Insert NaN to break the line
                            times_with_gaps.append(np.nan)
                            values_with_gaps.append(np.nan)
                    times_with_gaps.append(times_numeric[i])
                    values_with_gaps.append(values[i])
                
                times = times_with_gaps
                values = values_with_gaps
            else:
                # Convert to numeric even if no gaps detected
                times = mdates.date2num(times)
        else:
            # Convert to numeric for single point
            times = mdates.date2num(times)
        
        plt.plot(times, values, marker="o", markersize=3, label=sat)

    # Format x-axis as dates (set once after all plots)
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H:%M'))
    plt.gca().xaxis.set_major_locator(mdates.HourLocator(interval=1))
    plt.xticks(rotation=45)

    plt.title(f"{metric} over time")
    plt.xlabel("Epoch")
    plt.ylabel(metric)
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.legend(title="Satellite")
    plt.tight_layout()

    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output, dpi=150)
        print(f"Saved figure to {output}")
    else:
        plt.show()


def main() -> int:
    args = parse_arguments()
    data = load_nav_json(args.json_path)

    indexed = data.get("indexed_records")
    if not indexed or "by_time" not in indexed:
        raise ValueError(
            "The provided JSON does not contain the `indexed_records.by_time` structure. "
            "Make sure it was generated with the updated rinex_to_json.py script."
        )

    series = collect_series(
        indexed["by_time"],
        metric=args.metric,
        satellites_filter=args.satellites,
        constellation_filter=args.constellation,
        source_filter=args.source,
    )

    if not args.satellites:
        series = pick_top_satellites(series, args.top)

    plot_series(series, args.metric, args.output)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


