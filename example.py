#!/usr/bin/env python3
"""Run the full RINEX spoof-detection toolchain and generate output images.

Pipeline:
1. Combine high-rate RINEX NAV files into one JSON dataset.
2. Run spoofing detection and write findings JSON.
3. Generate multiple plot images with `visualize_nav.py`.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Iterable


def run_command(command: list[str], description: str) -> None:
    """Run a subprocess command and raise on failure."""
    print(f"\n=== {description} ===")
    print(f"Running: {' '.join(command)}")
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Command failed with exit code {result.returncode}")
        if result.stdout:
            print("STDOUT:")
            print(result.stdout)
        if result.stderr:
            print("STDERR:")
            print(result.stderr)
        raise RuntimeError(description)
    if result.stdout:
        print(result.stdout)


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments for the full toolchain run."""
    parser = argparse.ArgumentParser(
        description="Run combine -> spoof detect -> image generation pipeline.",
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("highrate_data"),
        help="Directory containing high-rate RINEX nav files (default: highrate_data).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("example_output"),
        help="Directory where JSON and plot outputs are written (default: example_output).",
    )
    parser.add_argument(
        "--constellation",
        default="G",
        help="Constellation filter passed to visualize_nav.py (default: G).",
    )
    parser.add_argument(
        "--top",
        default="5",
        help="Top-N satellites for plots, or 'all' (default: 5).",
    )
    return parser.parse_args(argv)


def _find_rinex_inputs(input_dir: Path) -> list[Path]:
    """Find .25n/.n nav files in the input directory."""
    if not input_dir.exists():
        return []
    files = list(input_dir.glob("*.25n")) + list(input_dir.glob("*.n"))
    return sorted(set(path.resolve() for path in files))


def create_combined_json(input_dir: Path, output_json: Path) -> Path:
    """Build combined JSON from NAV files, or fall back to existing combined JSON."""
    output_json.parent.mkdir(parents=True, exist_ok=True)
    input_files = _find_rinex_inputs(input_dir)

    if input_files and Path("combine_nav.py").exists():
        cmd = [
            sys.executable,
            "combine_nav.py",
            *[str(path) for path in input_files],
            "-o",
            str(output_json),
            "--per-source",
            "--pretty",
        ]
        run_command(cmd, "Combine Navigation Files")
        return output_json

    fallback = Path("combined_highrate.json")
    if fallback.exists():
        shutil.copy2(fallback, output_json)
        print(f"Using fallback combined dataset: {fallback} -> {output_json}")
        return output_json

    raise FileNotFoundError(
        "No RINEX nav files were found and combined_highrate.json fallback is missing."
    )


def run_detection(input_json: Path, findings_json: Path) -> None:
    """Run spoof_detection.py with high-rate checks enabled."""
    cmd = [
        sys.executable,
        "spoof_detection.py",
        str(input_json),
        "--enable-cross-satellite-checks",
        "--min-cross-satellite-correlation",
        "0.9",
        "--max-ephemeris-age-hours",
        "4.0",
        "--replay-sequence-length",
        "4",
        "--output",
        str(findings_json),
    ]
    run_command(cmd, "Run Spoof Detection")


def generate_plots(input_json: Path, plots_dir: Path, constellation: str, top: str) -> list[Path]:
    """Generate a set of navigation metric plots."""
    plots_dir.mkdir(parents=True, exist_ok=True)
    plot_specs = [
        ("SVclockBias", f"svclockbias_{constellation.lower()}.png"),
        ("IODE", f"iode_{constellation.lower()}.png"),
        ("TransTime", f"transtime_{constellation.lower()}.png"),
        ("Toe", f"toe_{constellation.lower()}.png"),
    ]

    created: list[Path] = []
    for metric, filename in plot_specs:
        output_path = plots_dir / filename
        cmd = [
            sys.executable,
            "visualize_nav.py",
            str(input_json),
            "--metric",
            metric,
            "--constellation",
            constellation,
            "--top",
            top,
            "--output",
            str(output_path),
        ]
        run_command(cmd, f"Generate Plot: {metric}")
        created.append(output_path)
    return created


def print_findings_summary(findings_json: Path) -> None:
    """Print a compact summary of findings JSON."""
    if not findings_json.exists():
        print(f"Findings file not found: {findings_json}")
        return
    try:
        findings = json.loads(findings_json.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        print(f"Could not parse findings JSON: {exc}")
        return
    print(f"Spoof detection complete: {len(findings)} finding(s).")
    if findings:
        sample = findings[0]
        print(
            "Sample: "
            f"sat={sample.get('satellite')} "
            f"code={sample.get('code')} "
            f"desc={sample.get('description')}"
        )


def main(argv: Iterable[str] | None = None) -> int:
    """Execute the full example pipeline."""
    args = parse_args(argv)

    required = ["spoof_detection.py", "visualize_nav.py"]
    missing = [name for name in required if not Path(name).exists()]
    if missing:
        print(
            "Missing required files in current directory: "
            + ", ".join(missing)
        )
        return 1

    output_dir = args.output_dir.resolve()
    plots_dir = output_dir / "plots"
    combined_json = output_dir / "combined_highrate.json"
    findings_json = output_dir / "spoofing_findings.json"

    print("RINEX Spoofing Detection - Full Toolchain Example")
    print("=" * 60)
    print(f"Input directory: {args.input_dir}")
    print(f"Output directory: {output_dir}")

    try:
        combined_path = create_combined_json(args.input_dir, combined_json)
        run_detection(combined_path, findings_json)
        images = generate_plots(combined_path, plots_dir, args.constellation, str(args.top))
    except Exception as exc:
        print(f"\nPipeline failed: {exc}")
        return 1

    print_findings_summary(findings_json)

    print("\nCreated artifacts:")
    print(f"- {combined_path}")
    print(f"- {findings_json}")
    for image in images:
        print(f"- {image}")
    print("\nDone.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
