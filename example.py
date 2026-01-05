#!/usr/bin/env python3
"""
Complete example demonstrating the RINEX Spoofing Detection workflow with highrate data.

This script shows how to:
1. Combine highrate RINEX navigation files into a single JSON dataset
2. Visualize navigation parameters over time
3. Run spoofing detection analysis

Requirements:
- Python 3.12+
- RINEX navigation files (.25n format) in highrate_data/ directory
- GeoRinex library (pip install georinex)
- Matplotlib and NumPy for visualization (pip install matplotlib numpy)
"""

from pathlib import Path
import subprocess
import sys
from typing import List


def run_command(command: List[str], description: str) -> None:
    """Run a command and print status."""
    print(f"\n=== {description} ===")
    print(f"Running: {' '.join(command)}")

    try:
        result = subprocess.run(command, capture_output=True, text=True, check=True)
        print("Success!")
        if result.stdout:
            print(result.stdout)
    except subprocess.CalledProcessError as e:
        print(f"Failed with exit code {e.returncode}")
        if e.stdout:
            print("STDOUT:", e.stdout)
        if e.stderr:
            print("STDERR:", e.stderr)
        raise


def example_combine_navigation_files():
    """Example 1: Combine highrate RINEX navigation files."""
    print("Example 1: Combining Highrate RINEX Navigation Files")
    print("=" * 50)

    # Focus on highrate data as requested
    highrate_dir = Path("highrate_data")

    input_files = []
    if highrate_dir.exists():
        input_files = list(highrate_dir.glob("*.25n"))
    else:
        print("highrate_data/ directory not found.")
        return False

    if not input_files:
        print("No RINEX files found in highrate_data/ directory.")
        print("Using existing combined highrate file instead...")
        combined_file = "combined_highrate.json"
        if Path(combined_file).exists():
            print(f"Using existing combined highrate file: {combined_file}")
            return combined_file
        return False

    print(f"Found {len(input_files)} highrate RINEX files:")
    for f in input_files[:5]:  # Show first 5
        print(f"  - {f}")
    if len(input_files) > 5:
        print(f"  ... and {len(input_files) - 5} more")

    # Combine the highrate files
    # Use --per-source to keep each highrate file as its own entry
    # This uses header time + quarter-hour offset instead of broadcast time
    output_file = "combined_highrate.json"
    cmd = [
        sys.executable, "combine_nav.py",
        *[str(f) for f in input_files],
        "-o", output_file,
        "--per-source",
        "--pretty"
    ]

    try:
        run_command(cmd, "Combining Navigation Files")
        return output_file
    except subprocess.CalledProcessError:
        print("Failed to combine files. Skipping to next example.")
        return None


def example_visualize_data(json_file: str):
    """Example 2: Visualize navigation parameters."""
    print("\nExample 2: Visualizing Navigation Data")
    print("=" * 50)

    if not Path(json_file).exists():
        print(f"JSON file {json_file} not found. Skipping visualization example.")
        return False

    # Visualize SVclockBias for GPS satellites (G constellation)
    cmd = [
        sys.executable, "visualize_nav.py",
        json_file,
        "--metric", "IODE",
        "--constellation", "G",
        "--top", "5",
        "--output", "navigation_plot.png"
    ]

    try:
        run_command(cmd, "Creating Navigation Visualization")
        print("Plot saved as 'navigation_plot.png'")
        return True
    except subprocess.CalledProcessError:
        print("Failed to create visualization. Skipping to next example.")
        return False


def example_spoofing_detection(json_file: str):
    """Example 3: Run spoofing detection analysis."""
    print("\nExample 3: Running Spoofing Detection")
    print("=" * 50)

    if not Path(json_file).exists():
        print(f"JSON file {json_file} not found. Skipping spoofing detection example.")
        return False

    # Run basic spoofing detection
    cmd = [
        sys.executable, "spoof_detection.py",
        json_file,
        "--tolerance", "0.0",
        "--max-interval-hours", "2.0",
        "--output", "spoofing_findings.json"
    ]

    try:
        run_command(cmd, "Running Spoofing Detection Analysis")

        # Check if findings were found
        findings_file = Path("spoofing_findings.json")
        if findings_file.exists():
            print("Findings saved as 'spoofing_findings.json'")
            # Show a summary of findings
            try:
                import json
                with open(findings_file) as f:
                    findings = json.load(f)
                print(f"Analysis complete. Found {len(findings)} potential spoofing indicators.")
                if findings:
                    print("Sample finding:")
                    finding = findings[0]
                    print(f"  Satellite: {finding['satellite']}")
                    print(f"  Code: {finding['code']}")
                    print(f"  Description: {finding['description']}")
            except Exception as e:
                print(f"Could not read findings file: {e}")
        else:
            print("No findings file created (no spoofing indicators detected)")

        return True
    except subprocess.CalledProcessError:
        print("Failed to run spoofing detection. Example complete.")
        return False


def example_advanced_analysis(json_file: str):
    """Example 4: Advanced analysis options."""
    print("\nExample 4: Advanced Analysis Options")
    print("=" * 50)

    if not Path(json_file).exists():
        print(f"JSON file {json_file} not found. Skipping advanced example.")
        return False

    print("Advanced Spoofing Detection Options:")
    print("- Use --tolerance to allow small parameter changes")
    print("- Use --ignore-satellites to skip known problematic PRNs")
    print("- Use --max-interval-hours to adjust staleness detection")

    # Example with tolerance
    cmd = [
        sys.executable, "spoof_detection.py",
        json_file,
        "--tolerance", "1e-10",  # Very small tolerance
        "--max-interval-hours", "1.0",  # Shorter interval
        "--ignore-satellites", "G01", "G05"  # Skip some satellites
    ]

    try:
        run_command(cmd, "Advanced Spoofing Detection with Custom Parameters")
        return True
    except subprocess.CalledProcessError:
        print("Advanced analysis example failed, but basic functionality is working.")
        return False


def main():
    """Run all examples in sequence."""
    print("RINEX Spoofing Detection - Complete Workflow Example")
    print("=" * 60)

    # Check if we're in the right directory
    required_files = ["visualize_nav.py", "spoof_detection.py"]
    missing_files = [f for f in required_files if not Path(f).exists()]
    if missing_files:
        print("Error: Please run this script from the Rinex Spoofing Detection directory.")
        print(f"Missing required files: {', '.join(missing_files)}")
        sys.exit(1)
    
    # Note: combine_nav.py is optional - if it exists, we'll use it, otherwise skip that step
    if not Path("combine_nav.py").exists():
        print("Note: combine_nav.py not found. Will skip file combination step.")
        print("If you have a combined JSON file, the script will use it instead.\n")

    # Example 1: Combine navigation files
    combined_file = example_combine_navigation_files()

    # If combining failed, try with existing combined highrate file
    if not combined_file:
        combined_file = "combined_highrate.json"
        if Path(combined_file).exists():
            print(f"Using existing combined highrate file: {combined_file}")
        else:
            print("No combined highrate file found. Examples will be limited.")
            combined_file = None

    # Example 2: Visualize data
    if combined_file:
        example_visualize_data(combined_file)

        # Example 3: Basic spoofing detection
        example_spoofing_detection(combined_file)

        # Example 4: Advanced analysis
        example_advanced_analysis(combined_file)

    print("\n" + "=" * 60)
    print("Example Complete!")
    print("\nNext Steps:")
    print("1. Examine the generated files: combined_highrate.json, navigation_plot.png, spoofing_findings.json")
    print("2. Modify the parameters in this script to analyze your specific highrate data")
    print("3. Integrate these tools into your GNSS monitoring workflow")
    print("\nFor more information, see the README.md file.")


if __name__ == "__main__":
    main()
