# RINEX to JSON Conversion

This repository includes utilities for converting RINEX observation (`*.o`) and navigation (`*.n`) files into structured JSON. The conversion relies on [GeoRinex](https://github.com/geospace-code/georinex) to parse the RINEX contents and then serialises the resulting datasets into plain Python objects that capture both metadata and time-indexed measurements.

## Requirements

- Python 3.12+
- GeoRinex and its dependencies (installed automatically when running `pip install georinex`)

## Usage

```bash
python rinex_to_json.py <path-to-rinex-file> --pretty
```

This prints the JSON representation to `stdout`. To convert multiple files at once and write the result to a file:

```bash
python rinex_to_json.py brdc0010.25n algo0010.25o --pretty -o outputs.json
```

The output JSON contains:

- `source`: absolute path to the original RINEX file.
- `rinex_type`: either `obs` (observation) or `nav` (navigation).
- `dimensions`: dimension names and lengths.
- `coordinates`: coordinate arrays with metadata.
- `data_variables`: measurement arrays including dimensions, attributes, and data (with `NaN` converted to `null`).
- `attributes`: header metadata extracted from the RINEX file.
- `indexed_records`: structure optimised for temporal queries. It contains:
  - `dimensions`: ordered list of the dataset dimensions and the specific names used for time and satellite axes.
  - `flat`: a list where each element stores the dimension values plus the decoded measurements for that coordinate.
  - `by_time`: dictionary keyed by epoch (ISO string) exposing all satellites seen during that epoch and, under each epoch, a `constellations` breakdown that groups satellites by their constellation prefix (e.g. `G`, `R`, `E`, `C`).
  - `by_constellation`: dictionary keyed by constellation → epoch → satellites, mirroring the per-epoch view but starting from the constellation perspective.

Example conversion results for the sample files in this repository are stored in `/tmp/brdc0010.json` and `/tmp/algo0010.json` after running the above commands locally.

## Spoofing Detection

```bash
python spoof_detection.py brdc0010.json --output findings.json
```

The detector analyses the `indexed_records` section and raises findings when:

- Broadcast parameters change without a matching `IODE`/`IODC` update.
- Navigation data persists longer than the configured interval (default 2 hours).
- `IODE`/`IODC` values regress while parameters change (suggesting replay or malformed uploads).

Use `--tolerance` to relax numeric comparisons, `--max-interval-hours` to alter the staleness window, and `--ignore-satellites` to skip known-problematic PRNs. When `--output` is provided, the findings are saved as JSON; otherwise a concise summary is printed to the terminal.

## Combining Navigation Files

```bash
python combine_nav.py data/ -o combined_nav.json --pretty
```

The combiner ingests every RINEX navigation file found in the supplied paths, keeps the most recent message when duplicate epochs are present, and writes a single JSON in the same format as the per-file converter. Pass `--per-source` to retain each file on its own `source` dimension—useful for high-rate captures where multiple uploads share the same epoch. Source file paths are stored under `attributes.sources` in the output.

