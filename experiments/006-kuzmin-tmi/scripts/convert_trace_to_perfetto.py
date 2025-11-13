#!/usr/bin/env python
"""
Convert PyTorch Profiler JSON trace to Perfetto protobuf format.

Usage:
    python convert_trace_to_perfetto.py <input.pt.trace.json> <output.pftrace>

Note: This is a simplified converter. For full Perfetto features, consider using
      trace_processor to convert the JSON trace.
"""
import json
import sys
from pathlib import Path


def convert_to_perfetto(input_file, output_file):
    """
    Convert Chrome trace JSON to Perfetto-compatible format.

    Note: This creates a JSON format that Perfetto can better understand.
    For true protobuf format, you'd need to use Perfetto's trace_processor.
    """
    print(f"Loading trace: {input_file}")
    with open(input_file, 'r') as f:
        data = json.load(f)

    # Perfetto can load Chrome trace JSON directly
    # But we can optimize it for better display

    events = data.get('traceEvents', [])
    print(f"Total events: {len(events):,}")

    # Filter out very short events (< 1us) to reduce file size
    MIN_DURATION_US = 1
    filtered_events = [e for e in events if e.get('dur', 0) >= MIN_DURATION_US or e.get('ph') != 'X']

    print(f"Filtered events: {len(filtered_events):,} (removed {len(events) - len(filtered_events):,} very short events)")

    # Create optimized trace
    optimized_data = {
        'traceEvents': filtered_events,
        'displayTimeUnit': 'ms',
        'systemTraceEvents': data.get('systemTraceEvents', {}),
        'otherData': {
            'version': data.get('schemaVersion', 1),
        }
    }

    # Add device properties if present
    if 'deviceProperties' in data:
        optimized_data['deviceProperties'] = data['deviceProperties']

    print(f"Writing optimized trace: {output_file}")
    with open(output_file, 'w') as f:
        json.dump(optimized_data, f)

    print(f"✅ Conversion complete!")
    print(f"   Original size: {Path(input_file).stat().st_size / 1024 / 1024:.1f} MB")
    print(f"   Optimized size: {Path(output_file).stat().st_size / 1024 / 1024:.1f} MB")
    print()
    print(f"Load in Perfetto: https://ui.perfetto.dev/")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(__doc__)
        sys.exit(1)

    input_file = Path(sys.argv[1])
    output_file = Path(sys.argv[2])

    if not input_file.exists():
        print(f"Error: Input file not found: {input_file}")
        sys.exit(1)

    convert_to_perfetto(input_file, output_file)
