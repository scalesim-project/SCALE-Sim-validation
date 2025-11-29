#!/usr/bin/env python3
"""
Script to analyze reduce-window kernel statistics from filtered_events.csv
Extracts kernel dimensions and computes statistics for reduce-window events.
"""

import pandas as pd
import numpy as np
import re
import sys
from pathlib import Path

def parse_kernel_dimensions(kernel_name):
    """
    Extract dimensions from kernel name like 'layer_norm_(2,3,128,128)'
    For pooling operations, this typically represents (batch, channels, height, width)
    Returns: (batch_size, channels, height, width)
    """
    match = re.search(r'layer_norm_\((\d+),(\d+),(\d+),(\d+)\)', kernel_name)
    if match:
        return int(match.group(1)), int(match.group(2)), int(match.group(3)), int(match.group(4))
    else:
        # Return None values if pattern doesn't match
        return None, None, None, None

def analyze_kernel_stats(input_file='filtered_events.csv', output_file='kernel_analysis.csv'):
    """
    Analyze reduce-window kernel statistics from the filtered events CSV file.
    """
    try:
        # Read the CSV file
        print(f"Reading {input_file}...")
        df = pd.read_csv(input_file)
        
        # Filter for reduce-window events only
        reduce_window_events = df[df['event_name'].str.contains('reduce-window', na=False)].copy()
        
        if reduce_window_events.empty:
            print("No reduce-window events found in the data!")
            return
        
        print(f"Found {len(reduce_window_events)} reduce-window events")
        
        # Extract dimensions from kernel names
        dimensions = reduce_window_events['kernel_name'].apply(parse_kernel_dimensions)
        reduce_window_events['batch_size'] = [d[0] for d in dimensions]
        reduce_window_events['channels'] = [d[1] for d in dimensions]
        reduce_window_events['height'] = [d[2] for d in dimensions]
        reduce_window_events['width'] = [d[3] for d in dimensions]
        
        # Group by kernel configuration and compute statistics
        grouped = reduce_window_events.groupby(['kernel_name', 'batch_size', 'channels', 'height', 'width'])
        
        stats_list = []
        
        for (kernel_name, batch_size, channels, height, width), group in grouped:
            durations = group['dur(us)']
            
            stats = {
                'kernel_name': kernel_name,
                'batch_size': batch_size,
                'channels': channels,
                'height': height,
                'width': width,
                'total_elements': batch_size * channels * height * width,
                'avg_duration_us': durations.mean(),
                'min_duration_us': durations.min(),
                'max_duration_us': durations.max(),
                'stddev_duration_us': durations.std(),
                'num_measurements': len(durations)
            }
            stats_list.append(stats)
        
        # Create output DataFrame
        output_df = pd.DataFrame(stats_list)
        
        # Sort by dimensions for better readability
        output_df = output_df.sort_values(['batch_size', 'channels', 'height', 'width'])
        
        # Save to CSV
        output_df.to_csv(output_file, index=False)
        print(f"Analysis complete! Results saved to {output_file}")
        
        # Print summary statistics
        print(f"\nSummary:")
        print(f"- Total unique kernel configurations: {len(output_df)}")
        print(f"- Batch sizes: {sorted(output_df['batch_size'].unique())}")
        print(f"- Channels: {sorted(output_df['channels'].unique())}")
        print(f"- Heights: {sorted(output_df['height'].unique())}")
        print(f"- Widths: {sorted(output_df['width'].unique())}")
        
        # Show first few rows
        print(f"\nFirst 5 results:")
        print(output_df.head().to_string(index=False))
        
        return output_df
        
    except Exception as e:
        print(f"Error processing data: {e}")
        sys.exit(1)

if __name__ == "__main__":
    # Set up paths
    script_dir = Path(__file__).parent
    input_file = script_dir / "filtered_events.csv"
    output_file = script_dir / "kernel_analysis.csv"
    
    if not input_file.exists():
        print(f"Error: {input_file} not found!")
        sys.exit(1)
    
    # Run analysis
    analyze_kernel_stats(str(input_file), str(output_file))
