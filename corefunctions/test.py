import matplotlib.pyplot as plt
import pandas as pd
import re
from datetime import datetime
import numpy as np

def analyze_log_file(log_file_path="log.txt"):
    # Read the log file
    with open(log_file_path, 'r') as file:
        log_data = file.readlines()
    
    # Parse log entries
    timestamps = []
    event_names = []
    durations = []
    
    # Regular expression to parse log entries
    pattern = r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}) - Event: (.+), Duration: (\d+\.\d+)s'
    
    for line in log_data:
        match = re.match(pattern, line.strip())
        if match:
            timestamp_str, event_name, duration_str = match.groups()
            timestamps.append(datetime.strptime(timestamp_str, '%Y-%m-%d %H:%M:%S'))
            event_names.append(event_name)
            durations.append(float(duration_str))
    
    # Create DataFrame
    data = {
        'timestamp': timestamps,
        'event_name': event_names,
        'duration': durations
    }
    df = pd.DataFrame(data)
    
    # Group by event name and calculate statistics
    event_stats = df.groupby('event_name')['duration'].agg(['mean', 'median', 'min', 'max', 'count'])
    event_stats = event_stats.sort_values('median', ascending=False)
    
    # Create bar plot
    plt.figure(figsize=(12, 8))
    bars = plt.bar(event_stats.index, event_stats['median'], color='skyblue')
    
    # Add error bars for min and max
    plt.errorbar(
        x=np.arange(len(event_stats.index)),
        y=event_stats['median'],
        yerr=[event_stats['median'] - event_stats['min'], event_stats['max'] - event_stats['median']],
        fmt='none',
        ecolor='black',
        capsize=5
    )
    
    # Add count annotations
    for i, bar in enumerate(bars):
        count = event_stats['count'].iloc[i]
        plt.text(
            bar.get_x() + bar.get_width()/2,
            bar.get_height() + 0.0001,
            f'n={count}',
            ha='center',
            va='bottom',
            fontsize=8
        )
    
    # Add plot details
    plt.title('Event Execution Time by Function Name')
    plt.xlabel('Event Name')
    plt.ylabel('Median Duration (seconds)')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Save the plot
    plt.savefig('event_durations.png', dpi=300)
    
    # Show the plot
    plt.show()
    
    # Print statistics
    print("Event execution statistics:")
    print(event_stats)
    
    return df, event_stats

if __name__ == "__main__":
    df, stats = analyze_log_file()
    print(f"Analyzed {len(df)} log entries across {len(stats)} unique events")