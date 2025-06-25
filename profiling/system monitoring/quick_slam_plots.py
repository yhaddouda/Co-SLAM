# quick_slam_plots.py - Simple plotting for rapid analysis
import pandas as pd
import matplotlib.pyplot as plt
import json
import argparse

def quick_performance_plot(data_file, metrics=['cpu', 'memory'], output_file=None):
    """
    Create a simple, quick plot for immediate analysis.
    
    This function is designed for rapid iteration when you want to
    quickly examine specific aspects of the performance data without
    the overhead of comprehensive analysis.
    """
    print(f"Creating quick plot for: {', '.join(metrics)}")
    
    # Load data
    if data_file.endswith('.json'):
        with open(data_file, 'r') as f:
            json_data = json.load(f)
        if 'performance_data' in json_data:
            data = pd.DataFrame(json_data['performance_data'])
        else:
            data = pd.DataFrame(json_data)
    else:
        data = pd.read_csv(data_file)
    
    # Filter to active SLAM periods
    slam_data = data[data['slam_process_count'] > 0]
    
    if len(slam_data) == 0:
        print("Warning: No SLAM processes detected in data")
        slam_data = data
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(12, 6))
    
    time_minutes = slam_data['relative_time_seconds'] / 60
    
    if 'cpu' in metrics:
        ax.plot(time_minutes, slam_data['slam_cpu_percent'], 
               label='SLAM CPU (%)', linewidth=2, color='blue')
    
    if 'memory' in metrics:
        ax2 = ax.twinx() if 'cpu' in metrics else ax
        ax2.plot(time_minutes, slam_data['slam_memory_mb'], 
                label='SLAM Memory (MB)', linewidth=2, color='red')
        ax2.set_ylabel('Memory (MB)', color='red')
        ax2.tick_params(axis='y', labelcolor='red')
    
    if 'gpu' in metrics and slam_data['gpu_utilization_percent'].max() > 0:
        ax3 = ax.twinx()
        ax3.spines['right'].set_position(('outward', 60))
        ax3.plot(time_minutes, slam_data['gpu_utilization_percent'], 
                label='GPU (%)', linewidth=2, color='orange')
        ax3.set_ylabel('GPU (%)', color='orange')
        ax3.tick_params(axis='y', labelcolor='orange')
    
    ax.set_xlabel('Time (minutes)')
    ax.set_ylabel('CPU (%)', color='blue' if 'cpu' in metrics else 'black')
    ax.set_title(f'Co-SLAM Performance: {", ".join(metrics).title()}')
    ax.grid(True, alpha=0.3)
    
    # Add legend
    lines1, labels1 = ax.get_legend_handles_labels()
    if 'memory' in metrics and 'cpu' in metrics:
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
    else:
        ax.legend()
    
    plt.tight_layout()
    
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"✓ Saved plot: {output_file}")
    
    plt.show()
    return fig

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Quick SLAM performance plots')
    parser.add_argument('data_file', help='Input data file (JSON or CSV)')
    parser.add_argument('--metrics', nargs='+', choices=['cpu', 'memory', 'gpu'], 
                       default=['cpu', 'memory'], help='Metrics to plot')
    parser.add_argument('--output', help='Output file path')
    
    args = parser.parse_args()
    quick_performance_plot(args.data_file, args.metrics, args.output)