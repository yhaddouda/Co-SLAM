import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import json
from datetime import datetime
import argparse
import os

class SLAMPerformanceVisualizer:
    def __init__(self, figure_size_scale=1.0):
        self.figure_scale = figure_size_scale
        self.base_figsize = (10, 6)
        
        self.setup_plotting_style()
        self.setup_color_palette()
        self.configure_matplotlib()
    
    def setup_plotting_style(self):
        # matplotlib.pyplot.style.use() - apply plotting style
        available_styles = plt.style.available
        scientific_styles = ['seaborn-v0_8-whitegrid', 'seaborn-whitegrid', 'seaborn-v0_8', 'seaborn', 'ggplot']
        
        for style in scientific_styles:
            if style in available_styles:
                try:
                    plt.style.use(style)
                    break
                except:
                    continue
        
        # seaborn color palette if available
        try:
            import seaborn as sns
            sns.set_palette("husl")
        except ImportError:
            pass
    
    def setup_color_palette(self):
        self.colors = {
            'slam_cpu': "#2d81bd", 'system_cpu': "#7f7f7f",
            'slam_memory': "#f37609", 'system_memory': "#d62728",
            'gpu': '#2ca02c', 'cores': '#9467bd', 'per_core': '#8c564b'
        }
    
    def configure_matplotlib(self):
        # matplotlib.rcParams - global plotting parameters
        params = {
            'font.size': 10, 'axes.titlesize': 12, 'axes.labelsize': 10,
            'xtick.labelsize': 9, 'ytick.labelsize': 9, 'legend.fontsize': 9,
            'figure.titlesize': 14, 'lines.linewidth': 1.5, 'figure.dpi': 100,
            'savefig.dpi': 300, 'savefig.bbox': 'tight', 'axes.grid': True,
            'grid.alpha': 0.3, 'axes.spines.top': False, 'axes.spines.right': False
        }
        
        for param, value in params.items():
            try:
                plt.rcParams[param] = value
            except KeyError:
                continue
    
    def load_performance_data(self, filename):
        file_extension = os.path.splitext(filename)[1].lower()
        
        if file_extension == '.json':
            with open(filename, 'r') as f:
                json_data = json.load(f)
            
            if isinstance(json_data, dict) and 'performance_data' in json_data:
                data = pd.DataFrame(json_data['performance_data'])
                metadata = json_data.get('metadata', {})
            else:
                data = pd.DataFrame(json_data)
                metadata = {}
                
        elif file_extension == '.csv':
            # pandas.read_csv() - load CSV data into DataFrame
            data = pd.read_csv(filename)
            metadata = {}
        else:
            raise ValueError(f"Unsupported file format: {file_extension}")
        
        required_columns = ['relative_time_seconds', 'slam_cpu_percent', 'slam_memory_mb']
        missing_columns = [col for col in required_columns if col not in data.columns]
        
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
        
        # pandas.to_numeric() - convert columns to numeric data types
        numeric_columns = [
            'relative_time_seconds', 'slam_cpu_percent', 'slam_memory_mb', 'slam_memory_percent',
            'cpu_cores_in_use', 'system_cpu_percent', 'system_memory_percent',
            'gpu_utilization_percent', 'gpu_memory_mb', 'cuda_cores'
        ]
        
        for col in numeric_columns:
            if col in data.columns:
                data[col] = pd.to_numeric(data[col], errors='coerce')
        
        return data, metadata
    
    def create_cpu_analysis_plot(self, data, save_path=None):
        slam_data = data[data['slam_process_count'] > 0].copy()
        if len(slam_data) == 0:
            slam_data = data
        
        # matplotlib.pyplot.subplots() - create figure with subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(self.figure_scale * self.base_figsize[0], 
                                                      self.figure_scale * 8), sharex=True)
        
        time_minutes = slam_data['relative_time_seconds'] / 60
        
        # Plot CPU usage over time
        ax1.plot(time_minutes, slam_data['slam_cpu_percent'], 
                color=self.colors['slam_cpu'], linewidth=2, label='SLAM CPU Usage')
        ax1.plot(time_minutes, slam_data['system_cpu_percent'], 
                color=self.colors['system_cpu'], alpha=0.7, label='System CPU Usage')
        
        ax1.set_ylabel('CPU Usage (%)')
        ax1.set_title('CPU Utilization Over Time')
        ax1.set_ylim(0, max(slam_data['slam_cpu_percent'].max() * 1.1, slam_data['system_cpu_percent'].max() * 1.1))
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot cores in use with total cores reference line
        if 'cpu_cores_in_use' in slam_data.columns:
            ax2.plot(time_minutes, slam_data['cpu_cores_in_use'], 
                    color=self.colors['cores'], linewidth=2, label='Cores in Use', marker='o', markersize=3)
            
            total_cores = slam_data['cpu_total_cores'].iloc[0] if 'cpu_total_cores' in slam_data.columns else 0
            if total_cores > 0:
                ax2.axhline(y=total_cores, color='red', linestyle='--', alpha=0.6, 
                           label=f'Max Cores Available ({total_cores})')
                ax2.set_ylim(0, total_cores + 1)
            
            ax2.set_ylabel('Number of Cores')
            ax2.set_title('CPU Cores Utilization Over Time')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
        
        ax2.set_xlabel('Time (minutes)')
        
        plt.suptitle('CPU Performance Analysis', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            # matplotlib.pyplot.savefig() - save figure to file
            plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        
        return fig
    
    def create_memory_analysis_plot(self, data, save_path=None):
        slam_data = data[data['slam_process_count'] > 0].copy()
        if len(slam_data) == 0:
            slam_data = data
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(self.figure_scale * self.base_figsize[0], 
                                                      self.figure_scale * 8), sharex=True)
        
        time_minutes = slam_data['relative_time_seconds'] / 60
        
        # Memory usage in MB with system limit reference
        ax1.plot(time_minutes, slam_data['slam_memory_mb'], 
                color=self.colors['slam_memory'], linewidth=2, label='SLAM Memory Usage')
        
        # Add total system memory reference line - get from metadata or calculate
        if 'system_memory_available_mb' in slam_data.columns:
            # Estimate total memory from available + used
            total_memory_mb = slam_data['system_memory_available_mb'].iloc[0] / (1 - slam_data['system_memory_percent'].iloc[0]/100)
            ax1.axhline(y=total_memory_mb, color='red', linestyle='--', alpha=0.6, 
                       label=f'Total System Memory ({total_memory_mb:.0f} MB)')
            ax1.set_ylim(0, total_memory_mb * 1.1)
        
        ax1.set_ylabel('Memory Usage (MB)')
        ax1.set_title('Memory Usage Over Time (Absolute)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Memory usage as percentage with 100% reference
        if 'slam_memory_percent' in slam_data.columns:
            ax2.plot(time_minutes, slam_data['slam_memory_percent'], 
                    color=self.colors['slam_memory'], linewidth=2, label='SLAM Memory %')
            ax2.plot(time_minutes, slam_data['system_memory_percent'], 
                    color=self.colors['system_memory'], alpha=0.7, label='System Memory %')
            
            # Add 100% memory reference line
            ax2.axhline(y=100, color='red', linestyle='--', alpha=0.6, label='100% Memory Limit')
            ax2.set_ylim(0, 105)
            ax2.set_ylabel('Memory Usage (%)')
            ax2.set_title('Memory Usage Over Time (Percentage)')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
        
        ax2.set_xlabel('Time (minutes)')
        
        plt.suptitle('Memory Performance Analysis', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        
        return fig
    
    def create_gpu_analysis_plot(self, data, save_path=None):
        slam_data = data[data['slam_process_count'] > 0].copy()
        if len(slam_data) == 0:
            slam_data = data
        
        # Check if GPU data is available
        has_gpu_data = ('gpu_utilization_percent' in slam_data.columns and 
                       slam_data['gpu_utilization_percent'].max() > 0)
        
        if not has_gpu_data:
            # Create a placeholder plot indicating no GPU data
            fig, ax = plt.subplots(figsize=(self.figure_scale * self.base_figsize[0], 
                                           self.figure_scale * 4))
            ax.text(0.5, 0.5, 'No GPU Data Available\n\nEither:\n• No GPU detected\n• GPU monitoring failed\n• No GPU usage during monitoring', 
                   transform=ax.transAxes, ha='center', va='center', fontsize=12,
                   bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.5))
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.axis('off')
            ax.set_title('GPU Performance Analysis - No Data', fontsize=14, fontweight='bold')
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
            return fig
        
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(self.figure_scale * self.base_figsize[0], 
                                                           self.figure_scale * 10), sharex=True)
        
        time_minutes = slam_data['relative_time_seconds'] / 60
        
        # GPU utilization with 100% reference line
        ax1.plot(time_minutes, slam_data['gpu_utilization_percent'], 
                color=self.colors['gpu'], linewidth=2, label='GPU Utilization')
        ax1.axhline(y=100, color='red', linestyle='--', alpha=0.6, label='100% GPU Limit')
        ax1.set_ylabel('GPU Utilization (%)')
        ax1.set_title('GPU Utilization Over Time')
        ax1.set_ylim(0, 105)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # GPU memory usage
        if 'gpu_memory_mb' in slam_data.columns:
            ax2.plot(time_minutes, slam_data['gpu_memory_mb'], 
                    color=self.colors['gpu'], linewidth=2, label='GPU Memory Usage')
            
            # Try to estimate max GPU memory (rough estimates for common GPUs)
            max_gpu_memory = 0
            if 'cuda_cores' in slam_data.columns and slam_data['cuda_cores'].iloc[0] > 0:
                cuda_cores = slam_data['cuda_cores'].iloc[0]
                # Rough GPU memory mapping based on CUDA cores
                gpu_memory_map = {
                    16384: 24576,  # RTX 4090: 24GB
                    9728: 16384,   # RTX 4080: 16GB  
                    5888: 12288,   # RTX 4070/3070: 12GB/8GB
                    10496: 24576,  # RTX 3090: 24GB
                    8704: 10240,   # RTX 3080: 10GB
                    2944: 8192,    # RTX 2080: 8GB
                    2560: 8192,    # GTX 1080: 8GB
                    5120: 32768,   # V100: 32GB
                    6912: 40960,   # A100: 40GB
                    14592: 80000   # H100: 80GB
                }
                max_gpu_memory = gpu_memory_map.get(cuda_cores, 0)
            
            if max_gpu_memory > 0:
                ax2.axhline(y=max_gpu_memory, color='red', linestyle='--', alpha=0.6, 
                           label=f'Estimated GPU Memory Limit ({max_gpu_memory/1024:.1f} GB)')
                ax2.set_ylim(0, max_gpu_memory * 1.1)
            
            ax2.set_ylabel('GPU Memory (MB)')
            ax2.set_title('GPU Memory Usage Over Time')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
        
        # GPU temperature or additional metric (if available in future)
        # For now, show CUDA cores info as bar chart
        if 'cuda_cores' in slam_data.columns and slam_data['cuda_cores'].iloc[0] > 0:
            cuda_cores = slam_data['cuda_cores'].iloc[0]
            ax3.barh(['CUDA Cores'], [cuda_cores], color=self.colors['gpu'], alpha=0.7)
            ax3.set_xlabel('Count')
            ax3.set_title(f'GPU Hardware Information (CUDA Cores: {cuda_cores})')
            ax3.grid(True, alpha=0.3)
        else:
            ax3.text(0.5, 0.5, 'GPU Hardware Information\nNot Available', 
                    transform=ax3.transAxes, ha='center', va='center',
                    bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.3))
            ax3.axis('off')
        
        ax2.set_xlabel('Time (minutes)')
        
        # Add CUDA cores info to title if available
        if 'cuda_cores' in slam_data.columns and slam_data['cuda_cores'].iloc[0] > 0:
            cuda_cores = slam_data['cuda_cores'].iloc[0]
            fig.suptitle(f'GPU Performance Analysis (CUDA Cores: {cuda_cores})', 
                        fontsize=14, fontweight='bold')
        else:
            fig.suptitle('GPU Performance Analysis', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        
        return fig
    
    def create_per_core_heatmap(self, data, save_path=None):
        slam_data = data[data['slam_process_count'] > 0].copy()
        if len(slam_data) == 0:
            slam_data = data
        
        # Check if per-core data exists
        per_core_columns = [col for col in slam_data.columns if col.startswith('core_') and col.endswith('_cpu_percent')]
        
        if not per_core_columns:
            print("No per-core CPU data available for heatmap")
            return None
        
        # Extract per-core data - pandas DataFrame selection
        per_core_data = slam_data[per_core_columns]
        time_minutes = slam_data['relative_time_seconds'] / 60
        
        fig, ax = plt.subplots(figsize=(self.figure_scale * 12, self.figure_scale * 6))
        
        # matplotlib.pyplot.imshow() - display data as heatmap
        im = ax.imshow(per_core_data.T, aspect='auto', cmap='viridis', 
                      extent=[time_minutes.min(), time_minutes.max(), 0, len(per_core_columns)])
        
        ax.set_xlabel('Time (minutes)')
        ax.set_ylabel('CPU Core')
        ax.set_title('Per-Core CPU Utilization Heatmap')
        
        # matplotlib.pyplot.colorbar() - add color scale bar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('CPU Usage (%)')
        
        # Set y-ticks to show core numbers
        ax.set_yticks(range(len(per_core_columns)))
        ax.set_yticklabels([f'Core {i}' for i in range(len(per_core_columns))])
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        
        return fig
    
    def create_comprehensive_temporal_plot(self, data, save_path=None):
        """Create a comprehensive temporal view of all metrics with reference lines"""
        slam_data = data[data['slam_process_count'] > 0].copy()
        if len(slam_data) == 0:
            slam_data = data
        
        # Create subplot layout - matplotlib.pyplot.subplots() with shared x-axis
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(self.figure_scale * 15, 
                                                                    self.figure_scale * 10), 
                                                     sharex=True)
        
        time_minutes = slam_data['relative_time_seconds'] / 60
        
        # CPU Usage without limit reference
        ax1.plot(time_minutes, slam_data['slam_cpu_percent'], 
                color=self.colors['slam_cpu'], linewidth=2, label='SLAM CPU')
        ax1.set_ylabel('CPU Usage (%)')
        ax1.set_title('CPU Utilization')
        ax1.set_ylim(0, slam_data['slam_cpu_percent'].max() * 1.1)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Memory Usage with system limit
        ax2.plot(time_minutes, slam_data['slam_memory_percent'] if 'slam_memory_percent' in slam_data.columns else slam_data['slam_memory_mb'], 
                color=self.colors['slam_memory'], linewidth=2, label='SLAM Memory')
        
        if 'slam_memory_percent' in slam_data.columns:
            ax2.axhline(y=100, color='red', linestyle='--', alpha=0.6, label='100% System Memory')
            ax2.set_ylabel('Memory Usage (% of System)')
            ax2.set_ylim(0, 105)
        else:
            ax2.set_ylabel('Memory Usage (MB)')
        
        ax2.set_title('Memory Utilization')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # CPU Cores with total cores limit
        if 'cpu_cores_in_use' in slam_data.columns:
            ax3.plot(time_minutes, slam_data['cpu_cores_in_use'], 
                    color=self.colors['cores'], linewidth=2, marker='o', markersize=3, label='Cores in Use')
            
            total_cores = slam_data['cpu_total_cores'].iloc[0] if 'cpu_total_cores' in slam_data.columns else 0
            if total_cores > 0:
                ax3.axhline(y=total_cores, color='red', linestyle='--', alpha=0.6, 
                           label=f'Max Cores ({total_cores})')
                ax3.set_ylim(0, total_cores + 1)
            
            ax3.set_ylabel('CPU Cores in Use')
            ax3.set_title('CPU Core Utilization')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
        else:
            ax3.text(0.5, 0.5, 'CPU Core Data\nNot Available', transform=ax3.transAxes, 
                    ha='center', va='center')
            ax3.set_title('CPU Core Utilization')
        
        # GPU Utilization with 100% limit
        if ('gpu_utilization_percent' in slam_data.columns and 
            slam_data['gpu_utilization_percent'].max() > 0):
            ax4.plot(time_minutes, slam_data['gpu_utilization_percent'], 
                    color=self.colors['gpu'], linewidth=2, label='GPU Utilization')
            ax4.axhline(y=100, color='red', linestyle='--', alpha=0.6, label='100% GPU Limit')
            ax4.set_ylabel('GPU Utilization (%)')
            ax4.set_ylim(0, 105)
            ax4.legend()
            ax4.grid(True, alpha=0.3)
        else:
            ax4.text(0.5, 0.5, 'GPU Data\nNot Available', transform=ax4.transAxes, 
                    ha='center', va='center')
        
        ax4.set_title('GPU Utilization')
        ax4.set_xlabel('Time (minutes)')
        ax3.set_xlabel('Time (minutes)')
        
        plt.suptitle('Comprehensive System Performance Timeline', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        
        return fig
        slam_data = data[data['slam_process_count'] > 0].copy()
        if len(slam_data) == 0:
            slam_data = data
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(self.figure_scale * 12, 
                                                                    self.figure_scale * 8))
        
        # CPU distribution - matplotlib.pyplot.hist()
        if len(slam_data['slam_cpu_percent'].dropna()) > 0:
            ax1.hist(slam_data['slam_cpu_percent'].dropna(), bins=20, alpha=0.7, 
                    color=self.colors['slam_cpu'], edgecolor='black')
            ax1.set_xlabel('CPU Usage (%)')
            ax1.set_ylabel('Frequency')
            ax1.set_title('CPU Usage Distribution')
            ax1.grid(True, alpha=0.3)
        
        # Memory distribution
        if len(slam_data['slam_memory_mb'].dropna()) > 0:
            ax2.hist(slam_data['slam_memory_mb'].dropna(), bins=20, alpha=0.7, 
                    color=self.colors['slam_memory'], edgecolor='black')
            ax2.set_xlabel('Memory Usage (MB)')
            ax2.set_ylabel('Frequency')
            ax2.set_title('Memory Usage Distribution')
            ax2.grid(True, alpha=0.3)
        
        # Cores usage distribution
        if 'cpu_cores_in_use' in slam_data.columns:
            unique_cores = sorted(slam_data['cpu_cores_in_use'].dropna().unique())
            counts = [len(slam_data[slam_data['cpu_cores_in_use'] == cores]) for cores in unique_cores]
            ax3.bar(unique_cores, counts, color=self.colors['cores'], alpha=0.7)
            ax3.set_xlabel('Number of Cores in Use')
            ax3.set_ylabel('Frequency')
            ax3.set_title('Core Usage Distribution')
            ax3.grid(True, alpha=0.3)
        
        # Performance summary text
        duration_minutes = slam_data['relative_time_seconds'].max() / 60
        avg_cpu = slam_data['slam_cpu_percent'].mean()
        max_cpu = slam_data['slam_cpu_percent'].max()
        avg_memory = slam_data['slam_memory_mb'].mean()
        max_memory = slam_data['slam_memory_mb'].max()
        
        summary_text = f"""Performance Summary

Duration: {duration_minutes:.1f} minutes
Samples: {len(slam_data)}

CPU Usage:
  Average: {avg_cpu:.1f}%
  Peak: {max_cpu:.1f}%

Memory Usage:
  Average: {avg_memory:.1f} MB
  Peak: {max_memory:.1f} MB"""
        
        if 'slam_memory_percent' in slam_data.columns:
            avg_mem_pct = slam_data['slam_memory_percent'].mean()
            max_mem_pct = slam_data['slam_memory_percent'].max()
            summary_text += f"""
  
Memory % of System:
  Average: {avg_mem_pct:.1f}%
  Peak: {max_mem_pct:.1f}%"""
        
        ax4.text(0.1, 0.9, summary_text, transform=ax4.transAxes,
                verticalalignment='top', fontfamily='monospace', fontsize=9,
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))
        ax4.set_xlim(0, 1)
        ax4.set_ylim(0, 1)
        ax4.axis('off')
        
        plt.suptitle('Performance Summary Dashboard', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        
        return fig


def main():
    parser = argparse.ArgumentParser(description='Visualize Co-SLAM performance data')
    parser.add_argument('data_file', help='Input data file (JSON or CSV)')
    parser.add_argument('--output', default='./', help='Output directory')
    parser.add_argument('--cpu', action='store_true', help='Create CPU analysis plot')
    parser.add_argument('--memory', action='store_true', help='Create memory analysis plot')
    parser.add_argument('--gpu', action='store_true', help='Create GPU analysis plot')
    parser.add_argument('--heatmap', action='store_true', help='Create per-core CPU heatmap')
    parser.add_argument('--summary', action='store_true', help='Create performance summary')
    parser.add_argument('--temporal', action='store_true', help='Create comprehensive temporal overview')
    parser.add_argument('--all', action='store_true', help='Create all available plots')
    parser.add_argument('--scale', type=float, default=1.0, help='Figure size scale')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.data_file):
        print(f"Error: Input file '{args.data_file}' not found")
        return 1
    
    os.makedirs(args.output, exist_ok=True)
    
    try:
        visualizer = SLAMPerformanceVisualizer(figure_size_scale=args.scale)
        data, metadata = visualizer.load_performance_data(args.data_file)
        
        base_name = os.path.splitext(os.path.basename(args.data_file))[0]
        
        # Create requested plots
        if args.all or args.cpu:
            cpu_path = os.path.join(args.output, f"{base_name}_cpu_analysis.png")
            visualizer.create_cpu_analysis_plot(data, cpu_path)
            print(f"✓ Created CPU analysis: {cpu_path}")
        
        if args.all or args.memory:
            memory_path = os.path.join(args.output, f"{base_name}_memory_analysis.png")
            visualizer.create_memory_analysis_plot(data, memory_path)
            print(f"✓ Created memory analysis: {memory_path}")
        
        if args.all or args.gpu:
            gpu_path = os.path.join(args.output, f"{base_name}_gpu_analysis.png")
            gpu_fig = visualizer.create_gpu_analysis_plot(data, gpu_path)
            print(f"✓ Created GPU analysis: {gpu_path}")
        
        if args.all or args.heatmap:
            heatmap_path = os.path.join(args.output, f"{base_name}_cpu_heatmap.png")
            heatmap_fig = visualizer.create_per_core_heatmap(data, heatmap_path)
            if heatmap_fig:
                print(f"✓ Created CPU heatmap: {heatmap_path}")
        
        if args.all or args.temporal:
            temporal_path = os.path.join(args.output, f"{base_name}_temporal_overview.png")
            visualizer.create_comprehensive_temporal_plot(data, temporal_path)
            print(f"✓ Created temporal overview: {temporal_path}")
        
        if args.all or args.summary or not any([args.cpu, args.memory, args.gpu, args.heatmap, args.temporal]):
            summary_path = os.path.join(args.output, f"{base_name}_summary.png")
            visualizer.create_performance_summary(data, summary_path)
            print(f"✓ Created performance summary: {summary_path}")
        
        print(f"\n✓ Visualization completed!")
        print(f"Output directory: {args.output}")
        
        try:
            plt.show()
        except:
            print("Note: Interactive display not available, plots saved to files")
        
        return 0
        
    except Exception as e:
        print(f"Error during visualization: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())