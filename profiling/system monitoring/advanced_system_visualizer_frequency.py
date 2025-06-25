import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import json
from datetime import datetime
import argparse
import os

class SLAMVisualizationEngine:
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
                color=self.colors['slam_cpu'], linewidth=2, label='SLAM CPU')
        ax1.plot(time_minutes, slam_data['system_cpu_percent'], 
                color=self.colors['system_cpu'], alpha=0.7, label='System CPU')
        ax1.set_ylabel('CPU Usage (%)')
        ax1.set_title('CPU Utilization Over Time')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot cores in use
        if 'cpu_cores_in_use' in slam_data.columns:
            ax2.step(time_minutes, slam_data['cpu_cores_in_use'], 
                    where='post', color=self.colors['cores'], linewidth=2, label='Cores in Use')
            total_cores = slam_data['cpu_total_cores'].iloc[0] if 'cpu_total_cores' in slam_data.columns else 'Unknown'
            ax2.axhline(y=total_cores if isinstance(total_cores, (int, float)) else 0, 
                       color='red', linestyle='--', alpha=0.5, label=f'Total Cores ({total_cores})')
            ax2.set_ylabel('Number of Cores')
            ax2.set_title('CPU Cores Utilization')
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
        
        # Memory usage in MB
        ax1.plot(time_minutes, slam_data['slam_memory_mb'], 
                color=self.colors['slam_memory'], linewidth=2, label='SLAM Memory (MB)')
        ax1.set_ylabel('Memory Usage (MB)')
        ax1.set_title('Memory Usage Over Time (Absolute)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Memory usage as percentage
        if 'slam_memory_percent' in slam_data.columns:
            ax2.plot(time_minutes, slam_data['slam_memory_percent'], 
                    color=self.colors['slam_memory'], linewidth=2, label='SLAM Memory %')
            ax2.plot(time_minutes, slam_data['system_memory_percent'], 
                    color=self.colors['system_memory'], alpha=0.7, label='System Memory %')
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
            print("No GPU data available for visualization")
            return None
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(self.figure_scale * self.base_figsize[0], 
                                                      self.figure_scale * 8), sharex=True)
        
        time_minutes = slam_data['relative_time_seconds'] / 60
        
        # GPU utilization
        ax1.plot(time_minutes, slam_data['gpu_utilization_percent'], 
                color=self.colors['gpu'], linewidth=2, label='GPU Utilization')
        ax1.set_ylabel('GPU Utilization (%)')
        ax1.set_title('GPU Utilization Over Time')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # GPU memory
        if 'gpu_memory_mb' in slam_data.columns:
            ax2.plot(time_minutes, slam_data['gpu_memory_mb'], 
                    color=self.colors['gpu'], linewidth=2, label='GPU Memory')
            ax2.set_ylabel('GPU Memory (MB)')
            ax2.set_title('GPU Memory Usage Over Time')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
        
        ax2.set_xlabel('Time (minutes)')
        
        # Add CUDA cores info if available
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
    
    def create_performance_summary(self, data, save_path=None):
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
    parser.add_argument('--all', action='store_true', help='Create all available plots')
    parser.add_argument('--scale', type=float, default=1.0, help='Figure size scale')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.data_file):
        print(f"Error: Input file '{args.data_file}' not found")
        return 1
    
    os.makedirs(args.output, exist_ok=True)
    
    try:
        visualizer = SLAMVisualizationEngine(figure_size_scale=args.scale)
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
            if gpu_fig:
                print(f"✓ Created GPU analysis: {gpu_path}")
        
        if args.all or args.heatmap:
            heatmap_path = os.path.join(args.output, f"{base_name}_cpu_heatmap.png")
            heatmap_fig = visualizer.create_per_core_heatmap(data, heatmap_path)
            if heatmap_fig:
                print(f"✓ Created CPU heatmap: {heatmap_path}")
        
        if args.all or args.summary or not any([args.cpu, args.memory, args.gpu, args.heatmap]):
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