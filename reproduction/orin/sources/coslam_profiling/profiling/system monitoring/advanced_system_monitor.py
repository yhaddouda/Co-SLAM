import psutil
import time
import json
import csv
from datetime import datetime

class SLAMPerformanceCollector:
    def __init__(self, target_process="python", collection_interval=0.5):
        print(f"Initializing SLAM performance collector...")
        print(f"Target process pattern: '{target_process}'")
        print(f"Collection interval: {collection_interval} seconds")
        
        self.target_process = target_process
        self.interval = collection_interval
        self.start_time = None
        self.monitoring_active = False
        self.performance_data = []
        self.total_samples = 0
        self.processes_detected = False
        
        # Get system info for reference - psutil.cpu_count() for logical cores
        self.cpu_count = psutil.cpu_count(logical=True)
        self.total_system_memory_mb = psutil.virtual_memory().total / (1024 * 1024)
        
        print(f"System info: {self.cpu_count} CPU cores, {self.total_system_memory_mb:.0f}MB total memory")
        print("✓ Collector initialized successfully")
    
    def find_slam_processes(self):
        matching_processes = []
        
        try:
            # psutil.process_iter() - iterate through all running processes
            for process in psutil.process_iter(['pid', 'name', 'cmdline']):
                process_info = process.info
                process_name = process_info['name'] or ""
                command_line = " ".join(process_info['cmdline'] or [])
                
                if (self.target_process.lower() in process_name.lower() or
                    'coslam' in command_line.lower() or 
                    'slam' in command_line.lower()):
                    matching_processes.append(process)
        
        except Exception as e:
            print(f"Warning: Error during process discovery: {e}")
        
        return matching_processes
    
    def get_gpu_info(self):
        gpu_utilization = 0.0
        gpu_memory_mb = 0.0
        cuda_cores = 0
        
        try:
            # nvidia-smi subprocess call for GPU metrics
            import subprocess
            result = subprocess.run(['nvidia-smi', '--query-gpu=utilization.gpu,memory.used,name', 
                                   '--format=csv,noheader,nounits'], 
                                  capture_output=True, text=True, timeout=2)
            if result.returncode == 0:
                lines = result.stdout.strip().split('\n')
                for line in lines:
                    parts = line.split(', ')
                    if len(parts) >= 2:
                        gpu_utilization += float(parts[0])
                        gpu_memory_mb += float(parts[1])
            
            # Get CUDA cores info - nvidia-ml-py would be better but subprocess works universally
            cuda_result = subprocess.run(['nvidia-smi', '--query-gpu=name', 
                                        '--format=csv,noheader'], 
                                       capture_output=True, text=True, timeout=2)
            if cuda_result.returncode == 0:
                gpu_name = cuda_result.stdout.strip()
                # Rough CUDA core mapping for common GPUs
                cuda_cores_map = {
                    'RTX 4090': 16384, 'RTX 4080': 9728, 'RTX 4070': 5888,
                    'RTX 3090': 10496, 'RTX 3080': 8704, 'RTX 3070': 5888,
                    'RTX 2080': 2944, 'GTX 1080': 2560, 'V100': 5120,
                    'A100': 6912, 'H100': 14592
                }
                for gpu_model, cores in cuda_cores_map.items():
                    if gpu_model in gpu_name:
                        cuda_cores = cores
                        break
                        
        except Exception:
            pass
        
        return gpu_utilization, gpu_memory_mb, cuda_cores
    
    def collect_system_snapshot(self):
        snapshot_time = time.time()
        relative_time = snapshot_time - self.start_time if self.start_time else 0
        
        slam_processes = self.find_slam_processes()
        
        total_cpu_percent = 0.0
        total_memory_mb = 0.0
        core_usage_count = 0
        process_count = len(slam_processes)
        
        if slam_processes:
            if not self.processes_detected:
                print(f"✓ Detected {process_count} SLAM-related processes")
                self.processes_detected = True
            
            for process in slam_processes:
                try:
                    # psutil.Process.cpu_percent() - CPU usage for specific process
                    cpu_percent = process.cpu_percent()
                    total_cpu_percent += cpu_percent
                    
                    # psutil.Process.memory_info() - memory usage in bytes
                    memory_info = process.memory_info()
                    memory_mb = memory_info.rss / (1024 * 1024)
                    total_memory_mb += memory_mb
                    
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue
        
        # psutil.cpu_percent(percpu=True) - per-core CPU usage percentages
        per_core_cpu = psutil.cpu_percent(percpu=True)
        cores_in_use = sum(1 for usage in per_core_cpu if usage > 5.0)  # Cores with >5% usage
        
        # psutil.cpu_percent() - overall system CPU usage
        system_cpu = psutil.cpu_percent()
        
        # psutil.virtual_memory() - system memory statistics
        system_memory = psutil.virtual_memory()
        slam_memory_percent = (total_memory_mb / self.total_system_memory_mb) * 100
        
        # GPU metrics via nvidia-smi
        gpu_utilization, gpu_memory_mb, cuda_cores = self.get_gpu_info()
        
        snapshot = {
            'timestamp': snapshot_time,
            'relative_time_seconds': relative_time,
            'datetime': datetime.fromtimestamp(snapshot_time).isoformat(),
            
            'slam_process_count': process_count,
            'slam_cpu_percent': round(total_cpu_percent, 2),
            'slam_memory_mb': round(total_memory_mb, 2),
            'slam_memory_percent': round(slam_memory_percent, 2),
            
            'cpu_cores_in_use': cores_in_use,
            'cpu_total_cores': self.cpu_count,
            'per_core_cpu_usage': [round(usage, 1) for usage in per_core_cpu],
            
            'system_cpu_percent': round(system_cpu, 2),
            'system_memory_percent': round(system_memory.percent, 2),
            'system_memory_available_mb': round(system_memory.available / (1024 * 1024), 2),
            
            'gpu_utilization_percent': round(gpu_utilization, 2),
            'gpu_memory_mb': round(gpu_memory_mb, 2),
            'cuda_cores': cuda_cores
        }
        
        return snapshot
    
    def monitor_performance(self, duration_seconds=120):
        print(f"\n=== Starting Performance Monitoring ===")
        print(f"Duration: {duration_seconds} seconds ({duration_seconds/60:.1f} minutes)")
        print(f"Expected samples: ~{int(duration_seconds / self.interval)}")
        print("=" * 50)
        
        self.start_time = time.time()
        self.monitoring_active = True
        self.total_samples = 0
        
        try:
            while self.monitoring_active and (time.time() - self.start_time) < duration_seconds:
                snapshot = self.collect_system_snapshot()
                self.performance_data.append(snapshot)
                self.total_samples += 1
                
                elapsed = time.time() - self.start_time
                if self.total_samples % 10 == 0:
                    if snapshot['slam_process_count'] > 0:
                        print(f"Sample {self.total_samples:3d} at {elapsed:5.1f}s: "
                              f"CPU={snapshot['slam_cpu_percent']:5.1f}% | "
                              f"Memory={snapshot['slam_memory_mb']:6.1f}MB ({snapshot['slam_memory_percent']:.1f}%) | "
                              f"Cores={snapshot['cpu_cores_in_use']}/{snapshot['cpu_total_cores']}")
                    else:
                        print(f"Sample {self.total_samples:3d} at {elapsed:5.1f}s: "
                              f"Waiting for SLAM processes...")
                
                time.sleep(self.interval)
                
        except KeyboardInterrupt:
            print(f"\nMonitoring interrupted by user after {self.total_samples} samples")
        except Exception as e:
            print(f"\nMonitoring error: {e}")
        finally:
            self.monitoring_active = False
            
        elapsed_total = time.time() - self.start_time
        print(f"\n=== Monitoring Completed ===")
        print(f"Total duration: {elapsed_total:.1f} seconds")
        print(f"Samples collected: {self.total_samples}")
        print(f"Average sampling rate: {self.total_samples/elapsed_total:.2f} samples/second")
        
        return self.performance_data
    
    def save_data_to_files(self, base_filename="slam_performance"):
        if not self.performance_data:
            print("No data to save!")
            return
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        json_filename = f"{base_filename}_{timestamp}.json"
        with open(json_filename, 'w') as f:
            json.dump({
                'metadata': {
                    'collection_start': self.start_time,
                    'total_samples': self.total_samples,
                    'target_process': self.target_process,
                    'collection_interval': self.interval,
                    'system_cpu_cores': self.cpu_count,
                    'system_memory_mb': self.total_system_memory_mb
                },
                'performance_data': self.performance_data
            }, f, indent=2)
        
        print(f"✓ Saved JSON data: {json_filename}")
        
        csv_filename = f"{base_filename}_{timestamp}.csv"
        if self.performance_data:
            # Flatten per-core data for CSV
            flattened_data = []
            for record in self.performance_data:
                flat_record = record.copy()
                per_core = flat_record.pop('per_core_cpu_usage', [])
                for i, usage in enumerate(per_core):
                    flat_record[f'core_{i}_cpu_percent'] = usage
                flattened_data.append(flat_record)
            
            fieldnames = flattened_data[0].keys()
            with open(csv_filename, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(flattened_data)
        
        print(f"✓ Saved CSV data: {csv_filename}")
        
        self.create_summary_report(f"{base_filename}_{timestamp}_summary.txt")
        
        return json_filename, csv_filename
    
    def create_summary_report(self, filename):
        if not self.performance_data:
            return
        
        slam_samples = [d for d in self.performance_data if d['slam_process_count'] > 0]
        
        with open(filename, 'w') as f:
            f.write("SLAM PERFORMANCE MONITORING SUMMARY\n")
            f.write("=" * 40 + "\n\n")
            
            f.write(f"Monitoring Duration: {self.performance_data[-1]['relative_time_seconds']:.1f} seconds\n")
            f.write(f"Total Samples: {len(self.performance_data)}\n")
            f.write(f"Samples with SLAM Processes: {len(slam_samples)}\n\n")
            
            if slam_samples:
                slam_cpu_values = [d['slam_cpu_percent'] for d in slam_samples]
                slam_memory_mb = [d['slam_memory_mb'] for d in slam_samples]
                slam_memory_pct = [d['slam_memory_percent'] for d in slam_samples]
                cores_used = [d['cpu_cores_in_use'] for d in slam_samples]
                
                f.write("SLAM PROCESS STATISTICS:\n")
                f.write(f"  CPU Usage - Average: {sum(slam_cpu_values)/len(slam_cpu_values):.1f}%, "
                       f"Peak: {max(slam_cpu_values):.1f}%\n")
                f.write(f"  Memory Usage - Average: {sum(slam_memory_mb)/len(slam_memory_mb):.1f}MB "
                       f"({sum(slam_memory_pct)/len(slam_memory_pct):.1f}%), "
                       f"Peak: {max(slam_memory_mb):.1f}MB ({max(slam_memory_pct):.1f}%)\n")
                f.write(f"  CPU Cores Used - Average: {sum(cores_used)/len(cores_used):.1f}, "
                       f"Max: {max(cores_used)}/{self.cpu_count}\n\n")
                
                gpu_util_values = [d['gpu_utilization_percent'] for d in slam_samples if d['gpu_utilization_percent'] > 0]
                if gpu_util_values:
                    cuda_cores = slam_samples[0]['cuda_cores']
                    f.write(f"  GPU Utilization - Average: {sum(gpu_util_values)/len(gpu_util_values):.1f}%, "
                           f"Peak: {max(gpu_util_values):.1f}%\n")
                    if cuda_cores > 0:
                        f.write(f"  CUDA Cores: {cuda_cores}\n")
            else:
                f.write("WARNING: No SLAM processes detected during monitoring!\n")
        
        print(f"✓ Saved summary report: {filename}")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Monitor Co-SLAM performance for research analysis')
    parser.add_argument('--target', default='python', help='Target process name pattern')
    parser.add_argument('--duration', type=int, default=120, help='Monitoring duration in seconds')
    parser.add_argument('--interval', type=float, default=0.5, help='Data collection interval in seconds')
    parser.add_argument('--output', default='slam_performance', help='Base filename for output files')
    
    args = parser.parse_args()
    
    print("Co-SLAM Performance Monitor")
    print("=" * 30)
    
    collector = SLAMPerformanceCollector(
        target_process=args.target,
        collection_interval=args.interval
    )
    
    try:
        performance_data = collector.monitor_performance(args.duration)
        json_file, csv_file = collector.save_data_to_files(args.output)
        
        print(f"\nMonitoring completed successfully!")
        print(f"Data files created:")
        print(f"  JSON: {json_file}")
        print(f"  CSV: {csv_file}")
        
    except Exception as e:
        print(f"Error during monitoring: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()