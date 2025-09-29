import re
import matplotlib.pyplot as plt

# Path to your txt file
filename = "Replica_size_table.txt"

sizes = []
times = []
gpu_mem = []

# Parse the file
with open(filename, "r") as f:
    for line in f:
        match = re.search(r"size=(\d+): time=(\d+)s, peak_GPU=(\d+)", line)
        if match:
            sizes.append(int(match.group(1)))
            times.append(int(match.group(2)))
            gpu_mem.append(int(match.group(3)))

# Plot
fig, ax1 = plt.subplots()

# Plot execution time
ax1.set_xlabel("Table size")
ax1.set_ylabel("Execution time (s)", color="tab:blue")
ax1.plot(sizes, times, marker="o", color="tab:blue", label="Execution time")
ax1.tick_params(axis="y", labelcolor="tab:blue")

# Add second y-axis for GPU memory
ax2 = ax1.twinx()
ax2.set_ylabel("Peak GPU memory (MB)", color="tab:red")
ax2.plot(sizes, gpu_mem, marker="s", color="tab:red", label="Peak GPU memory")
ax2.tick_params(axis="y", labelcolor="tab:red")

# Title and layout
plt.title("Execution time and Peak GPU memory vs Size for Replica/office0 on RTX4070")
fig.tight_layout()
plt.show()
