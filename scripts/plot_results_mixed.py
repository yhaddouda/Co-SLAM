import re
import matplotlib.pyplot as plt

# Path to your txt file
filename = "Replica_size_table_mixed.txt"

# Storage for both precisions
data = {
    "fp16": {"sizes": [], "times": [], "rams": []},
    "fp32": {"sizes": [], "times": [], "rams": []},
}

# Regex matching your exact format
pattern = re.compile(
    r"dtype=(fp16|fp32)\s*,\s*size=(\d+)\s*:\s*time=([\d.]+)s\s*,\s*peak_RAM=(\d+)\s*MB",
    re.IGNORECASE,
)

# Parse the file
with open(filename, "r") as f:
    for line in f:
        m = pattern.search(line)
        if not m:
            continue
        prec = m.group(1).lower()
        size = int(m.group(2))
        time_s = float(m.group(3))
        ram_mb = int(m.group(4))
        data[prec]["sizes"].append(size)
        data[prec]["times"].append(time_s)
        data[prec]["rams"].append(ram_mb)

# Sort each precision's series by size
for prec in data:
    sizes = data[prec]["sizes"]
    order = sorted(range(len(sizes)), key=lambda i: sizes[i])
    data[prec]["sizes"] = [data[prec]["sizes"][i] for i in order]
    data[prec]["times"] = [data[prec]["times"][i] for i in order]
    data[prec]["rams"]  = [data[prec]["rams"][i]  for i in order]

# --- Plot 1: Execution time vs size (fp16 & fp32) ---
plt.figure()
if data["fp16"]["sizes"]:
    plt.plot(data["fp16"]["sizes"], data["fp16"]["times"],
             marker="o", label="FP16 time")
if data["fp32"]["sizes"]:
    plt.plot(data["fp32"]["sizes"], data["fp32"]["times"],
             marker="s", label="FP32 time")
plt.xlabel("Table size")
plt.ylabel("Execution time (s)")
plt.title("Execution time vs Size for Replica/office0 on Orin")
plt.legend()
plt.grid(True, linestyle="--", alpha=0.4)
plt.tight_layout()
plt.savefig("replica_office0_Orin_time.png", dpi=300)

# --- Plot 2: Peak RAM vs size (fp16 & fp32) ---
plt.figure()
if data["fp16"]["sizes"]:
    plt.plot(data["fp16"]["sizes"], data["fp16"]["rams"],
             marker="o", label="FP16 peak RAM")
if data["fp32"]["sizes"]:
    plt.plot(data["fp32"]["sizes"], data["fp32"]["rams"],
             marker="s", label="FP32 peak RAM")
plt.xlabel("Table size")
plt.ylabel("Peak RAM (MB)")
plt.title("Peak RAM vs Size for Replica/office0 on Orin")
plt.legend()
plt.grid(True, linestyle="--", alpha=0.4)
plt.tight_layout()
plt.savefig("replica_office0_Orin_ram.png", dpi=300)
